"""Optional RL → near-goal planner handoff; never used to relabel PPO actions."""

from collections import deque
from dataclasses import asdict, dataclass

import numpy as np

from .local_map import LocalMap, robot_radius
from .paths import connectors, integrate, wrap


@dataclass(frozen=True)
class TerminalConfig:
    enter_distance: float = 2.0
    exit_distance: float = 3.0
    family: str = "auto"
    turning_radius: float = 0.6
    linear_speed: float = 0.25
    angular_speed: float = 0.6
    clearance: float = 0.1
    max_slope: float = 0.6
    max_height_residual: float = 0.16
    replan_steps: int = 10
    map_refresh_steps: int = 5
    retry_steps: int = 10
    max_control_steps: int = 250

    def __post_init__(self):
        values = [
            self.enter_distance,
            self.exit_distance,
            self.turning_radius,
            self.linear_speed,
            self.angular_speed,
            self.clearance,
            self.max_slope,
            self.max_height_residual,
        ]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(
                "Terminal distances, speeds and clearance must be positive"
            )
        if self.exit_distance <= self.enter_distance:
            raise ValueError("Terminal exit distance must exceed entry distance")
        if self.family not in {"auto", "rs", "diff-drive"}:
            raise ValueError("Invalid terminal path family")
        if (
            min(
                self.replan_steps,
                self.map_refresh_steps,
                self.retry_steps,
                self.max_control_steps,
            )
            < 1
        ):
            raise ValueError("Terminal step counts must be positive")


class TerminalController:
    def __init__(self, env, cfg=None):
        self.env, self.cfg = env, cfg or TerminalConfig()
        radius = robot_radius(env.terrain.spec["robot"]["path"])
        self.map = LocalMap(
            env.terrain,
            radius,
            extent=self.cfg.exit_distance + radius + self.cfg.clearance + 1,
            margin=self.cfg.clearance,
            max_slope=self.cfg.max_slope,
            max_height_residual=self.cfg.max_height_residual,
        )
        limits = env.backend.spec["robot"]["limits"]
        self.vmax = min(
            self.cfg.linear_speed,
            limits["max_linear_velocity"],
            -limits["min_linear_velocity"],
        )
        self.wmax = min(
            self.cfg.angular_speed,
            limits["max_angular_velocity"],
            -limits["min_angular_velocity"],
        )
        self.reset()

    def reset(self):
        self.active = False
        self.path = None
        self.segment = 0
        self.braking = False
        self.aligning = False
        self.final_direction = None
        self.parking = False
        self.turn_offset = 0.0
        self.turn_samples = deque(maxlen=15)
        self.previous_action = np.zeros(2)
        self.steps, self.control_steps, self.cooldown = 0, 0, 0
        self.last_plan = self.last_map = -1000000
        self.reason = "rl"
        self.last_failure = None
        self.stats = dict(
            handoffs=0,
            controlled_steps=0,
            plans=0,
            rs_plans=0,
            diff_drive_plans=0,
            planning_failures=0,
            safety_stops=0,
            fallbacks=0,
        )

    def pose(self):
        state = self.env.state
        return np.array(
            [
                *state["position"][:2],
                wrap(state["rpy"][2] + self.env.terrain.spec["heading_offset"]),
            ]
        )

    def goal(self):
        return np.array([*self.env.target[:2], self.env.target_heading])

    def report(self):
        return {
            **self.stats,
            "mode": self.reason,
            "config": asdict(self.cfg),
            "footprint_radius": self.map.radius,
            "last_failure": self.last_failure,
            "estimated_turn_offset": self.turn_offset,
        }

    def _fallback(self, reason):
        if self.active:
            self.stats["fallbacks"] += 1
        self.active, self.path = False, None
        self.cooldown = self.cfg.retry_steps
        self.reason = reason
        # One explicit stop before returning control to RL on the next step.
        return np.zeros(2, dtype=np.float32)

    def _plan(self, pose, goal):
        self.last_plan = self.steps
        for state, name in (
            (pose, "start_not_traversable"),
            (goal, "goal_not_traversable"),
        ):
            if not self.map.valid([state]):
                self.path = None
                self.last_failure = name
                self.stats["planning_failures"] += 1
                return False
        paths = connectors(
            pose,
            goal,
            radius=self.cfg.turning_radius,
            family="diff-drive" if self.parking else self.cfg.family,
            speed=self.vmax,
            yaw_rate=self.wmax,
            turn_offset=self.turn_offset if self.parking else 0,
        )
        self.path = next((p for p in paths if self.map.valid(p.poses)), None)
        self.segment = 0
        if self.path is None:
            self.last_failure = "no_clear_connector"
            self.stats["planning_failures"] += 1
            return False
        self.stats["plans"] += 1
        self.stats["rs_plans" if self.path.family == "rs" else "diff_drive_plans"] += 1
        return True

    def _pose_control(self, pose, goal, direction=None):
        # The native skid-steer rover's base frame translates sideways while
        # turning. Regulate the observed instantaneous turn centre; the target
        # is transformed using its FINAL heading, so final rotation brings the
        # original base frame to the original goal position.
        axle = pose[:2] - self.turn_offset * np.array(
            [np.cos(pose[2]), np.sin(pose[2])]
        )
        axle_goal = goal[:2] - self.turn_offset * np.array(
            [np.cos(goal[2]), np.sin(goal[2])]
        )
        delta = axle_goal - axle
        distance = np.linalg.norm(delta)
        heading_error = wrap(goal[2] - pose[2])
        # Hold inside stricter tolerances so the original lagged success check
        # can observe the pose on the next physics step.
        tolerance = self.env.cfg.success_distance
        if distance < min(0.07, 0.4 * tolerance):
            self.aligning = True
        elif distance > 0.65 * tolerance:
            self.aligning = False
        if self.aligning:
            return np.array([0.0, np.clip(2 * heading_error, -self.wmax, self.wmax)])
        alpha = wrap(np.arctan2(delta[1], delta[0]) - pose[2])
        if direction is None:
            direction = -1 if abs(alpha) > np.pi / 2 else 1
        if direction < 0:
            alpha = wrap(alpha - np.pi)
        if abs(alpha) > 0.2:
            return np.array([0.0, np.clip(2 * alpha, -self.wmax, self.wmax)])
        linear = direction * min(self.vmax, 0.6 * distance) * max(0, np.cos(alpha))
        angular = np.clip(2.0 * alpha, -self.wmax, self.wmax)
        return np.array([linear, angular])

    def _track(self, pose, goal):
        # Full-pose feedback aligns during approach; it also corrects final drift
        # caused by slip while rotating. It is checked against the same local map.
        if self.parking:
            return self._pose_control(pose, goal)
        while self.segment < len(self.path.segments):
            seg = self.path.segments[self.segment]
            endpoint = self.path.endpoints[self.segment]
            if abs(seg.rotation) > 0:
                error = wrap(endpoint[2] - pose[2])
                if abs(error) < 0.07:
                    self.segment += 1
                    continue
                return np.array([0, np.clip(2 * error, -self.wmax, self.wmax)])
            distance = np.linalg.norm(endpoint[:2] - pose[:2])
            if distance < 0.1:
                self.segment += 1
                continue
            origin = (
                self.path.poses[0]
                if self.segment == 0
                else self.path.endpoints[self.segment - 1]
            )
            ts = np.linspace(0, 1, max(2, int(abs(seg.length) / 0.04) + 1))
            samples = np.stack(
                [integrate(origin, seg.length * t, seg.curvature) for t in ts]
            )
            score = np.linalg.norm(samples[:, :2] - pose[:2], axis=1)
            score += 0.05 * np.abs(wrap(samples[:, 2] - pose[2]))
            nearest = int(score.argmin())
            ahead = min(len(ts) - 1, nearest + 5)
            target = samples[ahead]
            direction = np.sign(seg.length)
            bearing = wrap(
                np.arctan2(target[1] - pose[1], target[0] - pose[0]) - pose[2]
            )
            if direction < 0:
                bearing = wrap(bearing - np.pi)
            v = direction * min(self.vmax, 0.7 * distance) * max(0, np.cos(bearing))
            w = np.clip(seg.curvature * v + 2 * bearing, -self.wmax, self.wmax)
            return np.array([v, w])
        return self._pose_control(pose, goal)

    def action(self, rl_action):
        state = self.env.state
        yaw = self.pose()[2]
        vx, vy = (
            np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])
            @ np.asarray(state["linear_velocity"])[:2]
        )
        w = state["angular_velocity"][2]
        if abs(self.previous_action[0]) < 0.01 and abs(vx) < 0.05 and abs(w) > 0.08:
            estimate = vy / w
            if abs(estimate) < min(0.8, self.map.radius):
                self.turn_samples.append(estimate)
                if len(self.turn_samples) >= 4:
                    self.turn_offset = 0.8 * self.turn_offset + 0.2 * float(
                        np.median(self.turn_samples)
                    )
        action = self._action(rl_action)
        self.previous_action = np.asarray(action).copy()
        return action

    def _action(self, rl_action):
        self.steps += 1
        pose, goal = self.pose(), self.goal()
        distance = np.linalg.norm(goal[:2] - pose[:2])
        if distance < 0.6 and not self.parking:
            self.parking = True
            self.path = None
        elif distance > 0.9 and self.parking:
            self.parking = False
            self.aligning = False
            self.path = None
        if self.active and distance > self.cfg.exit_distance:
            return self._fallback("outside_exit_distance")
        if not self.active:
            if self.cooldown > 0:
                self.cooldown -= 1
                self.reason = "rl_cooldown"
                return np.asarray(rl_action, dtype=np.float32)
            if distance > self.cfg.enter_distance:
                self.reason = "rl"
                return np.asarray(rl_action, dtype=np.float32)
        # Refresh native rock poses and terrain checks periodically, before any
        # newly selected path is executed. Recenter before leaving mapped cells.
        if (
            self.steps - self.last_map >= self.cfg.map_refresh_steps
            or self.map.center is None
            or np.linalg.norm(pose[:2] - self.map.center) > 0.5
            or self.map.objects_changed()
        ):
            self.map.refresh(pose)
            self.last_map = self.steps
        if self.path is None or self.steps - self.last_plan >= self.cfg.replan_steps:
            if not self._plan(pose, goal):
                if self.active:
                    return self._fallback("no_valid_path")
                # A rejected handoff must not modify the pure-RL action stream.
                self.cooldown = self.cfg.retry_steps
                self.reason = "rl_no_valid_path"
                return np.asarray(rl_action, dtype=np.float32)
            if not self.active:
                self.active = True
                self.braking = True
                self.control_steps = 0
                self.stats["handoffs"] += 1
        self.control_steps += 1
        if self.control_steps > self.cfg.max_control_steps:
            return self._fallback("terminal_timeout")
        action = self._track(pose, goal)
        # Check a short swept arc in addition to the full planned path: the
        # feedback correction may leave its nominal centreline. Include measured
        # velocity so a stop is not mistaken for instantaneous braking.
        sweep = [pose]
        state = self.env.state
        measured_v = np.dot(
            np.asarray(state["linear_velocity"])[:2], [np.cos(pose[2]), np.sin(pose[2])]
        )
        measured_w = state["angular_velocity"][2]
        if self.braking:
            if (
                np.linalg.norm(state["linear_velocity"][:2]) > 0.1
                or abs(measured_w) > 0.2
            ):
                action = np.zeros(2)
                self.last_plan = -1000000  # Recompute from the pose after braking.
            else:
                self.braking = False
        for v, w in (action, (measured_v, measured_w)):
            for t in np.linspace(0.1, 1.0, 10):
                sweep.append(
                    integrate(
                        pose,
                        v * t,
                        w / v if abs(v) > 1e-8 else 0,
                        w * t if abs(v) <= 1e-8 else 0,
                    )
                )
        for t in np.linspace(0.1, 1.0, 10):
            sweep.append(
                np.r_[
                    pose[:2] + np.asarray(state["linear_velocity"])[:2] * t,
                    wrap(pose[2] + measured_w * t),
                ]
            )
        if not self.map.valid(np.stack(sweep)):
            self.stats["safety_stops"] += 1
            return self._fallback("unsafe_tracking_arc")
        self.reason = (
            "terminal_braking" if self.braking else "terminal_" + self.path.family
        )
        self.stats["controlled_steps"] += 1
        return action.astype(np.float32)


def add_terminal_arguments(parser):
    parser.add_argument(
        "--terminal-planner",
        action="store_true",
        help="Near-goal planning and feedback tracking; hybrid system score",
    )
    parser.add_argument("--terminal-distance", type=float, default=2.0)
    parser.add_argument("--terminal-exit-distance", type=float, default=3.0)
    parser.add_argument(
        "--terminal-family", choices=["auto", "rs", "diff-drive"], default="auto"
    )
    parser.add_argument("--terminal-radius", type=float, default=0.6)
    parser.add_argument(
        "--terminal-max-slope",
        type=float,
        default=0.6,
        help="Maximum smoothed ground slope, rise/run",
    )
    parser.add_argument(
        "--terminal-height-residual",
        type=float,
        default=0.16,
        help="Maximum raw ground deviation from smoothed surface, metres",
    )


def terminal_config(args):
    if not args.terminal_planner:
        return None
    return TerminalConfig(
        enter_distance=args.terminal_distance,
        exit_distance=args.terminal_exit_distance,
        family=args.terminal_family,
        turning_radius=args.terminal_radius,
        max_slope=args.terminal_max_slope,
        max_height_residual=args.terminal_height_residual,
    )
