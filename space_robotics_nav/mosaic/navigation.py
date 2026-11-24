"""Replaceable A* global planner and conservative local trajectory controller."""

import heapq
import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.ndimage import label

from .contracts import ArrivalRegion, GridMap, Localization, NavigationFeedback, Pose
from .maps import footprint_clearances, segment_safe, traversable, with_local_occupancy


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


class GlobalPlanner(Protocol):
    def plan(
        self, start: Pose, goal: Pose, grid: GridMap, radius: float
    ) -> list[Pose] | None:
        ...


def planning_start(start, grid, radius, free=None):
    """Connect a safe physical stop to the inflated planning grid, within 1m.

    A stop may drift into the grid's extra rounding/tracking allowance. Only a
    short, continuously footprint-checked connection is permitted; real blocked
    footprints and connections crossing obstacles remain invalid.
    """
    cell = grid.cell(start)
    if not grid.contains(cell):
        return None
    free = traversable(grid, radius) if free is None else free
    if free[cell[1], cell[0]]:
        return cell
    minimum = radius + 0.05
    if footprint_clearances(grid, [[start.x, start.y]], minimum + 0.01)[0] < minimum:
        return None
    yy, xx = np.nonzero(free)
    xy = np.c_[
        grid.origin[0] + xx * grid.resolution, grid.origin[1] + yy * grid.resolution
    ]
    distances = np.linalg.norm(xy - [start.x, start.y], axis=1)
    for index in np.argsort(distances):
        if distances[index] > 1.0:
            break
        point = Pose(*xy[index], frame=grid.frame)
        if segment_safe(start, point, grid, radius):
            return int(xx[index]), int(yy[index])
    return None


class AStarPlanner:
    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint

    def plan(self, start, goal, grid, radius):
        b = grid.cell(goal)
        free = traversable(grid, radius)
        if not grid.contains(b) or not free[b[1], b[0]]:
            return None
        a = planning_start(start, grid, radius, free)
        if a is None:
            return None
        # A diagonal cannot cut a blocked corner, so this graph has the same
        # components as four-connected free cells. Reject disconnected station
        # bids before exhaustively searching the entire larger open terrain.
        components, _ = label(free)
        if components[a[1], a[0]] != components[b[1], b[0]]:
            return None
        queue, costs, parent = [(0.0, a)], {a: 0.0}, {}
        closed = set()
        while queue:
            if self.checkpoint is not None and len(closed) % 128 == 0:
                self.checkpoint()
            _, current = heapq.heappop(queue)
            if current in closed:
                continue
            if current == b:
                cells = [b]
                while cells[-1] != a:
                    cells.append(parent[cells[-1]])
                path = [
                    Pose(
                        grid.origin[0] + x * grid.resolution,
                        grid.origin[1] + y * grid.resolution,
                        frame=grid.frame,
                    )
                    for x, y in reversed(cells)
                ]
                if a != grid.cell(start):
                    path.insert(0, start)
                    if len(path) == 2:
                        path.append(goal)
                    else:
                        path[-1] = goal
                else:
                    path[-1] = goal
                return path
            closed.add(current)
            for dx, dy in (
                (1, 0),
                (-1, 0),
                (0, 1),
                (0, -1),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                nxt = (current[0] + dx, current[1] + dy)
                if not grid.contains(nxt) or not free[nxt[1], nxt[0]]:
                    continue
                if (
                    dx
                    and dy
                    and not (free[current[1], nxt[0]] and free[nxt[1], current[0]])
                ):
                    continue
                cost = costs[current] + math.hypot(dx, dy)
                if cost < costs.get(nxt, math.inf):
                    costs[nxt], parent[nxt] = cost, current
                    heapq.heappush(queue, (cost + math.dist(nxt, b), nxt))
        return None


class LocalController(Protocol):
    def command(
        self, state: Localization, waypoint: Pose, goal: Pose, grid: GridMap, robot
    ) -> tuple[float, float]:
        ...


class PathTracker:
    def __init__(self):
        self.goal = None
        self.aligning = False
        self.recovering = self.recovery_finished = False

    def _escape_padding(self, state, waypoint, grid, robot):
        minimum = robot.radius + 0.05
        clearance = footprint_clearances(
            grid, [[state.pose.x, state.pose.y]], robot.radius + 2
        )[0]
        cell = grid.cell(state.pose)
        in_padding = (
            not grid.contains(cell)
            or not traversable(grid, robot.radius)[cell[1], cell[0]]
        )
        if not (self.recovering or in_padding):
            return None
        self.recovering = True
        if clearance < minimum:
            return (0.0, 0.0)
        if (
            not in_padding
            and clearance >= robot.radius + 2 * abs(robot.turn_offset) + 0.1
        ):
            self.recovering, self.recovery_finished = False, True
            return (0.0, 0.0)
        speed = min(0.08, robot.max_linear)
        angular = min(0.2, robot.max_angular)
        best = None
        for v in (-speed, 0.0, speed) if robot.allow_reverse else (0.0, speed):
            for w in (-angular, 0.0, angular):
                if v == w == 0:
                    continue
                points = self._rollout(state, (v, w), robot.turn_offset)
                clear = footprint_clearances(grid, points, robot.radius + 2)
                gain = clear[-1] - clearance
                if clear.min() < minimum or gain < 0.005:
                    continue
                score = (gain, -np.linalg.norm(points[-1] - [waypoint.x, waypoint.y]))
                if best is None or score > best[0]:
                    best = score, (v, w)
        if best is not None:
            return best[1]
        if not in_padding:
            self.recovering, self.recovery_finished = False, True
        return (0.0, 0.0)

    def command(self, state, waypoint, goal, grid, robot):
        self.recovery_finished = False
        escape = self._escape_padding(state, waypoint, grid, robot)
        if escape is not None:
            return escape
        pose = state.pose
        distance = pose.distance(goal)
        if self.goal != goal:
            self.goal, self.aligning = goal, False
        if distance < 1.0:
            # The source rover's base_link is forward of its turn centre. Use
            # the configured kinematic offset so final rotation does not push
            # the base out of the arrival tolerance and cause repeated turns.
            x = pose.x - robot.turn_offset * math.cos(pose.yaw)
            y = pose.y - robot.turn_offset * math.sin(pose.yaw)
            gx = goal.x - robot.turn_offset * math.cos(goal.yaw)
            gy = goal.y - robot.turn_offset * math.sin(goal.yaw)
            d = math.hypot(gx - x, gy - y)
            if d < max(0.10, robot.position_tolerance * 0.5):
                self.aligning = True
            elif d > max(0.22, robot.position_tolerance * 0.85):
                self.aligning = False
            if self.aligning:
                v, w = 0.0, float(
                    np.clip(
                        1.2 * wrap(goal.yaw - pose.yaw),
                        -robot.max_angular,
                        robot.max_angular,
                    )
                )
            else:
                error = wrap(math.atan2(gy - y, gx - x) - pose.yaw)
                direction = (
                    -1 if robot.allow_reverse and abs(error) > math.pi / 2 else 1
                )
                if direction < 0:
                    error = wrap(error - math.pi)
                v = (
                    direction
                    * min(robot.max_linear, d * 0.6)
                    * max(0.0, math.cos(error))
                )
                if abs(error) > 0.25:
                    v = 0.0
                w = float(np.clip(1.2 * error, -robot.max_angular, robot.max_angular))
        else:
            error = wrap(
                math.atan2(waypoint.y - pose.y, waypoint.x - pose.x) - pose.yaw
            )
            direction = -1 if robot.allow_reverse and abs(error) > math.pi / 2 else 1
            if direction < 0:
                error = wrap(error - math.pi)
            v = (
                direction
                * min(robot.max_linear, distance * 0.5)
                * max(0.0, math.cos(error))
            )
            if abs(error) > 0.6:
                v = 0.0
            w = float(np.clip(1.5 * error, -robot.max_angular, robot.max_angular))
        w = float(
            np.clip(
                w - robot.yaw_rate_damping * state.yaw_rate,
                -robot.max_angular,
                robot.max_angular,
            )
        )
        # Slow down near tight corners before waiting/replanning. A coarse
        # path and an offset base frame cannot always track at full speed.
        candidates = (
            (v, w),
            (v * 0.5, w * 0.5),
            (v * 0.2, w * 0.2),
            (0.0, w),
            (v * 0.2, 0.0),
        )
        for candidate in candidates:
            if self.safe(state, candidate, grid, robot.radius, robot.turn_offset):
                return float(candidate[0]), float(candidate[1])
        return (0.0, 0.0)

    @staticmethod
    def _rollout(state, velocity, turn_offset):
        x, y, yaw = state.pose.x, state.pose.y, state.pose.yaw
        axle_x = x - turn_offset * math.cos(yaw)
        axle_y = y - turn_offset * math.sin(yaw)
        # Include current speed: a stop request does not instantly stop physics.
        speed = max(abs(velocity[0]), abs(state.linear_speed))
        direction = (
            -1
            if velocity[0] < 0 or (velocity[0] == 0 and state.linear_speed < 0)
            else 1
        )
        points = []
        for _ in range(21):
            points.append([x, y])
            axle_x += direction * speed * math.cos(yaw) * 0.1
            axle_y += direction * speed * math.sin(yaw) * 0.1
            yaw += velocity[1] * 0.1
            x = axle_x + turn_offset * math.cos(yaw)
            y = axle_y + turn_offset * math.sin(yaw)
        return np.asarray(points)

    @staticmethod
    def safe(state, velocity, grid, radius, turn_offset=0.0):
        if state.pose.frame != grid.frame:
            return False
        points = PathTracker._rollout(state, velocity, turn_offset)
        return bool(
            np.all(footprint_clearances(grid, points, radius + 0.06) >= radius + 0.05)
        )


@dataclass
class GridNavigator:
    robot: object
    global_maps: object
    local_maps: object
    planner: GlobalPlanner
    controller: LocalController
    timeout: float = 180.0
    tracking_margin: float = 0.4
    replan_wait_timeout: float = 45.0

    def __post_init__(self):
        self.cancel()

    def start(self, goal, now, *, arrival_region=None, require_heading=True):
        if (
            arrival_region is not None
            and goal.distance(arrival_region.center) > arrival_region.radius
        ):
            raise ValueError("Navigation station must lie inside its arrival region")
        self.goal, self.started = goal, now
        self.navigation_budget, self.budget_planned = self.timeout, False
        self.arrival_region, self.require_heading = arrival_region, require_heading
        self.path, self.map_version, self.index = None, None, 0
        self.blocked_since = None
        self.last_local_replan = now
        self.local_replans = 0
        self.path_wait_since, self.waiting_reason = None, ""

    def cancel(self):
        self.goal, self.path = None, None
        self.path_wait_since, self.waiting_reason = None, ""

    def tick(self, state, now):
        if self.goal is None:
            return NavigationFeedback("failed", reason="no_goal")
        if not state.usable(now):
            return NavigationFeedback("failed", reason="localization_unavailable")
        global_grid = self.global_maps.snapshot(now)
        if state.pose.frame != self.goal.frame or state.pose.frame != global_grid.frame:
            return NavigationFeedback("failed", reason="frame_mismatch")
        local_grid = self.local_maps.snapshot(self.robot, state.pose, now)
        if not global_grid.usable(now) or not local_grid.usable(now):
            return NavigationFeedback("failed", reason="map_unavailable")
        if (
            state.pose.frame != self.goal.frame
            or state.pose.frame != global_grid.frame
            or local_grid.frame != global_grid.frame
        ):
            return NavigationFeedback("failed", reason="frame_mismatch")
        if now - self.started > self.navigation_budget:
            return NavigationFeedback("failed", reason="navigation_timeout")
        region = self.arrival_region or ArrivalRegion(
            self.goal, self.robot.position_tolerance
        )
        arrived = state.pose.distance(region.center) <= region.radius and (
            not self.require_heading or abs(wrap(self.goal.yaw - state.pose.yaw)) < 0.16
        )
        if arrived and PathTracker.safe(
            state, (0.0, 0.0), local_grid, self.robot.radius, self.robot.turn_offset
        ):
            self.waiting_reason = "Stopping inside arrival region"
            if state.speed < 0.03 and abs(state.yaw_rate) < 0.08:
                self.waiting_reason = ""
                return NavigationFeedback("succeeded")
            return NavigationFeedback("running")
        if self.path is None or self.map_version != global_grid.version:
            self.path = self.planner.plan(
                state.pose,
                self.goal,
                global_grid,
                self.robot.radius + self.tracking_margin,
            )
            if self.path is None:
                # A tracking margin is a preference. A map refresh must not
                # strand a physically valid robot just inside that extra band.
                self.path = self.planner.plan(
                    state.pose, self.goal, global_grid, self.robot.radius
                )
            self.map_version, self.index = global_grid.version, 0
            if self.path is None:
                # A live map may temporarily block a previously valid station.
                # Keep the claim, stop, and replan before treating this as a
                # failed task. Persistent blockage still has a bounded timeout.
                if self.path_wait_since is None:
                    self.path_wait_since = now
                self.waiting_reason = "Waiting for a collision-free global path"
                if now - self.path_wait_since > self.replan_wait_timeout:
                    return NavigationFeedback("failed", reason="no_global_path")
                return NavigationFeedback("running", reason="waiting_for_global_path")
            self.path_wait_since, self.waiting_reason = None, ""
            if not self.budget_planned:
                # A 0.25 m/s rover needs over 180 s for a 45 m route, even
                # before turns and settling. Budget the first valid route once;
                # replans must not keep extending a stalled task indefinitely.
                length = sum(a.distance(b) for a, b in zip(self.path, self.path[1:]))
                self.navigation_budget = max(
                    self.timeout, 1.6 * length / self.robot.max_linear + 60.0
                )
                self.budget_planned = True
        blocked_ahead = False
        while (
            self.index < len(self.path) - 1
            and state.pose.distance(self.path[self.index]) < 0.6
        ):
            if not segment_safe(
                state.pose, self.path[self.index + 1], local_grid, self.robot.radius
            ):
                blocked_ahead = True
                break
            self.index += 1
        tracking_goal = (
            self.goal
            if self.require_heading
            else Pose(self.goal.x, self.goal.y, state.pose.yaw, self.goal.frame)
        )
        goal_cell = local_grid.cell(self.goal)
        occupied_goal = (
            local_grid.contains(goal_cell)
            and local_grid.blocked[goal_cell[1], goal_cell[0]]
            and state.pose.distance(self.goal) < 2 * self.robot.radius + 0.6
        )
        if occupied_goal:
            # Near an actually occupied station, wait for its occupant rather
            # than starting an unrelated retreat. Distant approaches continue.
            if hasattr(self.controller, "recovering"):
                self.controller.recovering = False
                self.controller.recovery_finished = False
            command = (0.0, 0.0)
        else:
            command = self.controller.command(
                state, self.path[self.index], tracking_goal, local_grid, self.robot
            )
        if getattr(self.controller, "recovery_finished", False):
            self.path = None
            self.blocked_since = None
            self.waiting_reason = "Replanning after leaving clearance padding"
            return NavigationFeedback("running")
        stopped = abs(command[0]) < 0.005 and abs(command[1]) < 0.01
        if (blocked_ahead or stopped) and now - self.last_local_replan >= 2:
            alternative = self.planner.plan(
                state.pose,
                self.goal,
                with_local_occupancy(global_grid, local_grid),
                self.robot.radius,
            )
            self.last_local_replan = now
            if alternative is not None:
                self.path, self.index = alternative, 0
                self.local_replans += 1
                command = (0.0, 0.0)
        if abs(command[0]) < 0.005 and abs(command[1]) < 0.01:
            command = (0.0, 0.0)
            self.waiting_reason = "Waiting for a collision-free local path"
            if self.blocked_since is None:
                self.blocked_since = now
            if now - self.blocked_since > 30:
                return NavigationFeedback("failed", reason="local_path_blocked")
        else:
            self.blocked_since = None
            self.waiting_reason = ""
        return NavigationFeedback("running", command)
