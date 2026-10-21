"""Full-pose connectors: Reeds–Shepp and native differential-drive primitives."""

from dataclasses import dataclass

import numpy as np

from . import _rs_formulae as rs


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Segment:
    # Signed translation in metres and curvature in 1/metres; rotation in radians.
    length: float = 0.0
    curvature: float = 0.0
    rotation: float = 0.0


@dataclass
class Path:
    family: str
    segments: list[Segment]
    poses: np.ndarray
    endpoints: list[np.ndarray]
    cost: float


def integrate(pose, length, curvature=0.0, rotation=0.0):
    x, y, yaw = pose
    if abs(curvature) < 1e-10:
        x += length * np.cos(yaw)
        y += length * np.sin(yaw)
    else:
        angle = length * curvature
        x += (np.sin(yaw + angle) - np.sin(yaw)) / curvature
        y += (np.cos(yaw) - np.cos(yaw + angle)) / curvature
        yaw += angle
    return np.array([x, y, wrap(yaw + rotation)], dtype=float)


def make_path(start, segments, family, *, spacing=0.05, speed=0.25, yaw_rate=0.6):
    segments = [s for s in segments if abs(s.length) > 1e-8 or abs(s.rotation) > 1e-8]
    poses, endpoints = [np.asarray(start, dtype=float)], []
    cost, previous_direction = 0.0, 0
    for seg in segments:
        origin = poses[-1]
        n = max(
            1,
            int(np.ceil(abs(seg.length) / spacing)),
            int(np.ceil(abs(seg.rotation) / 0.08)),
        )
        for t in np.linspace(0, 1, n + 1)[1:]:
            poses.append(
                integrate(origin, seg.length * t, seg.curvature, seg.rotation * t)
            )
        endpoints.append(poses[-1])
        cost += max(
            abs(seg.length) / speed,
            abs(seg.length * seg.curvature + seg.rotation) / yaw_rate,
        )
        direction = np.sign(seg.length)
        if direction:
            cost += 0.1 * abs(seg.length) / speed * (direction < 0)
            if previous_direction and direction != previous_direction:
                cost += 1.0  # Stop before switching between forward and reverse.
            previous_direction = direction
    return Path(family, segments, np.stack(poses), endpoints, float(cost))


def connectors(
    start, goal, *, radius=0.6, family="auto", speed=0.25, yaw_rate=0.6, turn_offset=0.0
):
    """Enumerate analytic connectors. Collision checking is the caller's job."""
    if family not in {"auto", "rs", "diff-drive"} or radius <= 0:
        raise ValueError("Invalid connector family or turning radius")
    start, goal = np.asarray(start, dtype=float), np.asarray(goal, dtype=float)
    if abs(turn_offset) > 1e-8:
        axle_start, axle_goal = start.copy(), goal.copy()
        axle_start[:2] -= turn_offset * np.array([np.cos(start[2]), np.sin(start[2])])
        axle_goal[:2] -= turn_offset * np.array([np.cos(goal[2]), np.sin(goal[2])])
        paths = connectors(
            axle_start,
            axle_goal,
            radius=radius,
            family=family,
            speed=speed,
            yaw_rate=yaw_rate,
        )

        def body(poses):
            poses = np.asarray(poses).copy()
            poses[..., :2] += turn_offset * np.stack(
                [np.cos(poses[..., 2]), np.sin(poses[..., 2])], axis=-1
            )
            return poses

        for path in paths:
            path.poses = body(path.poses)
            path.endpoints = [body(p) for p in path.endpoints]
        return paths
    kwargs = dict(speed=speed, yaw_rate=yaw_rate)
    candidates = []
    delta = goal[:2] - start[:2]
    distance = np.linalg.norm(delta)
    if family in {"auto", "diff-drive"}:
        for direction in (1, -1):
            heading = (
                np.arctan2(delta[1], delta[0]) + (np.pi if direction < 0 else 0)
                if distance > 1e-10
                else start[2]
            )
            segments = [
                Segment(rotation=wrap(heading - start[2])),
                Segment(length=direction * distance),
                Segment(rotation=wrap(goal[2] - heading)),
            ]
            candidates.append(make_path(start, segments, "diff-drive", **kwargs))
    if family in {"auto", "rs"}:
        c, s = np.cos(start[2]), np.sin(start[2])
        x, y = np.array([[c, s], [-s, c]]) @ delta / radius
        phi = wrap(goal[2] - start[2])
        functions = [
            rs.left_straight_left,
            rs.left_straight_right,
            rs.left_x_right_x_left,
            rs.left_x_right_left,
            rs.left_right_x_left,
            rs.left_right_x_left_right,
            rs.left_x_right_left_x_right,
            rs.left_x_right90_straight_left,
            rs.left_x_right90_straight_right,
            rs.left_straight_right90_x_left,
            rs.left_straight_left90_x_right,
            rs.left_x_right90_straight_left90_x_right,
        ]
        for fn in functions:
            for time_sign, reflection in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                try:
                    valid, lengths, modes = fn(
                        time_sign * x, reflection * y, time_sign * reflection * phi
                    )
                except (ValueError, ZeroDivisionError, FloatingPointError):
                    continue  # Singular families do not remove other valid solutions.
                if not valid or not np.isfinite(lengths).all():
                    continue
                segments = [
                    Segment(
                        length=float(length) * radius * time_sign,
                        curvature={"L": 1, "S": 0, "R": -1}[mode] * reflection / radius,
                    )
                    for length, mode in zip(lengths, modes)
                ]
                path = make_path(start, segments, "rs", **kwargs)
                error = path.poses[-1] - goal
                error[2] = wrap(error[2])
                if np.linalg.norm(error) < 1e-6:
                    candidates.append(path)
        if np.linalg.norm(delta) < 1e-9 and abs(phi) < 1e-9:
            candidates.append(make_path(start, [], "rs", **kwargs))
    return sorted(candidates, key=lambda p: p.cost)
