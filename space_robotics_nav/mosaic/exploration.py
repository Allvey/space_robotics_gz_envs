"""Replaceable exploration task source using the currently available layout map.

This samples investigation stations; it is not online mapping or a sensor
coverage estimator. A frontier-based source can implement the same interface.
"""

import math

import numpy as np

from .contracts import Kind, Pose
from .maps import traversable, with_local_occupancy


class LayoutExploration:
    def __init__(
        self, planner, local_maps, localization, robots, *, spacing=6.0, clearance=0.6
    ):
        if (
            not math.isfinite(spacing)
            or spacing <= 0
            or not math.isfinite(clearance)
            or clearance < 0
        ):
            raise ValueError(
                "Exploration spacing must be positive and clearance nonnegative"
            )
        self.planner, self.local_maps = planner, local_maps
        self.localization, self.robots = localization, robots
        self.spacing, self.clearance = spacing, clearance
        self.state = "ready"
        self.remaining_sites, self.generated = 0, 0

    def snapshot(self):
        return dict(
            enabled=True,
            source="known_layout_stations",
            state=self.state,
            remaining_sites=self.remaining_sites,
            generated=self.generated,
            spacing=self.spacing,
        )

    def propose(self, robot, state, grid, pois, now):
        if not state.usable(now) or not grid.usable(now):
            self.state = "waiting_for_map_or_localization"
            return None
        x0, y0 = grid.origin
        x1 = x0 + (grid.blocked.shape[1] - 1) * grid.resolution
        y1 = y0 + (grid.blocked.shape[0] - 1) * grid.resolution
        free = traversable(grid, robot.radius + self.clearance)
        reserved = [
            p.approach
            for p in pois
            if p.kind == Kind.EXPLORE and p.approach.frame == grid.frame
        ]
        candidates = []
        for y in np.arange(y0 + self.spacing / 2, y1, self.spacing):
            for x in np.arange(x0 + self.spacing / 2, x1, self.spacing):
                pose = Pose(float(x), float(y), frame=grid.frame)
                cx, cy = grid.cell(pose)
                if not free[cy, cx] or any(
                    pose.distance(p) < self.spacing * 0.8 for p in reserved
                ):
                    continue
                candidates.append(pose)
        self.remaining_sites = len(candidates)
        if not candidates:
            self.state = "exhausted"
            return None
        local = self.local_maps.snapshot(robot, state.pose, now)
        if not local.usable(now):
            self.state = "waiting_for_local_map"
            return None
        planning_grid = with_local_occupancy(grid, local)
        for pose in sorted(candidates, key=state.pose.distance):
            if any(
                peer.name != robot.name
                and pose.distance(self.localization.get(peer.name).pose)
                < robot.radius
                + peer.radius
                + 0.8
                + robot.position_tolerance
                + 2 * abs(robot.turn_offset)
                for peer in self.robots.values()
            ):
                continue
            path = self.planner.plan(
                state.pose, pose, planning_grid, robot.radius + self.clearance
            )
            if path is None and self.clearance > 0:
                path = self.planner.plan(state.pose, pose, planning_grid, robot.radius)
            if path is not None:
                self.state = "active"
                self.generated += 1
                self.remaining_sites -= 1
                return pose
        self.state = "temporarily_unreachable"
        return None
