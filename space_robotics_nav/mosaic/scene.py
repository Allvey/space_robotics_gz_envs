"""Known navigation maps derived from the repository's actual SDF collision assets."""

from dataclasses import replace
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import label, map_coordinates

from ..config import ROOT
from ..world import describe, transform
from .contracts import GridMap, measurement_station_radius
from .maps import KnownMap, rectangle_mask
from .traversability import parameters, surface


class ProjectMap(KnownMap):
    """Explicit known-scene adapter, not online sensor mapping.

    Uses the same transformed terrain mesh and rock collision bounds as Gazebo.
    Dynamic rock poses can be refreshed by the simulation adapter after settling.
    Robot occupancy remains in the separate local-map provider.
    """

    def __init__(self, config):
        from ..terrain import geometry
        from ..planning.local_map import corners

        self.world = config["world"]
        self.terrain_parameters = parameters(config["map"])
        self.assets = str(Path(config.get("assets", ROOT / "assets/cache")).resolve())
        self.spec = describe(self.world, self.assets)
        ground, self.rays, self.objects = geometry(self.world, self.assets)
        bounds = config["map"].get(
            "bounds", [*ground.bounds[0, :2], *ground.bounds[1, :2]]
        )
        super().__init__(bounds, config["map"]["resolution"])
        self.xy = self.coordinates()
        self.heights = self.rays.heights(self.xy).reshape(self.grid.blocked.shape)
        support, self.terrain_blocked = surface(
            self.heights, self.grid.resolution, config["map"]
        )
        self.dy, self.dx = np.gradient(support, self.grid.resolution)
        self.poses = {name: list(obj["pose"]) for name, obj in self.objects.items()}
        self.object_corners = {
            name: corners(obj["bounds"]) for name, obj in self.objects.items()
        }
        self.manual_blocks = np.zeros_like(self.terrain_blocked)
        self._rebuild()

    def coordinates(self):
        g = self.grid
        x = g.origin[0] + np.arange(g.blocked.shape[1]) * g.resolution
        y = g.origin[1] + np.arange(g.blocked.shape[0]) * g.resolution
        xx, yy = np.meshgrid(x, y)
        return np.column_stack([xx.ravel(), yy.ravel()])

    def _rebuild(self):
        blocked = self.terrain_blocked.copy()
        self.traversable_objects = set()
        for name, points in self.object_corners.items():
            matrix = transform(" ".join(map(str, self.poses[name])))
            world = points @ matrix[:3, :3].T + matrix[:3, 3]
            try:
                ground = self.height(*self.poses[name][:2])
            except ValueError:
                ground = -math.inf
            if (
                float(world[:, 2].max()) - ground
                <= self.terrain_parameters["max_step_height"]
            ):
                self.traversable_objects.add(name)
                continue
            lo, hi = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
            blocked |= rectangle_mask(
                self.grid, [*(lo - self.grid.resolution), *(hi + self.grid.resolution)]
            )
        self.grid = replace(
            self.grid,
            blocked=blocked | self.manual_blocks,
            version=self.grid.version + 1,
            source=f"project_collision_geometry:{self.world}",
        )

    def ingest_objects(self, poses):
        changed = False
        for name in self.objects:
            if (
                name in poses
                and np.linalg.norm(np.asarray(poses[name]) - self.poses[name]) > 0.03
            ):
                self.poses[name] = list(poses[name])
                changed = True
        if changed:
            self._rebuild()

    def block(self, bounds, blocked=True):
        self.manual_blocks[rectangle_mask(self.grid, bounds)] = blocked
        self._rebuild()

    def _sample(self, field, x, y):
        g = self.grid
        cell = [(y - g.origin[1]) / g.resolution, (x - g.origin[0]) / g.resolution]
        if not (0 <= cell[0] < field.shape[0] and 0 <= cell[1] < field.shape[1]):
            raise ValueError("Position is outside the project map")
        value = float(
            map_coordinates(field, np.asarray(cell)[:, None], order=1, mode="nearest")[
                0
            ]
        )
        if not math.isfinite(value):
            raise ValueError("No terrain surface at requested position")
        return value

    def height(self, x, y):
        return self._sample(self.heights, x, y)

    def spawn(self, x, y, yaw):
        # Align the body approximately to the terrain; suspension settles in Gazebo.
        gx, gy = self._sample(self.dx, x, y), self._sample(self.dy, x, y)
        forward = gx * math.cos(yaw) + gy * math.sin(yaw)
        lateral = -gx * math.sin(yaw) + gy * math.cos(yaw)
        return [
            x,
            y,
            self.height(x, y) + 0.48,
            math.atan(lateral),
            -math.atan(forward),
            yaw,
        ]

    def target_position(self, object_id):
        pose = self.poses[object_id]
        return list(pose[:2])


def known_map(config):
    if config.get("scene_manifest"):
        from .generate import GeneratedMap

        return GeneratedMap(config)
    if config.get("world", "mining_vehicles") in {"moon", "mars"}:
        return ProjectMap(config)
    return KnownMap(
        config["map"]["bounds"],
        config["map"]["resolution"],
        [o["bounds"] for o in config.get("obstacles", [])],
    )


def nearest_approach(
    maps,
    target,
    robots,
    localization,
    kind,
    maximum_distance,
    clearance=0.2,
    planner=None,
    grid=None,
    prefer_robot=False,
    start_poses=None,
):
    """Closest collision-clear, reachable station for the required capability."""
    from .contracts import Pose
    from .maps import traversable
    from .navigation import AStarPlanner, planning_start

    grid = maps.snapshot(0) if grid is None else grid
    yy, xx = np.indices(grid.blocked.shape)
    xy = np.c_[
        grid.origin[0] + xx.ravel() * grid.resolution,
        grid.origin[1] + yy.ravel() * grid.resolution,
    ]
    distances = np.linalg.norm(xy - np.array([target.x, target.y]), axis=1)
    planner = planner or AStarPlanner()
    best = None
    for robot in robots.values():
        if kind not in robot.capabilities:
            continue
        # Keep a small tracking margin instead of choosing the hard boundary.
        radius = robot.radius + clearance
        free = traversable(grid, radius).ravel()
        start = (
            start_poses[robot.name]
            if start_poses is not None
            else localization.get(robot.name).pose
        )
        # Classify physical reachability once for all station candidates. A
        # disconnected region cannot become reachable by dropping the optional
        # tracking margin, and repeating A* for every cell there is expensive.
        components, _ = label(traversable(grid, robot.radius))
        start_cell = planning_start(start, grid, robot.radius)
        if start_cell is None:
            continue
        component = components[start_cell[1], start_cell[0]]
        if component == 0:
            continue
        free &= components.ravel() == component
        admissible = free & (distances <= maximum_distance)
        costs = (
            np.linalg.norm(xy - [start.x, start.y], axis=1)
            if prefer_robot
            else distances
        )
        candidates = np.argsort(
            np.where(admissible if prefer_robot else free, costs, np.inf)
        )
        for index in candidates:
            if not admissible[index]:
                break
            x, y = xy[index]
            pose = Pose(
                float(x), float(y), math.atan2(target.y - y, target.x - x), grid.frame
            )
            path = planner.plan(start, pose, grid, radius)
            if path is None and clearance > 0:
                # A previously accepted physical stop can lie inside the
                # preferred tracking band. The actual footprint remains hard.
                path = planner.plan(start, pose, grid, robot.radius)
            if path is not None:
                if best is None or distances[index] < best[0]:
                    best = distances[index], pose
                break
    if best is None:
        raise ValueError(
            f"No reachable {kind} station within {maximum_distance} m of target"
        )
    return [best[1].x, best[1].y, best[1].yaw]


class MeasurementStations:
    """Plan only within the instrument's distance limit; never widen that limit."""

    def __init__(self, maps, localization, planner, clearance=0.2, robots=None):
        self.maps, self.localization, self.planner, self.clearance = (
            maps,
            localization,
            planner,
            clearance,
        )
        self.robots = robots or {}
        self.replan_each_bid = bool(robots)

    def plan(self, robot, poi, now):
        states = {
            name: self.localization.get(name)
            for name in self.robots
            if name != robot.name
        }
        return measurement_station_from(
            robot,
            poi,
            self.localization.get(robot.name).pose,
            self.maps.snapshot(now),
            self.planner,
            self.clearance,
            {
                name: (self.robots[name].radius, state.pose)
                for name, state in states.items()
                if state.usable(now)
            },
            prefer_robot=bool(self.robots),
        )


def measurement_station_from(
    robot, poi, start, grid, planner, clearance, peers=None, *, prefer_robot=True
):
    """Pure station query from an explicit pose and immutable map/peer evidence."""
    from .contracts import Kind, Pose

    station_radius = measurement_station_radius(
        poi.measurement_radius, robot.position_tolerance
    )
    if station_radius <= 0:
        raise ValueError("Instrument range is smaller than arrival safety margin")
    blocked = grid.blocked.copy()
    yy, xx = np.indices(blocked.shape)
    x = grid.origin[0] + xx * grid.resolution
    y = grid.origin[1] + yy * grid.resolution
    for radius, pose in (peers or {}).values():
        blocked |= np.hypot(x - pose.x, y - pose.y) <= radius + 0.3
    peer_grid = replace(grid, blocked=blocked)
    for evidence in (peer_grid, grid) if peers else (grid,):
        try:
            return Pose(
                *nearest_approach(
                    None,
                    poi.target,
                    {robot.name: robot},
                    None,
                    Kind.ROCK_MEASUREMENT,
                    station_radius,
                    clearance,
                    planner=planner,
                    grid=evidence,
                    prefer_robot=prefer_robot,
                    start_poses={robot.name: start},
                )
            )
        except ValueError:
            if evidence is grid:
                raise
