"""Known-map provider and explicit known-map crop placeholder for local mapping."""

from dataclasses import replace
import math

import numpy as np
from scipy.ndimage import distance_transform_edt, maximum_filter

from .contracts import GridMap


def traversable(grid, radius):
    # Padding makes map boundaries obstacles as well. Diagonal allowance covers
    # rounded cell queries; unknown cells must already be marked blocked.
    free = np.pad(~grid.blocked, 1, constant_values=False)
    clearance = distance_transform_edt(free)[1:-1, 1:-1] * grid.resolution
    return clearance > radius + 0.1 + grid.resolution * np.sqrt(2)


def footprint_clearances(grid, points, limit):
    """Continuous circle clearance to occupied cell squares and map boundaries.

    Planning inflation includes rounding and tracking allowances. Physical
    motion checks use the actual centre and occupied cell area, so a vehicle
    in that extra allowance can still safely move out of it.
    """
    xy = np.asarray(points, dtype=float).reshape(-1, 2)
    origin = np.asarray(grid.origin)
    upper = origin + (np.asarray(grid.blocked.shape[::-1]) - 1) * grid.resolution
    result = np.minimum(xy - origin, upper - xy).min(axis=1)
    lo = np.floor((xy.min(axis=0) - limit - origin) / grid.resolution - 0.5).astype(int)
    hi = np.ceil((xy.max(axis=0) + limit - origin) / grid.resolution + 0.5).astype(int)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi, np.asarray(grid.blocked.shape[::-1]) - 1)
    yy, xx = np.nonzero(grid.blocked[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1])
    if len(xx):
        centres = origin + np.c_[xx + lo[0], yy + lo[1]] * grid.resolution
        delta = np.maximum(np.abs(xy[:, None, :] - centres) - grid.resolution / 2, 0)
        result = np.minimum(result, np.linalg.norm(delta, axis=2).min(axis=1))
    return np.minimum(result, limit)


def segment_safe(start, goal, grid, radius):
    points = np.linspace(
        [start.x, start.y],
        [goal.x, goal.y],
        max(2, math.ceil(start.distance(goal) / 0.05) + 1),
    )
    return bool(
        np.all(footprint_clearances(grid, points, radius + 0.06) >= radius + 0.05)
    )


def rectangle_mask(grid, bounds):
    x = grid.origin[0] + np.arange(grid.blocked.shape[1]) * grid.resolution
    y = grid.origin[1] + np.arange(grid.blocked.shape[0]) * grid.resolution
    x0, y0, x1, y1 = bounds
    if not np.isfinite(bounds).all() or x0 > x1 or y0 > y1:
        raise ValueError("Expected finite ordered [xmin, ymin, xmax, ymax]")
    return (y[:, None] >= y0) & (y[:, None] <= y1) & (x >= x0) & (x <= x1)


def with_local_occupancy(global_grid, local_grid):
    """Conservatively overlay transient local evidence for one planning call."""
    if global_grid.frame != local_grid.frame:
        raise ValueError("Local and global map frames differ")
    x = (
        global_grid.origin[0]
        + np.arange(global_grid.blocked.shape[1]) * global_grid.resolution
    )
    y = (
        global_grid.origin[1]
        + np.arange(global_grid.blocked.shape[0]) * global_grid.resolution
    )
    lx = np.rint((x - local_grid.origin[0]) / local_grid.resolution).astype(int)
    ly = np.rint((y - local_grid.origin[1]) / local_grid.resolution).astype(int)
    valid_x = (lx >= 0) & (lx < local_grid.blocked.shape[1])
    valid_y = (ly >= 0) & (ly < local_grid.blocked.shape[0])
    # Cover all smaller local cells overlapping a coarse global cell.
    aligned = math.isclose(global_grid.resolution, local_grid.resolution) and all(
        math.isclose(
            (a - b) / local_grid.resolution,
            round((a - b) / local_grid.resolution),
            abs_tol=1e-8,
        )
        for a, b in zip(global_grid.origin, local_grid.origin)
    )
    dilation = (
        0 if aligned else math.ceil(global_grid.resolution / local_grid.resolution)
    )
    evidence = maximum_filter(local_grid.blocked, size=2 * dilation + 1)
    blocked = global_grid.blocked.copy()
    blocked[np.ix_(valid_y, valid_x)] |= evidence[np.ix_(ly[valid_y], lx[valid_x])]
    return replace(
        global_grid, blocked=blocked, source=global_grid.source + "+transient_local"
    )


class KnownMap:
    def height(self, x, y):
        return 0.0

    def spawn(self, x, y, yaw):
        return [x, y, 0.43, 0, 0, yaw]

    def __init__(self, bounds, resolution, obstacles=()):
        x0, y0, x1, y1 = bounds
        if x1 <= x0 or y1 <= y0 or resolution <= 0:
            raise ValueError("Invalid map bounds or resolution")
        shape = (round((y1 - y0) / resolution) + 1, round((x1 - x0) / resolution) + 1)
        self.grid = GridMap(
            np.zeros(shape, bool),
            (x0, y0),
            resolution,
            0,
            0,
            "known_layout",
            static=True,
        )
        for bounds in obstacles:
            self.block(bounds)

    def snapshot(self, now):
        return self.grid

    def replace(self, grid):
        if grid.frame != self.grid.frame or grid.version <= self.grid.version:
            raise ValueError(
                "Replacement map requires matching frame and newer version"
            )
        self.grid = grid

    def block(self, bounds, blocked=True):
        cells = self.grid.blocked.copy()
        cells[rectangle_mask(self.grid, bounds)] = blocked
        self.grid = replace(self.grid, blocked=cells, version=self.grid.version + 1)


class KnownMapCrop:
    """Explicit placeholder, NOT a sensor-generated local map.

    Peers are transient occupancy, never written back to the global map. Last
    known poses remain obstacles even if a peer's localization becomes invalid.
    """

    def __init__(self, global_maps, localization, robots, extent=5.0):
        self.global_maps, self.localization = global_maps, localization
        self.robots, self.extent = robots, extent
        self.last_known = {}

    def snapshot(self, robot, pose, now):
        global_grid = self.global_maps.snapshot(now)
        cx, cy = global_grid.cell(pose)
        n = int(np.ceil(self.extent / global_grid.resolution))
        x0, x1 = max(0, cx - n), min(global_grid.blocked.shape[1], cx + n + 1)
        y0, y1 = max(0, cy - n), min(global_grid.blocked.shape[0], cy + n + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return replace(global_grid, valid=False)
        origin = (
            global_grid.origin[0] + x0 * global_grid.resolution,
            global_grid.origin[1] + y0 * global_grid.resolution,
        )
        cells = global_grid.blocked[y0:y1, x0:x1].copy()
        x = origin[0] + np.arange(cells.shape[1]) * global_grid.resolution
        y = origin[1] + np.arange(cells.shape[0]) * global_grid.resolution
        for peer in self.robots.values():
            if peer.name == robot.name:
                continue
            state = self.localization.get(peer.name)
            if state.usable(now):
                self.last_known[peer.name] = state.pose
            p = self.last_known.get(peer.name)
            if p is None:
                return replace(global_grid, valid=False)
            if p.frame != global_grid.frame:
                raise ValueError("Peer pose and map frames differ")
            cells |= np.hypot(x[None, :] - p.x, y[:, None] - p.y) <= peer.radius + 0.3
        return GridMap(
            cells,
            origin,
            global_grid.resolution,
            global_grid.version,
            now,
            "known_map_crop+peer_occupancy",
            global_grid.frame,
            global_grid.usable(now),
            static=False,
        )
