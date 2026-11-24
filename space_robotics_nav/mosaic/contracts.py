"""Transport-independent contracts. All poses are planar, in metres/radians.

Adapters must transform observations into the mission's map frame before exposing
them here. Unknown map cells are blocked. Ground truth is not a mission API.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Mapping, Protocol

import numpy as np


class Kind(StrEnum):
    MOVE = "move"
    EXPLORE = "explore"
    ROCK_CANDIDATE = "rock_candidate"
    GROUND_MEASUREMENT = "ground_measurement"
    ROCK_MEASUREMENT = "rock_measurement"


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    yaw: float = 0.0
    frame: str = "map"

    def __post_init__(self):
        if not np.isfinite([self.x, self.y, self.yaw]).all() or not self.frame:
            raise ValueError("Pose requires finite coordinates and a frame")

    def distance(self, other):
        if self.frame != other.frame:
            raise ValueError("Pose frames must match; transform in the adapter")
        return float(np.hypot(self.x - other.x, self.y - other.y))


@dataclass(frozen=True)
class Localization:
    pose: Pose
    stamp: float
    source: str
    valid: bool = True
    linear_speed: float = 0.0
    yaw_rate: float = 0.0
    # Row-major covariance of x, y, yaw; None means unknown, not zero error.
    covariance: tuple[float, ...] | None = None
    lateral_speed: float = 0.0

    @property
    def speed(self):
        return float(np.hypot(self.linear_speed, self.lateral_speed))

    def usable(self, now, max_age=0.6):
        return (
            self.valid
            and np.isfinite(
                [self.stamp, self.linear_speed, self.lateral_speed, self.yaw_rate]
            ).all()
            and 0 <= now - self.stamp <= max_age
        )


@dataclass(frozen=True)
class Robot:
    name: str
    capabilities: frozenset[Kind]
    home: Pose
    radius: float = 1.7
    max_linear: float = 0.25
    max_angular: float = 0.45
    preferences: Mapping[Kind, float] = field(default_factory=dict)
    turn_offset: float = 0.0  # forward base-frame displacement from the turn centre
    yaw_rate_damping: float = 0.0
    position_tolerance: float = 0.32
    allow_reverse: bool = True

    def __post_init__(self):
        values = [
            self.radius,
            self.max_linear,
            self.max_angular,
            self.turn_offset,
            self.yaw_rate_damping,
            self.position_tolerance,
            *self.preferences.values(),
        ]
        if (
            not self.name
            or not np.isfinite(values).all()
            or min(self.radius, self.max_linear, self.max_angular) <= 0
            or self.yaw_rate_damping < 0
            or self.position_tolerance <= 0
            or not isinstance(self.allow_reverse, bool)
        ):
            raise ValueError("Invalid robot name, footprint or speed limits")


@dataclass(frozen=True)
class Detection:
    object_id: str
    target: Pose
    approach: Pose
    stamp: float
    source: str
    confidence: float = 1.0
    measurement_approach: Pose | None = None
    confirmation_radius: float = 3.0
    measurement_radius: float | None = None

    def __post_init__(self):
        if not np.isfinite(self.confirmation_radius) or self.confirmation_radius <= 0:
            raise ValueError("Confirmation radius must be positive and finite")
        if self.measurement_radius is not None and (
            not np.isfinite(self.measurement_radius) or self.measurement_radius <= 0
        ):
            raise ValueError("Measurement radius must be positive and finite")


@dataclass(frozen=True)
class ArrivalRegion:
    center: Pose
    radius: float

    def __post_init__(self):
        if not np.isfinite(self.radius) or self.radius <= 0:
            raise ValueError("Arrival radius must be positive and finite")


def measurement_station_radius(maximum_distance, position_tolerance):
    """Keep the whole arrival tolerance and 5 cm settling drift inside range."""
    return maximum_distance - position_tolerance - 0.05


@dataclass(frozen=True)
class Health:
    available: bool = True
    collision: bool = False
    reason: str = ""


@dataclass(frozen=True)
class GridMap:
    blocked: np.ndarray
    origin: tuple[float, float]
    resolution: float
    version: int
    stamp: float
    source: str
    frame: str = "map"
    valid: bool = True
    static: bool = True

    def __post_init__(self):
        cells = np.array(self.blocked, dtype=bool, copy=True)
        if cells.ndim != 2 or min(cells.shape) < 2:
            raise ValueError("Map must be a nonempty 2D grid")
        if (
            len(self.origin) != 2
            or self.resolution <= 0
            or not np.isfinite([*self.origin, self.resolution, self.stamp]).all()
        ):
            raise ValueError("Invalid map geometry or timestamp")
        cells.setflags(write=False)
        object.__setattr__(self, "blocked", cells)

    def usable(self, now, max_age=0.6):
        return self.valid and (self.static or 0 <= now - self.stamp <= max_age)

    def cell(self, pose):
        if pose.frame != self.frame:
            raise ValueError("Map and pose frames must match")
        return (
            round((pose.x - self.origin[0]) / self.resolution),
            round((pose.y - self.origin[1]) / self.resolution),
        )

    def contains(self, cell):
        x, y = cell
        return 0 <= x < self.blocked.shape[1] and 0 <= y < self.blocked.shape[0]


class LocalizationProvider(Protocol):
    def get(self, robot: str) -> Localization:
        ...


class MapProvider(Protocol):
    def snapshot(self, now: float) -> GridMap:
        ...


class LocalMapProvider(Protocol):
    def snapshot(self, robot: Robot, pose: Pose, now: float) -> GridMap:
        ...


class DetectionProvider(Protocol):
    def poll(self, robot: Robot, state: Localization, now: float) -> list[Detection]:
        ...


class ExplorationProvider(Protocol):
    def propose(
        self, robot: Robot, state: Localization, grid: GridMap, pois, now: float
    ) -> Pose | None:
        ...

    def snapshot(self) -> dict:
        ...


class HealthProvider(Protocol):
    def get(self, robot: str) -> Health:
        ...


@dataclass(frozen=True)
class InstrumentFeedback:
    status: str
    result: dict | None = None
    reason: str = ""


class Instrument(Protocol):
    def start(self, robot: str, kind: Kind, target: Pose, now: float) -> str:
        ...

    def tick(self, handle: str, state: Localization, now: float) -> InstrumentFeedback:
        ...

    def cancel(self, handle: str) -> None:
        ...


@dataclass(frozen=True)
class NavigationFeedback:
    status: str  # running, succeeded, failed
    velocity: tuple[float, float] = (0.0, 0.0)
    reason: str = ""


class Navigator(Protocol):
    def start(
        self,
        goal: Pose,
        now: float,
        *,
        arrival_region: ArrivalRegion | None = None,
        require_heading: bool = True,
    ) -> None:
        ...

    def cancel(self) -> None:
        ...

    def tick(self, state: Localization, now: float) -> NavigationFeedback:
        ...
