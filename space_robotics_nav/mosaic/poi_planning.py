"""Snapshot-only, bounded multi-step greedy POI plans and soft team intentions.

Only the mission thread may claim a POI or change a running action. This module
has no detector, simulator, live localization, or POI-store reference.
"""

from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import math
import threading
import time

from .contracts import Kind, Pose, measurement_station_radius
from .navigation import AStarPlanner
from .maps import footprint_clearances
from .scene import measurement_station_from


@dataclass(frozen=True)
class PlanningSettings:
    depth: int = 3
    discount: float = 0.8
    interval: float = 2.0
    intention_ttl: float = 20.0
    switch_margin: float = 5.0
    wall_budget: float = 1.0
    max_queries: int = 128

    def __post_init__(self):
        if (
            type(self.depth) is not int
            or not 1 <= self.depth <= 8
            or type(self.max_queries) is not int
            or self.max_queries < 1
            or not all(
                math.isfinite(v)
                for v in (
                    self.discount,
                    self.interval,
                    self.intention_ttl,
                    self.switch_margin,
                    self.wall_budget,
                )
            )
            or not 0 < self.discount <= 1
            or self.interval <= 0
            or self.intention_ttl <= self.interval
            or self.switch_margin < 0
            or self.wall_budget <= 0
        ):
            raise ValueError("Invalid POI depth-planning settings")


@dataclass(frozen=True)
class TaskSnapshot:
    id: str
    kind: Kind
    target: Pose
    approach: Pose
    revision: int = 0
    assigned_to: str | None = None
    measurement_radius: float | None = None
    confirmation_radius: float = 3.0
    failures: tuple = ()
    retry_until: tuple = ()

    @classmethod
    def capture(cls, poi, retry_seconds):
        failures = Counter()
        retry_until = {}
        for attempt in poi.attempts:
            if not attempt["success"] and not attempt["cancelled"]:
                name = attempt["robot"]
                failures[name] += 1
                retry_until[name] = max(
                    retry_until.get(name, 0), attempt["stamp"] + retry_seconds
                )
        return cls(
            poi.id,
            poi.kind,
            poi.target,
            poi.approach,
            poi.revision,
            poi.assigned_to,
            poi.measurement_radius,
            poi.confirmation_radius,
            tuple(sorted(failures.items())),
            tuple(sorted(retry_until.items())),
        )

    def eligible(self, robot, when):
        return (
            self.kind in robot.capabilities
            and self.assigned_to in (None, robot.name)
            and when >= dict(self.retry_until).get(robot.name, -math.inf)
        )

    def reward(self, robot):
        return (
            robot.preferences.get(self.kind, 100.0)
            - 10 * dict(self.failures).get(robot.name, 0)
            + (1000 if self.assigned_to == robot.name else 0)
        )


@dataclass(frozen=True)
class RobotSnapshot:
    robot: object
    pose: Pose
    local_grid: object
    active: TaskSnapshot | None = None
    active_end: Pose | None = None
    active_remaining: float = 0.0
    active_travel_seconds: float = 0.0
    active_path_length: float = 0.0
    active_token: str | None = None


@dataclass(frozen=True)
class PlanningSnapshot:
    stamp: float
    grid: object
    robots: tuple[RobotSnapshot, ...]
    tasks: tuple[TaskSnapshot, ...]
    exploration_seconds: float = 10.0
    measurement_clearance: float = 0.6
    # Only unexpired published future intentions, never execution claims.
    previous_owners: dict = field(default_factory=dict)
    measurement_enabled: bool = True


@dataclass(frozen=True)
class PlannedStep:
    poi_id: str
    revision: int
    kind: Kind
    depth: int
    current: bool
    start: Pose
    approach: Pose
    path_length: float
    utility: float
    discounted_utility: float
    expected_start: float
    expected_arrival: float
    expected_finish: float

    def record(self):
        return asdict(self)


@dataclass
class PlanningResult:
    stamp: float
    map_version: int
    plans: dict
    active_ids: dict
    task_revisions: dict = field(default_factory=dict)
    station_failures: dict = field(default_factory=dict)
    queries: int = 0
    cache_hits: int = 0
    elapsed: float = 0.0
    truncated: bool = False


class _BudgetReached(Exception):
    pass


class GreedyPOIPlanner:
    """Paper-style greedy state rollouts; no exhaustive sequence optimization."""

    def __init__(self, settings, planner=None):
        self.settings = settings
        self.planner = planner or AStarPlanner()
        self.cache = OrderedDict()

    def plan(self, snapshot, cancelled=None):
        started = time.monotonic()
        settings = self.settings
        result = PlanningResult(
            snapshot.stamp,
            snapshot.grid.version,
            {r.robot.name: [] for r in snapshot.robots},
            {
                r.robot.name: (r.active.id, r.active_token) if r.active else None
                for r in snapshot.robots
            },
            {p.id: p.revision for p in snapshot.tasks},
        )
        contexts = {}
        available = {p.id: p for p in snapshot.tasks}
        for r in snapshot.robots:
            name = r.robot.name
            contexts[name] = [r, r.pose, snapshot.stamp]
            if r.active:
                end = r.active_end or r.active.approach
                finish = snapshot.stamp + max(0, r.active_remaining)
                utility = r.active.reward(r.robot) - r.active_path_length
                result.plans[name].append(
                    PlannedStep(
                        r.active.id,
                        r.active.revision,
                        r.active.kind,
                        0,
                        True,
                        r.pose,
                        end,
                        r.active_path_length,
                        utility,
                        utility,
                        snapshot.stamp,
                        snapshot.stamp + r.active_travel_seconds,
                        finish,
                    )
                )
                contexts[name][1:] = [end, finish]
                available.pop(r.active.id, None)

        def checkpoint():
            if (
                cancelled and cancelled.is_set()
            ) or time.monotonic() - started >= settings.wall_budget:
                raise _BudgetReached

        def budget():
            checkpoint()
            if result.queries >= settings.max_queries:
                raise _BudgetReached

        if isinstance(self.planner, AStarPlanner):
            self.planner.checkpoint = checkpoint

        class CheckedPaths:
            def plan(_, *args):
                checkpoint()
                return self.planner.plan(*args)

        paths = CheckedPaths()

        def evaluate(r, start, task, immediate):
            checkpoint()
            if (
                task.kind == Kind.ROCK_MEASUREMENT
                and task.measurement_radius is not None
                and not snapshot.measurement_enabled
            ):
                return None, math.inf, "No science station planner configured"
            peers = (
                {
                    peer.robot.name: (peer.robot.radius, peer.pose)
                    for peer in snapshot.robots
                    if peer.robot.name != r.robot.name
                }
                if immediate
                else {}
            )
            peer_key = (
                tuple(sorted(peers.items()))
                if task.measurement_radius is not None
                else ()
            )
            key = (
                snapshot.grid.version,
                r.robot.radius,
                r.robot.position_tolerance,
                start,
                task.id,
                task.revision,
                task.kind,
                task.target,
                task.approach,
                task.measurement_radius,
                task.confirmation_radius,
                snapshot.measurement_clearance,
                peer_key,
            )
            # In-range candidate stops depend on current local peer evidence.
            in_range = (
                task.kind == Kind.ROCK_CANDIDATE
                and start.distance(task.target) <= task.confirmation_radius
            )
            if not in_range and key in self.cache:
                result.cache_hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            budget()
            result.queries += 1
            approach = task.approach
            if (
                task.kind == Kind.ROCK_MEASUREMENT
                and task.measurement_radius is not None
            ):
                try:
                    approach = measurement_station_from(
                        r.robot,
                        task,
                        start,
                        snapshot.grid,
                        paths,
                        snapshot.measurement_clearance,
                        peers,
                    )
                except ValueError as error:
                    value = (None, math.inf, str(error))
                    self.cache[key] = value
                    while len(self.cache) > 4096:
                        self.cache.popitem(last=False)
                    return value
            evidence = r.local_grid if immediate else snapshot.grid
            if (
                in_range
                and footprint_clearances(
                    evidence, [[start.x, start.y]], r.robot.radius + 0.06
                )[0]
                >= r.robot.radius + 0.05
            ):
                return start, 0.0, ""
            path = paths.plan(start, approach, snapshot.grid, r.robot.radius)
            value = (
                approach,
                sum(a.distance(b) for a, b in zip(path, path[1:]))
                if path is not None
                else math.inf,
                "",
            )
            self.cache[key] = value
            self.cache.move_to_end(key)
            while len(self.cache) > 4096:
                self.cache.popitem(last=False)
            return value

        def choose(names, immediate):
            estimates = []
            for name in names:
                r, start, when = contexts[name]
                depth = len(result.plans[name])
                for p in available.values():
                    if (
                        not p.eligible(r.robot, when)
                        or p.target.frame != snapshot.grid.frame
                        or p.approach.frame != snapshot.grid.frame
                    ):
                        continue
                    distance = start.distance(p.approach)
                    if p.kind == Kind.ROCK_CANDIDATE:
                        distance = max(
                            0, start.distance(p.target) - p.confirmation_radius
                        )
                    elif (
                        p.kind == Kind.ROCK_MEASUREMENT
                        and p.measurement_radius is not None
                    ):
                        distance = max(
                            0,
                            start.distance(p.target)
                            - measurement_station_radius(
                                p.measurement_radius, r.robot.position_tolerance
                            ),
                        )
                    # A* rounds its first point to a cell; retain a safe upper bound.
                    distance = max(
                        0, distance - math.sqrt(2) * snapshot.grid.resolution
                    )
                    bonus = (
                        settings.switch_margin
                        if not immediate and snapshot.previous_owners.get(p.id) == name
                        else 0.0
                    )
                    upper = (
                        p.reward(r.robot) - distance
                    ) * settings.discount**depth + bonus
                    if upper > 0:
                        estimates.append((-upper, name, p.id, bonus))
            best = None
            for negative_upper, name, poi_id, bonus in sorted(estimates):
                if best and -negative_upper < best[0][0]:
                    break
                r, start, when = contexts[name]
                p = available[poi_id]
                try:
                    approach, length, error = evaluate(r, start, p, immediate)
                except _BudgetReached:
                    result.truncated = True
                    break
                if error and immediate:
                    result.station_failures.setdefault(p.id, {})[name] = error
                if approach is None or not math.isfinite(length):
                    continue
                utility = p.reward(r.robot) - length
                if utility <= 0:
                    continue
                depth = len(result.plans[name])
                weighted = settings.discount**depth * utility
                arrival = when + length / (r.robot.max_linear * 0.8)
                duration = (
                    snapshot.exploration_seconds if p.kind == Kind.EXPLORE else 2.0
                )
                step = PlannedStep(
                    p.id,
                    p.revision,
                    p.kind,
                    depth,
                    False,
                    start,
                    approach,
                    length,
                    utility,
                    weighted,
                    when,
                    arrival,
                    arrival + duration,
                )
                # Time breaks equal bids; names/IDs make coordination reproducible.
                rank = weighted + bonus, -arrival
                if best is None or rank > best[0]:
                    best = rank, name, step
            return None if best is None else best[1:]

        def append(name, step):
            result.plans[name].append(step)
            contexts[name][1:] = [step.approach, step.expected_finish]
            available.pop(step.poi_id)

        # Every idle robot's immediate opportunity precedes all future intentions.
        idle = {name for name, (r, _, _) in contexts.items() if not r.active}
        while idle:
            winner = choose(idle, True)
            if winner is None:
                break
            name, step = winner
            append(name, step)
            idle.remove(name)
            if result.truncated:
                break
        if not result.truncated:
            while available:
                names = [
                    name
                    for name, plan in result.plans.items()
                    if 0 < len(plan) < settings.depth
                ]
                winner = choose(names, False)
                if winner is None:
                    break
                append(*winner)
                if result.truncated:
                    break
        result.elapsed = time.monotonic() - started
        return result


class BackgroundPOIPlanner:
    """One worker over detached snapshots; polling never waits for a search."""

    def __init__(self, settings):
        self.settings = settings
        self.engine = GreedyPOIPlanner(settings)
        self.executor = None
        self.future = None
        self.cancelled = threading.Event()

    def submit(self, snapshot):
        if self.future is not None:
            return False
        if self.executor is None:
            self.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mosaic-poi"
            )
        self.cancelled.clear()
        self.future = self.executor.submit(self.engine.plan, snapshot, self.cancelled)
        return True

    def poll(self):
        if self.future is None or not self.future.done():
            return None
        future, self.future = self.future, None
        return future.result()

    def close(self):
        self.cancelled.set()
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
        self.future = None
