"""Per-robot bidding and interruptible execution, independent of Gazebo/ROS."""

from dataclasses import dataclass, replace
import math

from .contracts import ArrivalRegion, Kind, Pose, measurement_station_radius
from .navigation import PathTracker
from .poi import POI, Status


@dataclass
class Execution:
    poi_id: str
    token: str
    started: float
    revision: int
    phase: str = "navigate"
    action_handle: str | None = None
    analysis_started: float | None = None
    analysis_station: Pose | None = None
    result: dict | None = None
    last_station_try: float = -math.inf
    timeout_budget: float = 360.0


class Mission:
    def __init__(
        self,
        robots,
        store,
        localization,
        maps,
        local_maps,
        detection,
        health,
        navigators,
        planner,
        *,
        emit=None,
        auto_approve=False,
        instrument,
        exploration_seconds=10.0,
        exploration=None,
        exploration_queue=2,
        measurement_stations=None,
        poi_planning=None,
    ):
        self.robots, self.store, self.localization = robots, store, localization
        self.maps, self.local_maps, self.detection, self.health = (
            maps,
            local_maps,
            detection,
            health,
        )
        self.navigators, self.planner = navigators, planner
        self.instrument = instrument
        self.auto_approve = auto_approve
        if not math.isfinite(exploration_seconds) or exploration_seconds <= 0:
            raise ValueError(
                "Exploration analysis duration must be positive and finite"
            )
        self.exploration_seconds = exploration_seconds
        self.executions = {}
        self.yielding = set()
        self.modes = {name: "auto" for name in robots}
        self.manual = {}
        self.emit = emit or (lambda event: None)
        self.now = 0.0
        self.exploration, self.exploration_queue = exploration, exploration_queue
        if (
            not isinstance(exploration_queue, int)
            or isinstance(exploration_queue, bool)
            or exploration_queue < 1
        ):
            raise ValueError("Exploration queue must be a positive integer")
        self.next_exploration_check, self.exploration_serial = 0.0, 0
        self.measurement_stations = measurement_stations
        from .poi_planning import BackgroundPOIPlanner, PlanningSettings

        try:
            self.planning_settings = PlanningSettings(**(poi_planning or {"depth": 1}))
        except TypeError as error:
            raise ValueError(f"Unknown POI planning setting: {error}") from error
        self.sequence_planner = (
            BackgroundPOIPlanner(self.planning_settings)
            if self.planning_settings.depth > 1
            else None
        )
        self.poi_plans = {name: [] for name in robots}
        self.plan_version, self.plan_stamp = 0, None
        self.plan_map_version, self.plan_valid_until = None, -math.inf
        self.next_plan_at, self.requested_plan_signature = 0.0, None
        self.plan_stats, self.planning_error = {}, ""

    def close(self):
        if self.sequence_planner is not None:
            self.sequence_planner.close()

    def planning_status(self):
        owners = {}
        plans = {}
        for name, steps in self.poi_plans.items():
            plans[name] = [s.record() for s in steps]
            for step in steps:
                if step.depth > 0:
                    owners[step.poi_id] = dict(
                        robot=name,
                        depth=step.depth,
                        discounted_utility=step.discounted_utility,
                        expected_arrival=step.expected_arrival,
                        valid_until=self.plan_valid_until,
                    )
        return dict(
            depth=self.planning_settings.depth,
            discount=self.planning_settings.discount,
            version=self.plan_version,
            generated_at=self.plan_stamp,
            valid_until=self.plan_valid_until if self.plan_stamp is not None else None,
            map_version=self.plan_map_version,
            computing=bool(self.sequence_planner and self.sequence_planner.future),
            error=self.planning_error,
            stats=self.plan_stats.copy(),
            plans=plans,
            future_owners=owners,
        )

    def _planning_signature(self, grid):
        return (
            grid.version,
            tuple(
                (
                    p.id,
                    p.revision,
                    p.status,
                    p.assigned_to,
                    p.claimed_by,
                    len(p.attempts),
                )
                for p in self.store.items.values()
            ),
            tuple(
                (
                    name,
                    self.modes[name],
                    self.localization.get(name).usable(self.now),
                    self.health.get(name).available,
                    self.health.get(name).collision,
                )
                for name in self.robots
            ),
        )

    def _prune_plans(self, grid):
        active_ids = {t.poi_id for t in self.executions.values()}
        for name, robot in self.robots.items():
            state, health = self.localization.get(name), self.health.get(name)
            if (
                self.now > self.plan_valid_until
                or grid.version != self.plan_map_version
                or not grid.usable(self.now)
                or self.modes[name] != "auto"
                or not state.usable(self.now)
                or state.pose.frame != grid.frame
                or not health.available
                or health.collision
            ):
                self.poi_plans[name] = []
                continue
            valid = []
            for step in self.poi_plans[name]:
                p = self.store.items.get(step.poi_id)
                if step.current:
                    task = self.executions.get(name)
                    if task is None:
                        # Successful completion permits its old tail to remain
                        # a soft intention while a new first step is calculated.
                        if p is not None and p.status == Status.SUCCEEDED:
                            continue
                        break
                    if (
                        task.poi_id != step.poi_id
                        or p is None
                        or p.revision != step.revision
                    ):
                        break
                elif (
                    p is None
                    or p.status != Status.AVAILABLE
                    or p.revision != step.revision
                    or p.kind not in robot.capabilities
                    or p.assigned_to not in (None, name)
                    or p.id in active_ids
                ):
                    break
                valid.append(step)
            self.poi_plans[name] = valid

    def _planning_snapshot(self, grid):
        from .poi_planning import PlanningSnapshot, RobotSnapshot, TaskSnapshot

        def frozen_grid(value):
            blocked = value.blocked.copy()
            blocked.flags.writeable = False
            return replace(value, blocked=blocked)

        robots = []
        for name, robot in self.robots.items():
            state, health = self.localization.get(name), self.health.get(name)
            if (
                self.modes[name] != "auto"
                or not state.usable(self.now)
                or state.pose.frame != grid.frame
                or not health.available
                or health.collision
            ):
                continue
            local = self.local_maps.snapshot(robot, state.pose, self.now)
            if not local.usable(self.now):
                continue
            task = self.executions.get(name)
            active, end, remaining = None, None, 0.0
            travel_seconds, path_length = 0.0, 0.0
            if task:
                p = self.store.items[task.poi_id]
                active = TaskSnapshot.capture(p, self.store.retry_seconds)
                end = p.approach if task.phase == "navigate" else state.pose
                path = getattr(self.navigators[name], "path", None) or []
                path = path[getattr(self.navigators[name], "index", 0) :]
                points = (
                    ([state.pose, *path] if path else [state.pose, end])
                    if task.phase == "navigate"
                    else [state.pose]
                )
                path_length = sum(a.distance(b) for a, b in zip(points, points[1:]))
                travel_seconds = path_length / (robot.max_linear * 0.8)
                remaining = travel_seconds
                action_time = (
                    self.exploration_seconds if p.kind == Kind.EXPLORE else 2.0
                )
                if task.analysis_started is not None:
                    action_time = max(
                        0.0, action_time - (self.now - task.analysis_started)
                    )
                remaining += action_time
            robots.append(
                RobotSnapshot(
                    replace(robot, preferences=dict(robot.preferences)),
                    state.pose,
                    frozen_grid(local),
                    active,
                    end,
                    remaining,
                    travel_seconds,
                    path_length,
                    task.token if task else None,
                )
            )
        return PlanningSnapshot(
            self.now,
            frozen_grid(grid),
            tuple(robots),
            tuple(
                TaskSnapshot.capture(p, self.store.retry_seconds)
                for p in self.store.items.values()
                if p.status == Status.AVAILABLE
            ),
            self.exploration_seconds,
            getattr(self.measurement_stations, "clearance", 0.2),
            {
                p: owner["robot"]
                for p, owner in self.planning_status()["future_owners"].items()
            },
            self.measurement_stations is not None,
        )

    def _select_sequence(self):
        grid = self.maps.snapshot(self.now)
        self._prune_plans(grid)
        if not grid.usable(self.now):
            return
        try:
            result = self.sequence_planner.poll()
        except Exception as error:
            result = None
            self.planning_error = str(error)
            self.event("poi_planning_failed", reason=str(error))
        if result is not None:
            if (
                result.map_version == grid.version
                and 0 <= self.now - result.stamp <= self.planning_settings.intention_ttl
            ):
                self.poi_plans = {name: [] for name in self.robots}
                self.plan_version += 1
                self.plan_stamp, self.plan_map_version = (
                    result.stamp,
                    result.map_version,
                )
                self.plan_valid_until = (
                    result.stamp + self.planning_settings.intention_ttl
                )
                self.plan_stats = dict(
                    queries=result.queries,
                    cache_hits=result.cache_hits,
                    elapsed_seconds=result.elapsed,
                    truncated=result.truncated,
                )
                self.planning_error = ""
                for name, steps in result.plans.items():
                    task = self.executions.get(name)
                    if (
                        (task.poi_id, task.token) if task else None
                    ) == result.active_ids[name]:
                        self.poi_plans[name] = list(steps)
                self._prune_plans(grid)
                proposed_ids = {
                    s.poi_id for steps in self.poi_plans.values() for s in steps
                }
                for poi_id, errors in result.station_failures.items():
                    p = self.store.items.get(poi_id)
                    if (
                        p is None
                        or p.status != Status.AVAILABLE
                        or p.revision != result.task_revisions[poi_id]
                        or p.id in proposed_ids
                    ):
                        continue
                    compatible = [
                        r.name
                        for r in self.robots.values()
                        if p.kind in r.capabilities and p.assigned_to in (None, r.name)
                    ]
                    if compatible and all(name in errors for name in compatible):
                        p.status, p.blocked_reason = (
                            Status.NEEDS_HELP,
                            "measurement_station_unreachable",
                        )
                        self.event(
                            "measurement_station_unreachable",
                            robot=compatible[0],
                            robots=compatible,
                            poi=p.id,
                            maximum_distance=p.measurement_radius,
                            detail="; ".join(errors.values()),
                        )
                heads = sorted(
                    (
                        (-steps[0].utility, name, steps[0])
                        for name, steps in self.poi_plans.items()
                        if steps and steps[0].depth == 0 and not steps[0].current
                    ),
                    key=lambda value: (value[0], value[1]),
                )
                for _, name, step in heads:
                    robot, p = self.robots[name], self.store.items[step.poi_id]
                    state = self.localization.get(name)
                    if (
                        name in self.executions
                        or not self.store.eligible(p, robot, self.now)
                        or state.pose.distance(step.start) > 0.75
                    ):
                        self.poi_plans[name] = []
                        continue
                    station = None
                    if (
                        p.kind == Kind.ROCK_MEASUREMENT
                        and p.measurement_radius is not None
                    ):
                        try:
                            station = self.measurement_stations.plan(robot, p, self.now)
                        except ValueError:
                            self.poi_plans[name] = []
                            continue
                    if self._claim_task(name, p, step.utility, station):
                        self.poi_plans[name][0] = replace(
                            step, current=True, revision=p.revision, approach=p.approach
                        )
                        if (
                            station is not None
                            and station.distance(step.approach) > 0.5
                        ):
                            self.poi_plans[name] = self.poi_plans[name][:1]
                self._prune_plans(grid)
                self.event("poi_plan_published", **self.planning_status())
            else:
                self.event(
                    "poi_plan_discarded",
                    reason="map_changed_or_expired",
                    planned_at=result.stamp,
                )
        signature = self._planning_signature(grid)
        if self.sequence_planner.future is None and (
            self.now >= self.next_plan_at or signature != self.requested_plan_signature
        ):
            self.sequence_planner.submit(self._planning_snapshot(grid))
            self.next_plan_at = self.now + self.planning_settings.interval
            self.requested_plan_signature = signature

    def _claim_task(self, name, p, utility, station=None):
        token = self.store.claim(p.id, self.robots[name], self.now)
        if not token:
            return False
        if station is not None:
            p.approach = p.measurement_approach = station
            p.revision += 1
            self.event(
                "measurement_station_planned",
                robot=name,
                poi=p.id,
                position=[station.x, station.y],
                distance=station.distance(p.target),
                maximum_distance=p.measurement_radius,
                station_maximum_distance=measurement_station_radius(
                    p.measurement_radius, self.robots[name].position_tolerance
                ),
                position_tolerance=self.robots[name].position_tolerance,
            )
        self.executions[name] = Execution(p.id, token, self.now, p.revision)
        self.yielding.discard(name)
        self._start_navigation(name, p)
        self.event("task_claimed", robot=name, poi=p.id, kind=p.kind, utility=utility)
        return True

    def exploration_status(self):
        return (
            self.exploration.snapshot()
            if self.exploration
            else dict(enabled=False, state="disabled")
        )

    def is_complete(self):
        return (
            not self.executions
            and not self.yielding
            and all(
                p.status in {Status.SUCCEEDED, Status.REJECTED}
                for p in self.store.items.values()
            )
            and self.exploration_status()["state"] in {"disabled", "exhausted"}
        )

    def _replenish_exploration(self):
        if self.exploration is None or self.now < self.next_exploration_check:
            return
        self.next_exploration_check = self.now + 2.0
        scouts = [
            r
            for r in self.robots.values()
            if Kind.EXPLORE in r.capabilities
            and self.modes[r.name] == "auto"
            and self.localization.get(r.name).usable(self.now)
            and self.health.get(r.name).available
            and not self.health.get(r.name).collision
        ]
        scout_names = {r.name for r in scouts}
        pending = sum(
            p.kind == Kind.EXPLORE
            and p.status in {Status.AVAILABLE, Status.CLAIMED, Status.RUNNING}
            and p.assigned_to in scout_names | {None}
            for p in self.store.items.values()
        )
        desired = len(scouts) * self.exploration_queue
        while pending < desired:
            progressed = False
            for robot in scouts:
                if pending >= desired:
                    break
                before = self.exploration_status()["state"]
                pose = self.exploration.propose(
                    robot,
                    self.localization.get(robot.name),
                    self.maps.snapshot(self.now),
                    list(self.store.items.values()),
                    self.now,
                )
                if pose is None:
                    if self.exploration_status()["state"] != before:
                        self.event(
                            "exploration_source_state", **self.exploration_status()
                        )
                    continue
                while True:
                    self.exploration_serial += 1
                    poi_id = f"auto_explore_{self.exploration_serial:03d}"
                    if poi_id not in self.store.items:
                        break
                self.store.add(
                    POI(
                        poi_id,
                        Kind.EXPLORE,
                        pose,
                        pose,
                        task_source="known_layout_stations",
                    )
                )
                self.event(
                    "exploration_poi_generated",
                    robot=robot.name,
                    poi=poi_id,
                    position=[pose.x, pose.y],
                )
                pending += 1
                progressed = True
            if not progressed:
                break

    def event(self, event, **fields):
        self.emit(dict(stamp=self.now, event=event, **fields))

    def _finish(self, name, success, reason="", *, cancelled=False):
        self.yielding.discard(name)
        task = self.executions.pop(name, None)
        self.navigators[name].cancel()
        if task and task.action_handle:
            self.instrument.cancel(task.action_handle)
        if task and self.store.finish(
            task.poi_id,
            task.token,
            self.now,
            success=success,
            reason=reason,
            result=task.result,
            cancelled=cancelled,
        ):
            self.event(
                "task_finished",
                robot=name,
                poi=task.poi_id,
                success=success,
                reason=reason,
                result=task.result,
            )

    def _select(self):
        if self.sequence_planner is not None:
            self._select_sequence()
            return
        # Each robot computes its own preferences; shared claims arbitrate them.
        proposals = []
        stations = {}
        station_failures = {}
        grid = self.maps.snapshot(self.now)
        if not grid.usable(self.now):
            return
        for name, robot in self.robots.items():
            state = self.localization.get(name)
            health = self.health.get(name)
            if (
                name in self.executions
                or self.modes[name] != "auto"
                or not state.usable(self.now)
                or not health.available
                or health.collision
            ):
                continue
            for p in self.store.items.values():
                if not self.store.eligible(p, robot, self.now):
                    continue
                approach = p.approach
                if p.kind == Kind.ROCK_MEASUREMENT and p.measurement_radius is not None:
                    station = p.measurement_approach
                    station_radius = measurement_station_radius(
                        p.measurement_radius, robot.position_tolerance
                    )
                    if (
                        getattr(self.measurement_stations, "replan_each_bid", False)
                        or station is None
                        or station.distance(p.target) > station_radius
                    ):
                        try:
                            if self.measurement_stations is None:
                                raise ValueError(
                                    "No science station planner configured"
                                )
                            station = self.measurement_stations.plan(robot, p, self.now)
                            if station.distance(p.target) > station_radius:
                                raise ValueError(
                                    "Science station leaves no margin for arrival tolerance"
                                )
                        except ValueError as error:
                            station_failures.setdefault(p.id, {})[name] = str(error)
                            continue
                    approach = station
                    stations[name, p.id] = station
                if state.pose.frame != grid.frame or approach.frame != grid.frame:
                    continue
                path = (
                    [state.pose]
                    if p.kind == Kind.ROCK_CANDIDATE
                    and state.pose.distance(p.target) <= p.confirmation_radius
                    and PathTracker.safe(
                        state,
                        (0.0, 0.0),
                        self.local_maps.snapshot(robot, state.pose, self.now),
                        robot.radius,
                        robot.turn_offset,
                    )
                    else self.planner.plan(state.pose, approach, grid, robot.radius)
                )
                if path is None:
                    continue
                length = sum(a.distance(b) for a, b in zip(path, path[1:]))
                failures = sum(
                    a["robot"] == name and not a["success"] and not a["cancelled"]
                    for a in p.attempts
                )
                reward = robot.preferences.get(p.kind, 100.0)
                score = (
                    reward
                    - length
                    - failures * 10
                    + (1000 if p.assigned_to == name else 0)
                )
                if score > 0:
                    proposals.append((-score, name, p.id))
        # One scientist's start state must not permanently block the task for
        # the rest of the fleet. A busy compatible robot can try when free.
        proposed_ids = {entry[2] for entry in proposals}
        for poi_id, errors in station_failures.items():
            p = self.store.items[poi_id]
            compatible = [
                r.name
                for r in self.robots.values()
                if p.kind in r.capabilities and p.assigned_to in (None, r.name)
            ]
            if poi_id in proposed_ids or any(name not in errors for name in compatible):
                continue
            p.status, p.blocked_reason = (
                Status.NEEDS_HELP,
                "measurement_station_unreachable",
            )
            self.event(
                "measurement_station_unreachable",
                robot=next(iter(errors)),
                robots=list(errors),
                poi=p.id,
                maximum_distance=p.measurement_radius,
                detail="; ".join(
                    f"{name}: {reason}" for name, reason in errors.items()
                ),
            )
        for negative_score, name, poi_id in sorted(proposals):
            if name in self.executions:
                continue
            self._claim_task(
                name,
                self.store.items[poi_id],
                -negative_score,
                stations.get((name, poi_id)),
            )

    def _start_navigation(self, name, poi):
        self.navigators[name].start(
            poi.approach,
            self.now,
            arrival_region=ArrivalRegion(poi.target, poi.confirmation_radius)
            if poi.kind == Kind.ROCK_CANDIDATE
            else None,
            require_heading=poi.kind not in {Kind.EXPLORE, Kind.ROCK_CANDIDATE}
            and (
                poi.kind != Kind.ROCK_MEASUREMENT
                or getattr(self.instrument, "requires_heading", True)
            ),
        )

    def _observe(self, name, exploration):
        found, observed = [], []
        for detection in self.detection.poll(
            self.robots[name], self.localization.get(name), self.now
        ):
            if (
                not 0 <= self.now - detection.stamp <= 0.6
                or detection.confidence < 0.5
                or detection.target.frame != self.maps.snapshot(self.now).frame
                or detection.approach.frame != detection.target.frame
                or detection.approach.distance(detection.target)
                > detection.confirmation_radius
            ):
                continue
            poi, created = self.store.observe(detection)
            observed.append(poi.id)
            if created:
                found.append(poi.id)
                poi.discovered_by, poi.source_exploration = name, exploration.id
                self.event(
                    "target_detected",
                    robot=name,
                    poi=poi.id,
                    source=detection.source,
                    exploration=exploration.id,
                )
        return sorted(set(found)), sorted(set(observed))

    def _yield_if_blocking_science(self, name, state):
        """Idle scouts clear a science station; a newly selected task takes priority."""
        robot = self.robots[name]
        if Kind.ROCK_CANDIDATE not in robot.capabilities:
            return (0.0, 0.0)
        blocking = any(
            poi.kind == Kind.ROCK_MEASUREMENT
            and poi.status in {Status.AVAILABLE, Status.CLAIMED, Status.RUNNING}
            and Kind.ROCK_MEASUREMENT in peer.capabilities
            and state.pose.distance(poi.approach)
            < self._station_separation(peer, robot)
            for poi in self.store.items.values()
            for peer in self.robots.values()
            if peer.name != name
        )
        if not blocking:
            if name in self.yielding:
                if state.speed >= 0.03 or abs(state.yaw_rate) >= 0.04:
                    return (0.0, 0.0)
                self.yielding.remove(name)
                self.navigators[name].cancel()
                self.event("site_clearance_finished", robot=name)
            return (0.0, 0.0)
        if name not in self.yielding:
            self.yielding.add(name)
            self.navigators[name].start(robot.home, self.now, require_heading=False)
            self.event("site_clearance_started", robot=name)
        feedback = self.navigators[name].tick(state, self.now)
        if feedback.status == "failed":
            self.yielding.remove(name)
            self.navigators[name].cancel()
            self.event("site_clearance_failed", robot=name, reason=feedback.reason)
            return (0.0, 0.0)
        return feedback.velocity

    @staticmethod
    def _station_separation(scientist, scout):
        # Leave room for arrival tolerance and the body's sweep around an
        # offset turn centre, as well as the local occupancy inflation.
        return (
            scientist.radius
            + scout.radius
            + 0.8
            + scientist.position_tolerance
            + 2 * abs(scientist.turn_offset)
        )

    def poi_wait_reason(self, poi):
        if poi.blocked_reason:
            return f"Science station unreachable within {poi.measurement_radius:g} m; not measured"
        if poi.status == Status.WAITING_APPROVAL:
            return "Waiting for operator approval"
        if poi.status == Status.NEEDS_HELP:
            failure = next(
                (
                    a
                    for a in reversed(poi.attempts)
                    if not a["success"] and not a["cancelled"]
                ),
                None,
            )
            if failure:
                return f"Needs help: {failure['reason']} (last failure at {failure['stamp']:.1f} s)"
        if poi.claimed_by:
            task = self.executions.get(poi.claimed_by)
            navigation_wait = (
                getattr(self.navigators[poi.claimed_by], "waiting_reason", "")
                if task and task.phase == "navigate"
                else ""
            )
            return f"{poi.claimed_by}: {navigation_wait or (task.phase if task else 'starting')}"
        if poi.status != Status.AVAILABLE:
            return poi.status
        compatible = [
            r
            for r in self.robots.values()
            if poi.kind in r.capabilities and poi.assigned_to in (None, r.name)
        ]
        if not compatible:
            return "No robot has the required capability"
        for robot in compatible:
            if robot.name in self.executions:
                return f"Waiting for {robot.name}: {self.executions[robot.name].poi_id}"
        if not any(self.store.eligible(poi, r, self.now) for r in compatible):
            return "Waiting for retry cooldown"
        if not any(
            self.modes[r.name] == "auto"
            and self.localization.get(r.name).usable(self.now)
            for r in compatible
        ):
            return "Compatible robot is paused, disabled or unlocalized"
        return "Waiting for a reachable approach"

    def robot_wait_reason(self, robot):
        if robot.name in self.executions:
            return ""
        if self.is_complete():
            return "All mission tasks completed"
        if Kind.EXPLORE in robot.capabilities and self.exploration is not None:
            return f"Exploration source: {self.exploration_status()['state']}"
        if robot.name in self.yielding:
            return "Clearing space for science vehicle"
        if Kind.ROCK_MEASUREMENT in robot.capabilities:
            for poi in self.store.items.values():
                if poi.blocked_reason:
                    return f"Blocked science task: {poi.id} ({poi.measurement_radius:g} m range)"
                if poi.status == Status.WAITING_APPROVAL:
                    return f"Waiting for approval: {poi.id}"
                if poi.kind == Kind.ROCK_CANDIDATE and poi.status in {
                    Status.AVAILABLE,
                    Status.CLAIMED,
                    Status.RUNNING,
                }:
                    return f"Waiting for Scout confirmation: {poi.id}"
        if Kind.ROCK_MEASUREMENT in robot.capabilities:
            return "Waiting for Scout discoveries / confirmation"
        return "Waiting for a compatible task"

    def tick(self, now):
        if not math.isfinite(now) or now < self.now:
            raise ValueError("Mission time must be monotonic")
        self.now = now
        self.store.expire(now)
        commands = {name: (0.0, 0.0) for name in self.robots}
        # Finish/release old tasks before considering new bids.
        for name in list(self.executions):
            state, health = self.localization.get(name), self.health.get(name)
            if not state.usable(now):
                self._finish(name, False, "localization_unavailable")
            elif not health.available or health.collision:
                self._finish(name, False, health.reason or "collision")
            elif not self.store.renew(
                self.executions[name].poi_id, self.executions[name].token, now
            ):
                lost = self.executions.pop(name)
                if lost.action_handle:
                    self.instrument.cancel(lost.action_handle)
                self.navigators[name].cancel()
                self.event("claim_lost", robot=name)
        self._replenish_exploration()
        self._select()
        for name, robot in self.robots.items():
            state = self.localization.get(name)
            if self.modes[name] == "teleop":
                velocity, until = self.manual[name]
                health = self.health.get(name)
                local = self.local_maps.snapshot(robot, state.pose, now)

                if (
                    now < until
                    and state.usable(now)
                    and health.available
                    and not health.collision
                    and local.usable(now)
                    and PathTracker.safe(
                        state, velocity, local, robot.radius, robot.turn_offset
                    )
                ):
                    commands[name] = velocity
                continue
            if name not in self.executions:
                health = self.health.get(name)
                if (
                    self.modes[name] == "auto"
                    and state.usable(now)
                    and health.available
                    and not health.collision
                ):
                    commands[name] = self._yield_if_blocking_science(name, state)
                continue
            task, navigator = self.executions[name], self.navigators[name]
            p = self.store.items[task.poi_id]
            # Allow the navigation module's finite route budget plus time for
            # analysis/measurement. This is still relative to the task's first
            # claim, so station updates cannot reset its overall time limit.
            task.timeout_budget = max(
                task.timeout_budget,
                getattr(navigator, "navigation_budget", 180.0)
                + max(180.0, self.exploration_seconds + 30.0),
            )
            if now - task.started > task.timeout_budget:
                self._finish(name, False, "task_timeout")
                continue
            if task.revision != p.revision:
                if task.action_handle:
                    self.instrument.cancel(task.action_handle)
                task.revision, task.phase = p.revision, "navigate"
                task.action_handle, task.analysis_started = None, None
                self._start_navigation(name, p)
                self.event("target_updated", robot=name, poi=p.id)
            if task.phase == "navigate":
                feedback = navigator.tick(state, now)
                commands[name] = feedback.velocity
                if (
                    p.kind == Kind.ROCK_MEASUREMENT
                    and self.measurement_stations is not None
                    and getattr(navigator, "blocked_since", None) is not None
                    and now - navigator.blocked_since >= 3
                    and now - task.last_station_try >= 5
                ):
                    task.last_station_try = now
                    try:
                        alternative = self.measurement_stations.plan(robot, p, now)
                    except ValueError:
                        alternative = None
                    if (
                        alternative is not None
                        and alternative.distance(p.approach) >= 0.5
                    ):
                        p.approach = p.measurement_approach = alternative
                        p.revision += 1
                        task.revision = p.revision
                        self._start_navigation(name, p)
                        commands[name] = (0.0, 0.0)
                        self.event(
                            "science_station_replanned",
                            robot=name,
                            poi=p.id,
                            position=[alternative.x, alternative.y],
                            distance=alternative.distance(p.target),
                            reason="transient_local_blockage",
                        )
                        continue
                if feedback.status == "failed":
                    commands[name] = (0.0, 0.0)
                    self._finish(name, False, feedback.reason)
                elif feedback.status == "succeeded":
                    commands[name] = (0.0, 0.0)
                    if p.kind == Kind.MOVE:
                        self._finish(name, True)
                    elif p.kind == Kind.EXPLORE:
                        task.phase, task.analysis_started, task.analysis_station = (
                            "analyze",
                            now,
                            state.pose,
                        )
                        navigator.cancel()
                        self.event(
                            "analysis_started",
                            robot=name,
                            poi=p.id,
                            x=state.pose.x,
                            y=state.pose.y,
                            yaw=state.pose.yaw,
                        )
                    else:
                        task.phase = "action"
                        self.event("action_started", robot=name, poi=p.id, kind=p.kind)
            elif task.phase == "analyze":
                if not self.maps.snapshot(now).usable(
                    now
                ) or not self.local_maps.snapshot(robot, state.pose, now).usable(now):
                    self._finish(name, False, "map_unavailable")
                    continue
                if state.speed >= 0.03 or abs(state.yaw_rate) >= 0.04:
                    task.analysis_started = None
                elif (
                    task.analysis_started is None
                    or state.pose.distance(task.analysis_station) > 0.05
                ):
                    task.analysis_started, task.analysis_station = now, state.pose
                elif now - task.analysis_started >= self.exploration_seconds:
                    found, observed = self._observe(name, p)
                    task.result = dict(
                        simulated=True,
                        kind=Kind.EXPLORE,
                        robot=name,
                        stamp=now,
                        analysis_seconds=now - task.analysis_started,
                        candidates=found,
                        observed_objects=observed,
                    )
                    self.event(
                        "analysis_finished",
                        robot=name,
                        poi=p.id,
                        seconds=now - task.analysis_started,
                        candidates=found,
                        observed_objects=observed,
                    )
                    self._finish(name, True)
            elif task.phase == "action":
                if not self.maps.snapshot(now).usable(
                    now
                ) or not self.local_maps.snapshot(robot, state.pose, now).usable(now):
                    self._finish(name, False, "map_unavailable")
                    continue
                if p.kind in {Kind.ROCK_MEASUREMENT, Kind.GROUND_MEASUREMENT}:
                    if (
                        p.measurement_radius is not None
                        and state.pose.distance(p.target) > p.measurement_radius
                    ):
                        if task.action_handle:
                            self.instrument.cancel(task.action_handle)
                        task.action_handle, task.phase = None, "navigate"
                        self._start_navigation(name, p)
                        self.event(
                            "science_position_correction",
                            robot=name,
                            poi=p.id,
                            reason="outside_measurement_range",
                            target_distance=state.pose.distance(p.target),
                            maximum_distance=p.measurement_radius,
                            station_error=state.pose.distance(p.approach),
                        )
                        continue
                    if state.pose.distance(p.approach) > robot.position_tolerance:
                        if task.action_handle:
                            self.instrument.cancel(task.action_handle)
                        task.action_handle, task.phase = None, "navigate"
                        self._start_navigation(name, p)
                        continue
                if task.action_handle is None:
                    task.action_handle = self.instrument.start(
                        name, p.kind, p.target, now
                    )
                if (
                    p.kind == Kind.ROCK_CANDIDATE
                    and state.pose.distance(p.target) > p.confirmation_radius
                ):
                    self.instrument.cancel(task.action_handle)
                    task.action_handle, task.phase = None, "navigate"
                    self._start_navigation(name, p)
                    continue
                feedback = self.instrument.tick(task.action_handle, state, now)
                if feedback.status == "failed":
                    self._finish(name, False, feedback.reason)
                elif feedback.status == "succeeded":
                    task.result, task.action_handle = feedback.result, None
                    if p.kind in {
                        Kind.ROCK_MEASUREMENT,
                        Kind.GROUND_MEASUREMENT,
                    } and isinstance(task.result, dict):
                        task.result = {
                            **task.result,
                            "target_distance": state.pose.distance(p.target),
                            "maximum_distance": p.measurement_radius,
                            "station_error": state.pose.distance(p.approach),
                        }
                    if p.kind == Kind.ROCK_CANDIDATE:
                        if (
                            not isinstance(feedback.result, dict)
                            or type(feedback.result.get("confirmed_measurement"))
                            is not bool
                        ):
                            self._finish(name, False, "confirmation_unavailable")
                            continue
                        confirmed = feedback.result["confirmed_measurement"]
                        self._finish(name, True)
                        self.event(
                            "candidate_confirmed"
                            if confirmed
                            else "candidate_rejected",
                            robot=name,
                            poi=p.id,
                        )
                    else:
                        self._finish(name, True)
        if self.sequence_planner is not None:
            self._prune_plans(self.maps.snapshot(now))
        return commands

    def command(self, message, now):
        if not math.isfinite(now) or now < self.now:
            raise ValueError("Mission time must be finite and monotonic")
        self.now = now
        op = message["op"]
        if op in {"approve", "reject"}:
            self.store.approve(message["poi"], op == "approve")
        elif op == "add_poi":
            pose = Pose(*message["pose"])
            grid = self.maps.snapshot(now)
            if not grid.contains(grid.cell(pose)):
                raise ValueError("POI must be inside the mission map")
            self.store.add(
                POI(message["id"], Kind(message.get("kind", "move")), pose, pose)
            )
        elif op == "remove_poi":
            p = self.store.items[message["poi"]]
            if p.claimed_by:
                self._finish(p.claimed_by, False, "poi_deactivated", cancelled=True)
            p.status = Status.REJECTED
        elif op in {"pause", "resume", "disable", "teleop", "assign"}:
            name = message["robot"]
            robot = self.robots[name]
            if op == "assign":
                p = self.store.items[message["poi"]]
                if p.kind not in robot.capabilities or p.status != Status.AVAILABLE:
                    raise ValueError("Assigned task must be available and supported")
            if op == "teleop":
                v, w = map(float, message["velocity"])
                lease = float(message.get("seconds", 1.0))
                if (
                    not all(math.isfinite(x) for x in (v, w, lease))
                    or not 0 < lease <= 5
                ):
                    raise ValueError(
                        "Teleop requires finite velocity and a lease in (0,5] seconds"
                    )
                velocity = (
                    max(-robot.max_linear, min(robot.max_linear, v)),
                    max(-robot.max_angular, min(robot.max_angular, w)),
                )
            self._finish(name, False, op, cancelled=True)
            self.modes[name] = {
                "pause": "paused",
                "disable": "disabled",
                "teleop": "teleop",
            }.get(op, "auto")
            if op == "teleop":
                self.manual[name] = velocity, now + lease
            elif op == "assign":
                p.assigned_to = name
                p.revision += 1
        else:
            raise ValueError(f"Unknown command: {op}")
        self.event("operator_command", command=message)
        if self.sequence_planner is not None:
            self._prune_plans(self.maps.snapshot(now))

    def summary(self):
        return dict(
            sim_time=self.now,
            modes=self.modes.copy(),
            pois=self.store.records(),
            exploration=self.exploration_status(),
            poi_planning=self.planning_status(),
            active={
                **{name: task.poi_id for name, task in self.executions.items()},
                **{name: "site_clearance" for name in self.yielding},
            },
        )
