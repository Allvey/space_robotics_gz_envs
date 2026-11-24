"""Sequence coordination, causal snapshots and nonblocking mission integration."""

from dataclasses import replace
from types import SimpleNamespace
import threading
import time

import pytest

from space_robotics_nav.mosaic.contracts import (
    Health,
    Kind,
    Localization,
    NavigationFeedback,
    Pose,
    Robot,
)
from space_robotics_nav.mosaic.maps import KnownMap, KnownMapCrop
from space_robotics_nav.mosaic.mission import Mission
from space_robotics_nav.mosaic.navigation import AStarPlanner
from space_robotics_nav.mosaic.poi import POI, POIStore, Status
from space_robotics_nav.mosaic.poi_planning import (
    GreedyPOIPlanner,
    PlanningSettings,
    PlanningSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from space_robotics_nav.mosaic.providers import SimulatedInstrument
from space_robotics_nav.mosaic.scene import measurement_station_from


def task(name, x, y=0, kind=Kind.MOVE):
    return TaskSnapshot(name, kind, Pose(x, y), Pose(x, y))


def robot(name, x, y=0):
    return Robot(
        name,
        frozenset({Kind.MOVE, Kind.EXPLORE, Kind.ROCK_CANDIDATE}),
        Pose(x, y),
        radius=0.4,
    )


def snapshot(robots, tasks, owners=None):
    grid = KnownMap([-20, -15, 20, 15], 0.25).grid
    return PlanningSnapshot(
        0,
        grid,
        tuple(
            r if isinstance(r, RobotSnapshot) else RobotSnapshot(r, r.home, grid)
            for r in robots
        ),
        tuple(tasks),
        previous_owners=owners or {},
    )


def test_depth_rolls_positions_forward_and_coordinates_unique_discounted_intentions():
    tasks = [
        task("left", -8),
        task("right", 8),
        task("upper", 0, 4),
        task("lower", 0, -4),
        task("nw", -5, 7),
        task("ne", 5, 7),
    ]
    source = snapshot([robot("a", -10), robot("b", 10)], tasks)
    result = GreedyPOIPlanner(PlanningSettings(wall_budget=5)).plan(source)
    shallow = GreedyPOIPlanner(PlanningSettings(depth=1, wall_budget=5)).plan(source)
    ids = []
    assert [len(p) for p in result.plans.values()] == [3, 3]
    for name, steps in result.plans.items():
        assert steps[0].poi_id == shallow.plans[name][0].poi_id
        for index, step in enumerate(steps):
            ids.append(step.poi_id)
            assert step.discounted_utility == pytest.approx(step.utility * 0.8**index)
            assert step.expected_finish >= step.expected_arrival >= step.expected_start
            if index:
                assert step.start == steps[index - 1].approach
                assert step.expected_start == steps[index - 1].expected_finish
    assert len(set(ids)) == len(ids) == len(tasks)
    assert source.tasks == tuple(tasks)  # prediction does not complete/create tasks


def test_idle_current_opportunity_takes_precedence_over_a_future_owner():
    a, b = robot("a", -10), robot("b", 0.5)
    source = snapshot([a, b], [])
    busy = RobotSnapshot(
        a, a.home, source.grid, task("executing", -8), Pose(-8, 0), 100
    )
    source = replace(
        source,
        robots=(busy, source.robots[1]),
        tasks=(task("shared", 0), task("other", -5, 4)),
        previous_owners={"shared": "a"},
    )
    result = GreedyPOIPlanner(PlanningSettings(switch_margin=100, wall_budget=5)).plan(
        source
    )
    assert result.plans["b"][0].poi_id == "shared"
    assert all(s.poi_id != "shared" for s in result.plans["a"])
    assert result.plans["a"][0].current


def test_future_owner_hysteresis_prevents_small_swaps_but_yields_to_a_better_bid():
    a, b = robot("a", -3), robot("b", 4)
    base = snapshot([a, b], [task("future", 0)], {"future": "b"})
    busy = tuple(
        replace(
            r,
            active=task("busy_" + r.robot.name, r.pose.x),
            active_end=r.pose,
            active_remaining=5,
        )
        for r in base.robots
    )
    engine = GreedyPOIPlanner(PlanningSettings(wall_budget=5))
    result = engine.plan(replace(base, robots=busy))
    assert result.plans["b"][1].poi_id == "future"
    busy = (
        replace(busy[0], active_end=Pose(0, 0)),
        replace(busy[1], active_end=Pose(12, 0)),
    )
    result = engine.plan(replace(base, robots=busy))
    assert result.plans["a"][1].poi_id == "future"


def test_science_station_is_computed_from_the_explicit_predicted_pose():
    grid = KnownMap([-12, -8, 12, 8], 0.25).grid
    r = Robot("science", frozenset({Kind.ROCK_MEASUREMENT}), Pose(-8, 0), radius=0.4)
    p = replace(task("rock", 0, kind=Kind.ROCK_MEASUREMENT), measurement_radius=3.5)
    left = measurement_station_from(r, p, Pose(-8, 0), grid, AStarPlanner(), 0.2)
    right = measurement_station_from(r, p, Pose(8, 0), grid, AStarPlanner(), 0.2)
    assert left.x < 0 < right.x
    assert left.distance(p.target) <= 3.5 - r.position_tolerance - 0.05
    assert right.distance(p.target) <= 3.5 - r.position_tolerance - 0.05


@pytest.fixture
def mission():
    robots = {r.name: r for r in [robot("a", -10), robot("b", 10)]}
    states = {name: Localization(r.home, 0, "test") for name, r in robots.items()}
    locations = SimpleNamespace(get=states.__getitem__)
    maps = KnownMap([-20, -15, 20, 15], 0.25)
    local = KnownMapCrop(maps, locations, robots)
    store = POIStore()
    for p in [
        task("left", -8),
        task("right", 8),
        task("upper", 0, 4),
        task("lower", 0, -4),
        task("nw", -5, 7),
        task("ne", 5, 7),
    ]:
        store.add(POI(p.id, p.kind, p.target, p.approach))
    nav = {
        name: SimpleNamespace(
            start=lambda *a, **k: None,
            cancel=lambda: None,
            tick=lambda s, t: NavigationFeedback("running", (0.1, 0)),
        )
        for name in robots
    }
    events = []
    m = Mission(
        robots,
        store,
        locations,
        maps,
        local,
        SimpleNamespace(poll=lambda *args: []),
        SimpleNamespace(get=lambda name: Health()),
        nav,
        AStarPlanner(),
        instrument=SimulatedInstrument(),
        emit=events.append,
        poi_planning={"depth": 3, "wall_budget": 5},
    )
    yield m, states, events
    m.close()


def accept_first_plan(m):
    m.tick(0)
    assert m.sequence_planner.future is not None
    m.sequence_planner.future.result(timeout=10)  # wait outside the control tick
    m.tick(0)


def test_only_heads_are_claimed_and_worker_cannot_mutate_poi_store(mission):
    m, _, _ = mission
    m.tick(0)
    result = m.sequence_planner.future.result(timeout=10)
    assert len(m.store.items) == 6 and all(
        p.status == Status.AVAILABLE for p in m.store.items.values()
    )
    assert all(p.token is None for p in m.store.items.values())
    assert sum(len(p) for p in result.plans.values()) == 6
    m.tick(0)
    assert len(m.executions) == 2
    status = m.planning_status()
    assert len(status["future_owners"]) == 4
    for name, steps in m.poi_plans.items():
        assert steps[0].current and m.store.items[steps[0].poi_id].claimed_by == name
        for step in steps[1:]:
            p = m.store.items[step.poi_id]
            assert p.status == Status.AVAILABLE and p.token is None


def test_pending_snapshot_is_discarded_on_map_change(mission):
    m, _, events = mission
    m.tick(0)
    m.sequence_planner.future.result(timeout=10)
    m.maps.block([-1, -1, 1, 1])
    m.tick(0)
    assert not m.executions
    assert any(e["event"] == "poi_plan_discarded" for e in events)


def test_expiry_and_pause_release_intentions_without_canceling_other_current_task(
    mission,
):
    m, _, _ = mission
    accept_first_plan(m)
    other = m.executions["b"]
    m.command({"op": "pause", "robot": "a"}, 0)
    m._prune_plans(m.maps.grid)
    assert not m.poi_plans["a"]
    assert m.executions["b"] is other
    m.now = m.plan_valid_until + 0.01
    m._prune_plans(m.maps.grid)
    assert not m.planning_status()["future_owners"]
    assert m.executions["b"] is other


def test_planning_does_not_block_control_tick(mission):
    m, _, _ = mission
    entered, release = threading.Event(), threading.Event()
    original = m.sequence_planner.engine.plan

    def gated(snapshot, cancelled):
        entered.set()
        assert release.wait(5)
        return original(snapshot, cancelled)

    m.sequence_planner.engine.plan = gated
    try:
        m.tick(0)
        assert entered.wait(2)
        start = time.monotonic()
        assert all(v == (0, 0) for v in m.tick(0).values())
        assert time.monotonic() - start < 0.1
        assert not m.executions
    finally:
        release.set()


def test_bounded_partial_plan_cannot_publish_duplicate_or_fictional_targets():
    source = snapshot(
        [robot("a", -10), robot("b", 10)], [task(str(i), i - 5, 4) for i in range(10)]
    )
    result = GreedyPOIPlanner(PlanningSettings(max_queries=1, wall_budget=5)).plan(
        source
    )
    assert result.truncated and result.queries == 1
    ids = [s.poi_id for steps in result.plans.values() for s in steps]
    assert len(ids) == len(set(ids)) and set(ids) <= {t.id for t in source.tasks}


def test_unknown_future_measurements_are_not_created_by_candidate_prediction():
    source = snapshot(
        [robot("scout", -10)], [task("candidate", -6, kind=Kind.ROCK_CANDIDATE)]
    )
    result = GreedyPOIPlanner(PlanningSettings(wall_budget=5)).plan(source)
    assert len(result.plans["scout"]) == 1
    assert result.plans["scout"][0].kind == Kind.ROCK_CANDIDATE


def test_cooperative_cancellation_discards_incomplete_search():
    source = snapshot([robot("a", -10)], [task("p", 8)])
    cancelled = threading.Event()
    cancelled.set()
    result = GreedyPOIPlanner(PlanningSettings(wall_budget=5)).plan(source, cancelled)
    assert result.truncated and result.plans["a"] == [] and result.queries == 0


def test_operator_reassignment_invalidates_pending_plan(mission):
    m, _, _ = mission
    m.tick(0)
    result = m.sequence_planner.future.result(timeout=10)
    first = result.plans["a"][0].poi_id
    m.command({"op": "assign", "robot": "b", "poi": first}, 0)
    m.tick(0)
    assert m.store.items[first].claimed_by is None
    assert all(s.poi_id != first for s in m.poi_plans["a"])


def test_published_plans_do_not_preempt_active_navigation(mission):
    m, _, _ = mission
    accept_first_plan(m)
    original = dict(m.executions)
    m.store.add(
        POI("new_high_value_candidate", Kind.ROCK_CANDIDATE, Pose(-9, 0), Pose(-9, 0))
    )
    m.tick(0)
    if m.sequence_planner.future is not None:
        m.sequence_planner.future.result(timeout=10)
        m.tick(0)
    assert m.executions == original
    assert m.store.items["new_high_value_candidate"].claimed_by is None


@pytest.mark.parametrize(
    "options",
    [
        {"depth": 0},
        {"depth": True},
        {"discount": 0},
        {"discount": 1.1},
        {"intention_ttl": 2},
        {"wall_budget": float("nan")},
        {"max_queries": 0},
    ],
)
def test_invalid_planning_settings_are_rejected(options):
    with pytest.raises(ValueError):
        PlanningSettings(**options)
