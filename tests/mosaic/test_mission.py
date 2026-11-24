"""Mission invariants and module replacement, without assuming simulated motion."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from space_robotics_nav.mosaic.contracts import (
    ArrivalRegion,
    Detection,
    Health,
    Kind,
    Localization,
    NavigationFeedback,
    Pose,
    Robot,
)
from space_robotics_nav.mosaic.maps import KnownMap, KnownMapCrop, traversable
from space_robotics_nav.mosaic.mission import Mission
from space_robotics_nav.mosaic.navigation import (
    AStarPlanner,
    GridNavigator,
    PathTracker,
)
from space_robotics_nav.mosaic.poi import POI, POIStore, Status
from space_robotics_nav.mosaic.run import CommandFile
from space_robotics_nav.mosaic.providers import SimulatedInstrument
from space_robotics_nav.mosaic.contracts import InstrumentFeedback


def robot(
    name="scout", kinds=(Kind.MOVE, Kind.EXPLORE, Kind.ROCK_CANDIDATE), home=(-5, 4)
):
    return Robot(name, frozenset(kinds), Pose(*home), radius=0.4)


def poi(id="p", kind=Kind.MOVE, xy=(0, 4)):
    p = Pose(*xy)
    return POI(id, kind, p, p)


def test_atomic_claim_capabilities_expiry_and_late_completion():
    store = POIStore(retry_seconds=1)
    store.add(poi())
    robots = [robot(f"r{i}") for i in range(8)]
    with ThreadPoolExecutor(8) as pool:
        tokens = list(pool.map(lambda r: store.claim("p", r, 0), robots))
    assert sum(t is not None for t in tokens) == 1
    old = next(t for t in tokens if t)
    store.expire(6)
    new = store.claim("p", robots[0], 7)
    assert new and new != old
    assert not store.finish("p", old, 8, success=True)
    assert store.finish("p", new, 8, success=True)
    assert store.items["p"].status == Status.SUCCEEDED
    assert store.items["p"].attempts[0]["reason"] == "lease_expired"
    assert store.claim("p", robots[1], 9) is None
    store.add(poi("measurement", Kind.ROCK_MEASUREMENT))
    assert store.claim("measurement", robots[0], 9) is None


def test_failures_cooldown_and_help_do_not_loop_forever():
    store = POIStore(retry_seconds=10, max_failures=2)
    r = robot()
    store.add(poi())
    token = store.claim("p", r, 0)
    assert store.finish("p", token, 1, success=False, reason="unreachable")
    assert store.claim("p", r, 2) is None
    token = store.claim("p", r, 12)
    assert store.finish("p", token, 13, success=False, reason="unreachable")
    assert store.items["p"].status == Status.NEEDS_HELP


class LocalizationBuffer:
    def __init__(self, robots):
        self.states = {
            name: Localization(r.home, 0, "external_localization")
            for name, r in robots.items()
        }

    def get(self, name):
        return self.states[name]

    def advance(self, now):
        self.states = {name: replace(s, stamp=now) for name, s in self.states.items()}


class HealthBuffer:
    def __init__(self, robots):
        self.states = {name: Health() for name in robots}

    def get(self, name):
        return self.states[name]


class DetectorBuffer:
    def __init__(self):
        self.observations = []

    def poll(self, robot, state, now):
        return (
            [replace(d, stamp=now) for d in self.observations]
            if Kind.ROCK_CANDIDATE in robot.capabilities
            else []
        )


class ActionStub:
    """Navigation action server test double; no replacement vehicle dynamics."""

    def __init__(self):
        self.goals = []
        self.status = "running"

    def start(self, goal, now, **options):
        self.goals.append(goal)
        self.options = options

    def cancel(self):
        pass

    def tick(self, state, now):
        return NavigationFeedback(
            self.status, (0.2, 0) if self.status == "running" else (0, 0)
        )


def fixture_mission():
    a, b = robot(), robot("scientist", (Kind.MOVE, Kind.ROCK_MEASUREMENT), (-5, -4))
    robots = {r.name: r for r in (a, b)}
    locations, health, detector = (
        LocalizationBuffer(robots),
        HealthBuffer(robots),
        DetectorBuffer(),
    )
    maps = KnownMap([-10, -10, 10, 10], 0.25)
    local = KnownMapCrop(maps, locations, robots)
    navigators = {name: ActionStub() for name in robots}
    events = []
    mission = Mission(
        robots,
        POIStore(),
        locations,
        maps,
        local,
        detector,
        health,
        navigators,
        AStarPlanner(),
        emit=events.append,
        instrument=SimulatedInstrument(),
    )
    return mission, locations, health, detector, navigators, events


def test_close_confirmation_releases_science_and_scout_selects_next_task_independently():
    m, loc, _, _, nav, events = fixture_mission()
    observation = Detection(
        "rock", Pose(4, 4), Pose(2, 4), 0, "external", measurement_approach=Pose(4, 2)
    )
    p, _ = m.store.observe(observation)
    m.store.add(poi("next_exploration", Kind.EXPLORE, (-8, 4)))
    m.robots["scout"] = replace(
        m.robots["scout"], preferences={Kind.ROCK_CANDIDATE: 130, Kind.EXPLORE: 100}
    )
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(2, 4, 2.6))
    nav["scout"].status = "succeeded"
    m.tick(0)
    assert nav["scout"].options["arrival_region"] == ArrivalRegion(Pose(4, 4), 3)
    assert not nav["scout"].options["require_heading"]
    for now in (0.2, 2.4):
        loc.advance(now)
        m.tick(now)
    assert p.kind == Kind.ROCK_MEASUREMENT and p.status == Status.AVAILABLE
    assert p.confirmation_result["confirmed_measurement"]
    assert p.approach == Pose(4, 2)
    assert "scout" not in m.executions
    m.store.observe(replace(observation, stamp=2.4))
    assert p.approach == Pose(4, 2)
    nav["scout"].status = "running"
    loc.advance(2.6)
    m.tick(2.6)
    assert m.executions["scout"].poi_id == "next_exploration"
    assert m.executions["scientist"].poi_id == p.id
    assert nav["scout"].goals[-1] == Pose(-8, 4)
    assert nav["scientist"].goals[-1] == Pose(4, 2)
    nav["scientist"].status = "succeeded"
    loc.states["scientist"] = replace(loc.get("scientist"), pose=Pose(4, 2))
    for now in (2.8, 3, 5.2):
        loc.advance(now)
        m.tick(now)
    assert p.status == Status.SUCCEEDED and p.result["robot"] == "scientist"
    assert [a["kind"] for a in p.attempts] == [
        Kind.ROCK_CANDIDATE,
        Kind.ROCK_MEASUREMENT,
    ]
    assert any(e["event"] == "candidate_confirmed" for e in events)


def test_stale_localization_stops_and_other_capable_robot_reclaims():
    m, loc, _, _, nav, _ = fixture_mission()
    m.store.add(poi(xy=(-4, 4)))
    commands = m.tick(0)
    assert commands["scout"][0] > 0
    # Only scientist continues to publish localization. The mission gets no raw truth.
    loc.states["scientist"] = replace(loc.states["scientist"], stamp=1)
    commands = m.tick(1)
    assert commands["scout"] == (0, 0)
    assert m.store.items["p"].claimed_by == "scientist"
    assert m.store.items["p"].attempts[0]["reason"] == "localization_unavailable"


def test_manual_preemption_command_lease_and_resume():
    m, loc, _, _, _, _ = fixture_mission()
    m.store.add(poi())
    m.tick(0)
    m.command(
        {"op": "teleop", "robot": "scout", "velocity": [0.1, 0], "seconds": 0.5}, 0
    )
    assert m.store.items["p"].claimed_by is None
    assert m.store.items["p"].attempts[-1]["cancelled"]
    assert m.tick(0)["scout"] == (0.1, 0)
    loc.advance(0.6)
    assert m.tick(0.6)["scout"] == (0, 0)
    m.command({"op": "resume", "robot": "scout"}, 0.6)
    assert m.modes["scout"] == "auto"


def test_reobserved_candidate_updates_goal_without_duplicate_or_reopening_completed_task():
    m, loc, _, _, nav, _ = fixture_mission()
    observation = Detection("rock", Pose(4, 4), Pose(2, 4), 0, "external")
    p, _ = m.store.observe(observation)
    m.tick(0)
    updated = replace(observation, target=Pose(4, 5), approach=Pose(2, 5), stamp=0.2)
    assert not m.store.observe(updated)[1]
    loc.advance(0.2)
    m.tick(0.2)
    assert len(m.store.items) == 1 and nav["scout"].goals[-1] == Pose(2, 5)
    p.status = Status.REJECTED
    m.store.observe(updated)
    assert p.status == Status.REJECTED


def test_exploration_waits_ten_seconds_without_turning_and_registers_a_candidate_batch():
    m, loc, _, detector, nav, events = fixture_mission()
    m.store.add(poi("survey", Kind.EXPLORE))
    detector.observations = [
        Detection("a", Pose(4, 4), Pose(2, 4), 0, "external"),
        Detection("b", Pose(4, 6), Pose(2, 6), 0, "external"),
    ]
    m.tick(0)
    assert set(m.store.items) == {"survey"}  # no transit detection
    nav["scout"].status = "succeeded"
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(0, 4, 2.8))
    loc.advance(1)
    m.tick(1)
    assert m.executions["scout"].phase == "analyze"
    assert not nav["scout"].options["require_heading"]
    for now in (1.2, 3, 5, 7, 9, 10.8):
        loc.advance(now)
        commands = m.tick(now)
        assert commands["scout"] == (0, 0)
        assert m.executions["scout"].poi_id == "survey"
        assert set(m.store.items) == {"survey"}
    loc.advance(11)
    commands = m.tick(11)
    assert commands["scout"] == (0, 0)
    assert m.store.items["survey"].status == Status.SUCCEEDED
    assert m.store.items["survey"].result["candidates"] == ["rock:a", "rock:b"]
    assert m.store.items["rock:a"].source_exploration == "survey"
    assert m.store.items["rock:b"].discovered_by == "scout"
    assert not any(e["event"] in {"task_preempted", "scan_started"} for e in events)
    loc.advance(11.2)
    m.tick(11.2)
    assert m.executions["scout"].poi_id == "rock:a"


def test_no_candidate_continues_to_next_exploration_and_does_not_preempt_current_task():
    m, loc, _, _, nav, events = fixture_mission()
    m.store.add(poi("first", Kind.EXPLORE))
    m.store.add(poi("second", Kind.EXPLORE, (-8, 4)))
    m.tick(0)
    assert m.executions["scout"].poi_id == "second"  # shorter path wins
    nav["scout"].status = "succeeded"
    loc.advance(0.2)
    m.tick(0.2)
    candidate, _ = m.store.observe(
        Detection("new", Pose(4, 4), Pose(2, 4), 0.2, "external")
    )
    loc.advance(1)
    m.tick(1)
    assert m.executions["scout"].poi_id == "second"
    for now in (4, 7, 10.2):
        loc.advance(now)
        m.tick(now)
    assert m.store.items["second"].result["candidates"] == []
    assert m.store.items["second"].status == Status.SUCCEEDED
    nav["scout"].status = "running"
    m.robots["scout"] = replace(
        m.robots["scout"], preferences={Kind.EXPLORE: 140, Kind.ROCK_CANDIDATE: 100}
    )
    loc.advance(10.4)
    m.tick(10.4)
    assert m.executions["scout"].poi_id == "first"
    assert candidate.status == Status.AVAILABLE
    assert not any(e["event"] == "task_preempted" for e in events)


@pytest.mark.parametrize("confirmed", [True, False, None])
def test_candidate_requires_close_range_and_explicit_confirmation_result(confirmed):
    class ConfirmInstrument:
        def start(self, *args):
            return "confirmation"

        def tick(self, *args):
            return InstrumentFeedback("succeeded", {"confirmed_measurement": confirmed})

        def cancel(self, *args):
            pass

    m, loc, _, _, nav, _ = fixture_mission()
    m.instrument = ConfirmInstrument()
    p, _ = m.store.observe(Detection("sample", Pose(4, 4), Pose(2, 4), 0, "test"))
    nav["scout"].status = "succeeded"
    m.tick(0)
    loc.advance(0.2)
    m.tick(0.2)
    assert p.kind == Kind.ROCK_CANDIDATE and p.confirmation_result is None
    assert (
        m.executions["scout"].phase == "navigate"
    )  # outside the region, even if an external navigator reports success
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(2, 4, -2.7))
    for now in (0.4, 0.6):
        loc.advance(now)
        m.tick(now)
    if confirmed is None:
        assert p.attempts[-1]["reason"] == "confirmation_unavailable"
    elif confirmed:
        assert p.kind == Kind.ROCK_MEASUREMENT and p.status == Status.AVAILABLE
    else:
        assert p.status == Status.REJECTED and p.kind == Kind.ROCK_CANDIDATE


def test_arrival_region_accepts_any_safe_point_inside_radius_without_yaw_alignment():
    r = robot(home=(1.2, 0))
    loc = LocalizationBuffer({r.name: r})
    maps = KnownMap([-10, -10, 10, 10], 0.25)
    nav = GridNavigator(
        r, maps, KnownMapCrop(maps, loc, {r.name: r}), AStarPlanner(), PathTracker()
    )
    state = replace(loc.get(r.name), pose=Pose(1.2, 0, 2.8))
    nav.start(
        Pose(2.5, 0),
        0,
        arrival_region=ArrivalRegion(Pose(4, 0), 3),
        require_heading=False,
    )
    assert nav.tick(state, 0).status == "succeeded"
    assert nav.tick(replace(state, linear_speed=0.1), 0).velocity == (0, 0)
    assert nav.tick(replace(state, linear_speed=0.1), 0).status == "running"
    nav.start(Pose(1.2, 0, 0), 0, require_heading=False)
    assert nav.tick(state, 0).status == "succeeded"


def test_scientist_starts_navigation_while_idle_scout_yields():
    m, loc, _, _, nav, events = fixture_mission()
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(2.5, 4))
    m.store.add(poi("science", Kind.ROCK_MEASUREMENT, (3, 4)))
    commands = m.tick(0)
    assert m.executions["scientist"].phase == "navigate"
    assert commands["scientist"][0] > 0
    assert nav["scientist"].goals[-1] == Pose(3, 4)
    assert "scout" in m.yielding
    assert nav["scout"].goals[-1] == m.robots["scout"].home
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(1, 4))
    loc.advance(0.2)
    commands = m.tick(0.2)
    assert "scout" not in m.yielding
    assert m.executions["scientist"].phase == "navigate"
    assert commands["scout"] == (0, 0) and commands["scientist"][0] > 0
    assert not any(e["event"] == "science_waiting_for_clearance" for e in events)


def test_new_scout_task_overrides_idle_clearance_without_waiting_for_science():
    m, loc, _, _, nav, _ = fixture_mission()
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(2.5, 4))
    m.store.add(poi("science", Kind.ROCK_MEASUREMENT, (3, 4)))
    m.tick(0)
    assert "scout" in m.yielding
    m.store.add(poi("next", Kind.EXPLORE, (-8, 4)))
    loc.advance(0.2)
    m.tick(0.2)
    assert m.executions["scout"].poi_id == "next" and "scout" not in m.yielding
    assert nav["scout"].goals[-1] == Pose(-8, 4)
    assert m.executions["scientist"].phase == "navigate"


def test_forward_only_navigation_turns_before_driving_to_a_goal_behind():
    r = replace(robot(home=(0, 0)), allow_reverse=False)
    grid = KnownMap([-10, -10, 10, 10], 0.25).snapshot(0)
    state = Localization(Pose(0, 0), 0, "test")
    command = PathTracker().command(state, Pose(-2, 0), Pose(-3, 0), grid, r)
    assert command[0] == 0 and abs(command[1]) > 0
    reverse = PathTracker().command(
        state, Pose(-2, 0), Pose(-3, 0), grid, replace(r, allow_reverse=True)
    )
    assert reverse[0] < 0


def test_lateral_sliding_does_not_count_as_stopped_for_measurement():
    instrument = SimulatedInstrument()
    handle = instrument.start("science", Kind.ROCK_MEASUREMENT, Pose(0, 0), 0)
    sliding = Localization(Pose(0, 0), 0, "test", lateral_speed=0.12)
    assert instrument.tick(handle, sliding, 0).status == "running"
    assert instrument.tick(handle, replace(sliding, stamp=3), 3).status == "running"
    stationary = replace(sliding, lateral_speed=0, stamp=4)
    assert instrument.tick(handle, stationary, 4).status == "running"
    result = instrument.tick(handle, replace(stationary, stamp=6), 6)
    assert result.status == "succeeded" and result.result["settled_seconds"] >= 2


def test_real_navigation_replans_on_map_update_and_stops_on_stale_map():
    r = robot(home=(-5, 0))
    robots = {r.name: r}
    loc = LocalizationBuffer(robots)
    maps = KnownMap([-10, -10, 10, 10], 0.25)
    local = KnownMapCrop(maps, loc, robots)
    nav = GridNavigator(r, maps, local, AStarPlanner(), PathTracker())
    nav.start(Pose(5, 0), 0)
    assert nav.tick(loc.get(r.name), 0).status == "running"
    previous = nav.path
    maps.block([-0.5, -1, 0.5, 1])
    assert nav.tick(loc.get(r.name), 0).status == "running"
    assert previous != nav.path and max(abs(p.y) for p in nav.path) > 1
    grid = maps.snapshot(0)
    maps.replace(replace(grid, version=grid.version + 1, static=False, stamp=0))
    loc.advance(1)
    result = nav.tick(loc.get(r.name), 1)
    assert result.status == "failed" and result.reason == "map_unavailable"
    assert result.velocity == (0, 0)


def test_large_map_route_allows_normal_travel_but_replans_cannot_renew_its_deadline():
    r = robot(home=(-40, 0))
    robots = {r.name: r}
    loc = LocalizationBuffer(robots)
    maps = KnownMap([-50, -10, 50, 10], 0.25)
    local = KnownMapCrop(maps, loc, robots)
    nav = GridNavigator(r, maps, local, AStarPlanner(), PathTracker())
    nav.start(Pose(40, 0), 0, require_heading=False)
    assert nav.tick(loc.get(r.name), 0).status == "running"
    budget = nav.navigation_budget
    assert budget > 80 / r.max_linear
    loc.states[r.name] = Localization(Pose(0, 0), 181, "test")
    assert nav.tick(loc.get(r.name), 181).status == "running"
    maps.block([20, 2, 21, 3])
    assert nav.tick(loc.get(r.name), 181).status == "running"
    assert nav.navigation_budget == budget
    loc.states[r.name] = Localization(Pose(20, 0), budget + 1, "test")
    result = nav.tick(loc.get(r.name), budget + 1)
    assert result.status == "failed" and result.reason == "navigation_timeout"


def test_task_budget_follows_long_route_but_station_updates_do_not_reset_task_age():
    m, loc, _, _, nav, _ = fixture_mission()
    m.store.add(poi())
    m.tick(0)
    nav["scout"].navigation_budget = 600.0
    for now in range(1, 401):
        loc.advance(now)
        m.tick(now)
    assert m.store.items["p"].status == Status.RUNNING
    m.store.items["p"].revision += 1
    for now in range(401, 782):
        loc.advance(now)
        m.tick(now)
    assert m.store.items["p"].attempts[-1]["reason"] == "task_timeout"


def test_temporary_global_blockage_keeps_navigation_running_until_clear_or_timeout():
    r = robot(home=(-5, 0))
    loc = LocalizationBuffer({r.name: r})
    maps = KnownMap([-10, -10, 10, 10], 0.25)
    local = KnownMapCrop(maps, loc, {r.name: r})
    nav = GridNavigator(r, maps, local, AStarPlanner(), PathTracker())
    nav.start(Pose(5, 0), 0)
    assert nav.tick(loc.get(r.name), 0).status == "running"
    maps.block([4, -1, 6, 1])
    for now in (1, 10, 30):
        loc.advance(now)
        feedback = nav.tick(loc.get(r.name), now)
        assert feedback.status == "running" and feedback.velocity == (0, 0)
        assert feedback.reason == "waiting_for_global_path"
    maps.block([4, -1, 6, 1], blocked=False)
    loc.advance(31)
    assert nav.tick(loc.get(r.name), 31).velocity[0] > 0
    assert not nav.waiting_reason
    maps.block([4, -1, 6, 1])
    loc.advance(32)
    assert nav.tick(loc.get(r.name), 32).status == "running"
    loc.advance(78)
    feedback = nav.tick(loc.get(r.name), 78)
    assert feedback.status == "failed" and feedback.reason == "no_global_path"
    assert feedback.velocity == (0, 0)


def test_local_occupancy_stops_before_peer_without_polluting_global_map():
    a, b = robot(home=(0, 0)), robot("peer", home=(1, 0))
    loc = LocalizationBuffer({a.name: a, b.name: b})
    maps = KnownMap([-10, -10, 10, 10], 0.1)
    local = KnownMapCrop(maps, loc, {a.name: a, b.name: b})
    patch = local.snapshot(a, a.home, 0)
    assert not PathTracker.safe(loc.get(a.name), (0.25, 0), patch, a.radius)
    assert not maps.snapshot(0).blocked.any()
    assert patch.source == "known_map_crop+peer_occupancy"


def test_occupied_station_allows_approach_then_local_wait_and_resume():
    scientist = robot("scientist", home=(-5, 0))
    scout = robot("scout", home=(0, 0))
    robots = {r.name: r for r in (scientist, scout)}
    loc = LocalizationBuffer(robots)
    maps = KnownMap([-10, -10, 10, 10], 0.1)
    nav = GridNavigator(
        scientist, maps, KnownMapCrop(maps, loc, robots), AStarPlanner(), PathTracker()
    )
    goal = Pose(0, 0)
    nav.start(goal, 0)
    assert nav.tick(loc.get("scientist"), 0).velocity[0] > 0

    # Exercise the same occupied destination from a nearby safe stopping point.
    loc.states["scientist"] = replace(loc.get("scientist"), pose=Pose(-1.22, 0))
    loc.advance(0.2)
    nav.start(goal, 0.2)
    for now in (0.2, 2.4):
        loc.advance(now)
        feedback = nav.tick(loc.get("scientist"), now)
        assert feedback.status == "running" and feedback.velocity == (0, 0)
    assert nav.waiting_reason == "Waiting for a collision-free local path"
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(0, 5))
    loc.advance(2.6)
    assert nav.tick(loc.get("scientist"), 2.6).velocity[0] > 0
    assert not nav.waiting_reason and nav.goal == goal


def test_maps_are_immutable_and_frame_errors_are_explicit():
    maps = KnownMap([-2, -2, 2, 2], 0.1)
    with pytest.raises(ValueError):
        maps.snapshot(0).blocked[0, 0] = True
    with pytest.raises(ValueError, match="frames"):
        Pose(0, 0).distance(Pose(0, 0, frame="camera"))
    assert not traversable(maps.snapshot(0), 0.4)[0, 0]


def test_local_replanning_routes_around_stationary_teammate():
    a, b = robot(home=(0, 0)), robot("peer", home=(1.4, 0))
    robots = {a.name: a, b.name: b}
    loc = LocalizationBuffer(robots)
    maps = KnownMap([-10, -10, 10, 10], 0.1)
    local = KnownMapCrop(maps, loc, robots)
    nav = GridNavigator(a, maps, local, AStarPlanner(), PathTracker())
    nav.start(Pose(5, 0), 0)
    # A short, slow approach is physically safe before reaching the peer's
    # buffer. Once there, stop and route around it rather than waiting forever.
    first = nav.tick(loc.get(a.name), 0).velocity
    assert 0 <= first[0] <= 0.05
    loc.states[a.name] = replace(loc.get(a.name), pose=Pose(0.15, 0))
    loc.advance(2.1)
    nav.tick(loc.get(a.name), 2.1)
    assert nav.local_replans == 1
    assert max(abs(p.y) for p in nav.path) > 1
    assert not maps.snapshot(0).blocked.any()


def test_replace_instrument_without_changing_mission_state_machine():
    class ExternalInstrument:
        def start(self, robot, kind, target, now):
            return "external-request"

        def tick(self, handle, state, now):
            assert handle == "external-request"
            return InstrumentFeedback(
                "succeeded", {"simulated": False, "spectrum": [1, 2, 3]}
            )

        def cancel(self, handle):
            pass

    m, loc, _, _, nav, _ = fixture_mission()
    m.instrument = ExternalInstrument()
    m.store.add(poi(kind=Kind.ROCK_MEASUREMENT, xy=(0, -4)))
    nav["scientist"].status = "succeeded"
    loc.states["scientist"] = replace(loc.get("scientist"), pose=Pose(0, -4))
    m.tick(0)
    loc.advance(0.2)
    m.tick(0.2)
    assert m.store.items["p"].status == Status.SUCCEEDED
    assert m.store.items["p"].result["spectrum"] == [1, 2, 3]
    assert m.store.items["p"].result["simulated"] is False
    assert m.store.items["p"].result["station_error"] == 0


def test_failed_external_navigator_cannot_leave_a_motion_command():
    m, _, _, _, nav, _ = fixture_mission()
    m.store.add(poi())
    nav["scout"].tick = lambda state, now: NavigationFeedback(
        "failed", (0.2, 0), "external_failure"
    )
    assert m.tick(0)["scout"] == (0, 0)
    assert m.store.items["p"].attempts[-1]["reason"] == "external_failure"


def test_reverse_safety_checks_the_space_behind_the_robot():
    grid = KnownMap([-5, -5, 5, 5], 0.1, [[-1.3, -0.3, -1, 0.3]]).snapshot(0)
    state = Localization(Pose(0, 0), 0, "external")
    assert PathTracker.safe(state, (0.25, 0), grid, 0.4)
    assert not PathTracker.safe(state, (-0.25, 0), grid, 0.4)


def test_invalid_peer_updates_do_not_remove_last_known_occupancy():
    a, b = robot(home=(0, 0)), robot("peer", home=(2, 0))
    robots = {a.name: a, b.name: b}
    loc = LocalizationBuffer(robots)
    maps = KnownMap([-10, -10, 10, 10], 0.1)
    provider = KnownMapCrop(maps, loc, robots)
    before = provider.snapshot(a, a.home, 0)
    loc.states[b.name] = Localization(Pose(9, 9), 1, "external", valid=False)
    after = provider.snapshot(a, a.home, 1)
    np.testing.assert_array_equal(before.blocked, after.blocked)


def test_command_file_handles_partial_lines_and_invalid_json(tmp_path):
    path = tmp_path / "commands.jsonl"
    file = CommandFile(path)
    path.write_text('{"op":"pause"')
    assert file.read() == []
    with path.open("a") as out:
        out.write(',"robot":"scout"}\nnot json\n')
    messages = file.read()
    assert messages[0] == {"op": "pause", "robot": "scout"}
    assert messages[1]["op"] == "invalid"
    assert file.read() == []
