"""Continuous investigation tasks and measurement validity, independent of physics."""

from dataclasses import replace

import pytest

from space_robotics_nav.mosaic.contracts import Detection, Kind, Pose
from space_robotics_nav.mosaic.exploration import LayoutExploration
from space_robotics_nav.mosaic.poi import POI, Status
from test_mission import fixture_mission


def attach_source(m, spacing=5):
    m.exploration = LayoutExploration(
        m.planner,
        m.local_maps,
        m.localization,
        m.robots,
        spacing=spacing,
        clearance=0.1,
    )


def test_empty_analysis_adds_new_exploration_tasks_and_keeps_mission_running():
    m, loc, _, _, nav, events = fixture_mission()
    attach_source(m)
    m.tick(0)
    original = set(m.store.items)
    assert len(original) == 2 and all(
        p.task_source == "known_layout_stations" for p in m.store.items.values()
    )
    first = m.store.items[m.executions["scout"].poi_id]
    loc.states["scout"] = replace(loc.get("scout"), pose=first.approach)
    nav["scout"].status = "succeeded"
    for now in (0.2, 2, 4, 6, 8, 10.2):
        loc.advance(now)
        m.tick(now)
    assert first.status == Status.SUCCEEDED and first.result["candidates"] == []
    nav["scout"].status = "running"
    for now in (10.4, 12.4):
        loc.advance(now)
        m.tick(now)
    assert set(m.store.items) > original
    assert m.executions["scout"].poi_id != first.id and not m.is_complete()
    assert any(e["event"] == "exploration_poi_generated" for e in events)
    assert all(
        p.approach.distance(first.approach) >= 4
        for p in m.store.items.values()
        if p.id not in original
    )


def test_all_seeded_pois_finished_does_not_mean_exploration_finished():
    m, _, _, _, _, _ = fixture_mission()
    p = POI("old", Kind.EXPLORE, Pose(0, 0), Pose(0, 0), status=Status.SUCCEEDED)
    m.store.add(p)
    attach_source(m)
    assert not m.is_complete()
    m.tick(0)
    assert any(
        p.status in {Status.CLAIMED, Status.RUNNING, Status.AVAILABLE}
        for p in m.store.items.values()
    )


def test_generated_exploration_competes_with_candidates_without_manual_assignment_bonus():
    m, _, _, _, _, events = fixture_mission()
    attach_source(m)
    m.store.add(POI("candidate", Kind.ROCK_CANDIDATE, Pose(-4, 4), Pose(-4, 4)))
    m.tick(0)
    assert any(p.kind == Kind.EXPLORE for p in m.store.items.values())
    assert all(
        p.assigned_to is None
        for p in m.store.items.values()
        if p.task_source == "known_layout_stations"
    )
    assert m.executions["scout"].poi_id == "candidate"
    assert all(e["utility"] <= 100 for e in events if e["event"] == "task_claimed")


def test_source_exhaustion_is_distinct_from_temporarily_unreachable_stations():
    m, loc, _, _, _, _ = fixture_mission()
    attach_source(m, spacing=10)
    # All four investigation stations have already been assigned or investigated.
    for x in (-5, 5):
        for y in (-5, 5):
            m.store.add(
                POI(
                    f"{x}:{y}",
                    Kind.EXPLORE,
                    Pose(x, y),
                    Pose(x, y),
                    status=Status.SUCCEEDED,
                )
            )
    m.tick(0)
    assert m.exploration_status()["state"] == "exhausted" and m.is_complete()
    # Unvisited stations across a blocking wall must not be called explored.
    m, loc, _, _, _, _ = fixture_mission()
    attach_source(m, spacing=10)
    m.maps.block([-0.5, -10, 0.5, 10])
    for y in (-5, 5):
        m.store.add(
            POI(
                f"left:{y}",
                Kind.EXPLORE,
                Pose(-5, y),
                Pose(-5, y),
                status=Status.SUCCEEDED,
            )
        )
    m.tick(0)
    assert m.exploration_status()["state"] == "temporarily_unreachable"
    assert not m.is_complete()


class NoStation:
    def plan(self, *args):
        raise ValueError("No safe station inside instrument range")


def test_unreachable_science_is_retained_and_does_not_stop_exploration():
    m, _, _, _, _, events = fixture_mission()
    attach_source(m)
    m.measurement_stations = NoStation()
    p = POI(
        "rock", Kind.ROCK_MEASUREMENT, Pose(4, 4), Pose(2, 4), measurement_radius=3.5
    )
    m.store.add(p)
    m.tick(0)
    assert (
        p.status == Status.NEEDS_HELP
        and p.blocked_reason == "measurement_station_unreachable"
    )
    assert p.result is None and "scientist" not in m.executions
    assert "scout" in m.executions and not m.is_complete()
    assert "not measured" in m.poi_wait_reason(p)
    assert any(e["event"] == "measurement_station_unreachable" for e in events)


def test_reobservation_without_a_station_preserves_the_planned_science_station():
    m, _, _, _, _, _ = fixture_mission()
    observation = Detection(
        "rock", Pose(4, 4), Pose(2, 4), 0, "detector", measurement_radius=3.5
    )
    p, _ = m.store.observe(observation)
    p.kind, p.measurement_approach, p.approach = (
        Kind.ROCK_MEASUREMENT,
        Pose(4, 2),
        Pose(4, 2),
    )
    p.status = Status.RUNNING
    revision = p.revision
    m.store.observe(replace(observation, stamp=10))
    assert p.measurement_approach == Pose(4, 2) and p.approach == Pose(4, 2)
    assert p.status == Status.RUNNING and p.revision == revision


@pytest.mark.parametrize("outside_range", [True, False])
def test_science_requires_actual_arrival_and_instrument_range_even_if_navigator_reports_success(
    outside_range,
):
    class MustNotMeasure:
        def start(self, *args):
            raise AssertionError("Instrument must not start at an invalid position")

    m, loc, _, _, nav, events = fixture_mission()
    m.instrument = MustNotMeasure()
    p = POI(
        "rock",
        Kind.ROCK_MEASUREMENT,
        Pose(4, 2),
        Pose(4, 0),
        measurement_approach=Pose(4, 0),
        measurement_radius=3.5,
    )
    m.store.add(p)
    if not outside_range:
        loc.states["scientist"] = replace(loc.get("scientist"), pose=Pose(1, 2))
    nav["scientist"].status = "succeeded"
    m.tick(0)
    loc.advance(0.2)
    m.tick(0.2)
    assert p.result is None
    assert m.executions["scientist"].phase == "navigate" and not p.attempts
    if outside_range:
        assert any(e["event"] == "science_position_correction" for e in events)


def test_out_of_range_arrival_is_corrected_without_failed_attempts_then_measured():
    m, loc, _, _, nav, events = fixture_mission()
    p = POI(
        "rock",
        Kind.ROCK_MEASUREMENT,
        Pose(4, 2),
        Pose(4, 0),
        measurement_approach=Pose(4, 0),
        measurement_radius=3.5,
    )
    m.store.add(p)
    nav["scientist"].status = "succeeded"
    m.tick(0)
    loc.advance(0.2)
    m.tick(0.2)
    assert m.executions["scientist"].phase == "navigate" and not p.attempts
    assert p.result is None
    correction = next(e for e in events if e["event"] == "science_position_correction")
    assert correction["target_distance"] > correction["maximum_distance"]
    loc.states["scientist"] = replace(loc.get("scientist"), pose=Pose(4, 0))
    for now in (0.4, 0.6, 2.8):
        loc.advance(now)
        m.tick(now)
    assert p.status == Status.SUCCEEDED
    assert len(p.attempts) == 1 and p.attempts[0]["success"]
    assert p.result["target_distance"] <= p.measurement_radius


def test_help_details_report_last_failure_reason():
    m, _, _, _, _, _ = fixture_mission()
    p = POI(
        "rock", Kind.ROCK_MEASUREMENT, Pose(0, 0), Pose(2, 0), status=Status.NEEDS_HELP
    )
    p.attempts = [
        dict(
            success=False,
            cancelled=False,
            reason="outside_measurement_range",
            stamp=169.6,
        )
    ]
    assert "outside_measurement_range" in m.poi_wait_reason(p)


def test_one_scientists_station_failure_does_not_block_another_scientist():
    m, loc, health, _, nav, events = fixture_mission()
    other = replace(m.robots["scientist"], name="scientist_2")
    m.robots[other.name] = other
    loc.states[other.name] = loc.get("scientist")
    health.states[other.name] = health.get("scientist")
    nav[other.name] = type(nav["scientist"])()
    m.modes[other.name] = "auto"

    class PerRobotStation:
        def plan(self, robot, poi, now):
            if robot.name == "scientist":
                raise ValueError("No path from this robot")
            return Pose(4, 0)

    m.measurement_stations = PerRobotStation()
    p = POI(
        "rock", Kind.ROCK_MEASUREMENT, Pose(4, 2), Pose(4, 0), measurement_radius=3.5
    )
    m.store.add(p)
    m.tick(0)
    assert p.claimed_by == other.name and p.status in {Status.CLAIMED, Status.RUNNING}
    assert not any(e["event"] == "measurement_station_unreachable" for e in events)


def test_science_approach_avoids_peer_and_uses_nearby_in_range_station():
    from space_robotics_nav.mosaic.scene import MeasurementStations

    m, loc, _, _, _, _ = fixture_mission()
    loc.states["scout"] = replace(loc.get("scout"), pose=Pose(1, 0))
    loc.states["scientist"] = replace(loc.get("scientist"), pose=Pose(-2.75, 0))
    m.maps.block([-0.15, -0.15, 0.15, 0.15])
    p = POI(
        "rock", Kind.ROCK_MEASUREMENT, Pose(0, 0), Pose(0, 0), measurement_radius=3.5
    )
    planner = MeasurementStations(
        m.maps, loc, m.planner, clearance=0.2, robots=m.robots
    )
    station = planner.plan(m.robots["scientist"], p, 0)
    assert (
        station.distance(p.target)
        <= 3.5 - m.robots["scientist"].position_tolerance - 0.05
    )
    assert station.distance(loc.get("scout").pose) > 1.5
    assert station.distance(loc.get("scientist").pose) < 0.3


def test_blocked_science_changes_station_without_releasing_claim_or_failing_attempt():
    m, loc, _, _, nav, events = fixture_mission()
    p = POI(
        "rock",
        Kind.ROCK_MEASUREMENT,
        Pose(4, 2),
        Pose(4, 0),
        measurement_approach=Pose(4, 0),
        measurement_radius=3.5,
    )
    m.store.add(p)
    m.tick(0)
    execution = m.executions["scientist"]
    token = execution.token

    class AlternateStation:
        def plan(self, robot, poi, now):
            return Pose(2, 2)

    m.measurement_stations = AlternateStation()
    nav["scientist"].blocked_since = 0
    loc.advance(3.2)
    m.tick(3.2)
    assert m.executions["scientist"] is execution and execution.token == token
    assert p.approach == Pose(2, 2) and nav["scientist"].goals[-1] == p.approach
    assert not p.attempts and p.claimed_by == "scientist"
    assert any(e["event"] == "science_station_replanned" for e in events)


def test_winning_scientist_uses_its_own_station_proposal():
    m, loc, health, _, nav, _ = fixture_mission()
    other = replace(
        m.robots["scientist"],
        name="scientist_2",
        home=Pose(5, 4),
        preferences={Kind.ROCK_MEASUREMENT: 200},
    )
    m.robots[other.name] = other
    loc.states[other.name] = replace(loc.get("scientist"), pose=other.home)
    health.states[other.name] = health.get("scientist")
    nav[other.name] = type(nav["scientist"])()
    m.modes[other.name] = "auto"

    class PerRobotStation:
        def plan(self, robot, poi, now):
            return Pose(3 if robot.name == other.name else -3, 0)

    m.measurement_stations = PerRobotStation()
    p = POI(
        "rock", Kind.ROCK_MEASUREMENT, Pose(0, 0), Pose(0, 0), measurement_radius=3.5
    )
    m.store.add(p)
    m.tick(0)
    assert p.claimed_by == other.name
    assert p.approach == Pose(3, 0) and nav[other.name].goals[-1] == p.approach


def test_omnidirectional_simulated_instrument_does_not_require_body_alignment():
    from space_robotics_nav.mosaic.providers import SimulatedInstrument

    m, _, _, _, nav, _ = fixture_mission()
    m.instrument = SimulatedInstrument(requires_heading=False)
    p = POI(
        "rock",
        Kind.ROCK_MEASUREMENT,
        Pose(4, 2),
        Pose(4, 0),
        measurement_approach=Pose(4, 0),
        measurement_radius=3.5,
    )
    m.store.add(p)
    m.tick(0)
    assert not nav["scientist"].options["require_heading"]
