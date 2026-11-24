"""Opt-in multi-robot physics checks on the real, shared Gazebo world."""

import os
import json

import numpy as np
import pytest

from space_robotics_nav.config import ROOT
from space_robotics_nav.mosaic.backend import MultiRobotBackend
from space_robotics_nav.mosaic.run import load_config
from space_robotics_nav.mosaic.run import main
from space_robotics_nav.mosaic.run import create_mission
from space_robotics_nav.mosaic.providers import GazeboHealth
from space_robotics_nav.mosaic.scene import known_map
import yaml

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GAZEBO_TESTS") != "1",
    reason="Requires the Gazebo image rebuilt with mosaic-worker",
)


def test_r2_rolls_over_low_rock_without_a_collision_failure():
    config = load_config(ROOT / "configs/mosaic/demo.yaml")
    config.update(world="mining_vehicles", targets=[], exploration={"enabled": False})
    config["providers"]["global_map"] = "known_layout"
    config["map"] = dict(bounds=[-8, -6, 8, 6], resolution=0.1)
    config["robots"] = config["robots"][:1]
    config["robots"][0]["home"] = [-2, 0, 0]
    config["obstacles"] = [
        dict(id="low_rock", bounds=[-0.12, -1.5, 0.12, 1.5], height=0.1)
    ]
    config["pois"] = [dict(id="cross_low_rock", kind="move", pose=[2, 0, 0])]
    backend = MultiRobotBackend(config)
    try:
        # A synthetic flat fixture isolates the mapping/health boundary; the
        # box remains a real collision in Gazebo throughout the crossing.
        backend.maps.block(config["obstacles"][0]["bounds"], False)
        backend.maps.traversable_objects = {"low_rock"}
        mission, loc, health = create_mission(config, backend.state, maps=backend.maps)
        touched = False
        for _ in range(150):
            state = backend.state
            loc.ingest(state)
            health.ingest(state)
            touched |= "low_rock" in state["robots"]["explorer_r2"]["contacts"]
            assert not state["robots"]["explorer_r2"]["body_contacts"]
            assert not health.get("explorer_r2").collision
            if mission.store.items["cross_low_rock"].status.value == "succeeded":
                break
            commands = mission.tick(state["sim_time"])
            backend.step(commands)
        assert touched
        assert mission.store.items["cross_low_rock"].status.value == "succeeded"
        assert backend.state["robots"]["explorer_r2"]["position"][0] > 1.5
    finally:
        backend.close()


def test_shared_clock_isolated_controls_and_no_mission_reset():
    config = load_config(ROOT / "configs/mosaic/flat_regression.yaml")
    backend = MultiRobotBackend(config)
    try:
        initial = backend.state
        actions = {"lunar_hauler": [0.2, 0], "lunar_excavator": [0, 0]}
        state = initial
        for _ in range(20):
            state = backend.step(actions)
        assert state["iterations"] - initial["iterations"] == 20 * 40
        assert state["sim_time"] - initial["sim_time"] == pytest.approx(4)
        a, b = state["robots"], initial["robots"]
        assert a["lunar_hauler"]["position"][0] - b["lunar_hauler"]["position"][0] > 0.4
        assert (
            np.linalg.norm(
                np.array(a["lunar_excavator"]["position"])
                - b["lunar_excavator"]["position"]
            )
            < 0.02
        )
        assert all(not r["contacts"] for r in a.values())
        with pytest.raises(RuntimeError, match="no episode reset"):
            backend.reset()
        with pytest.raises(ValueError, match="one action per robot"):
            backend.step({"lunar_hauler": [0, 0]})
    finally:
        backend.close()


def test_physical_obstacle_contacts_are_reported():
    config = load_config(ROOT / "configs/mosaic/flat_regression.yaml")
    config["obstacles"].append(
        dict(id="contact_wall", bounds=[-4, 3, -3.7, 5], height=1)
    )
    backend = MultiRobotBackend(config)
    try:
        contacts = []
        for _ in range(60):
            state = backend.step({"lunar_hauler": [0.25, 0], "lunar_excavator": [0, 0]})
            contacts = state["robots"]["lunar_hauler"]["contacts"]
            if "contact_wall" in contacts:
                break
        assert "contact_wall" in contacts
    finally:
        backend.close()


def test_complete_exploration_inspection_and_measurement(tmp_path):
    output = tmp_path / "mission"
    assert (
        main(
            [
                "--config",
                str(ROOT / "configs/mosaic/flat_regression.yaml"),
                "--auto-approve",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    assert summary["exit_reason"] == "completed" and not summary["active"]
    assert all(p["status"] == "succeeded" for p in summary["pois"])
    rock = next(p for p in summary["pois"] if p["object_id"] == "sample_rock")
    assert rock["result"]["simulated"] and rock["result"]["robot"] == "lunar_excavator"
    assert [a["kind"] for a in rock["attempts"]] == [
        "rock_candidate",
        "rock_measurement",
    ]
    for line in (output / "telemetry.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert all(
            not s["contacts"] for s in record["truth_for_evaluation"]["robots"].values()
        )


def test_failure_reassignment_teleop_and_localization_loss(tmp_path):
    config = load_config(ROOT / "configs/mosaic/flat_regression.yaml")
    config["obstacles"], config["targets"] = [], []
    config["pois"] = [dict(id="shared_move", kind="move", pose=[0, -4, 0])]
    config_path, script = tmp_path / "scenario.yaml", tmp_path / "commands.yaml"
    config_path.write_text(yaml.safe_dump(config))
    script.write_text(
        yaml.safe_dump(
            [
                dict(at=1, op="disable", robot="lunar_excavator"),
                dict(
                    at=12,
                    op="teleop",
                    robot="lunar_hauler",
                    velocity=[0.1, 0],
                    seconds=1,
                ),
                dict(at=15, op="resume", robot="lunar_hauler"),
                dict(at=25, op="localization_valid", robot="lunar_hauler", valid=False),
                dict(at=29, op="localization_valid", robot="lunar_hauler", valid=True),
            ]
        )
    )
    output = tmp_path / "faults"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--script",
                str(script),
                "--seconds",
                "180",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    summary = json.loads((output / "summary.json").read_text())
    task = summary["pois"][0]
    assert task["status"] == "succeeded"
    assert task["attempts"][0]["robot"] == "lunar_excavator"
    assert task["attempts"][-1]["robot"] == "lunar_hauler"
    assert any(a["reason"] == "teleop" for a in task["attempts"])
    assert any(a["reason"] == "localization_unavailable" for a in task["attempts"])
    for line in (output / "telemetry.jsonl").read_text().splitlines():
        row = json.loads(line)
        if not row["poses"]["lunar_hauler"]["valid"]:
            assert row["velocities"]["lunar_hauler"] == [0, 0]
        if row["stamp"] >= 5:
            assert row["velocities"]["lunar_excavator"] == [0, 0]


def test_original_moon_shared_clock_height_and_visualization_are_physical():
    config = load_config(ROOT / "configs/mosaic/demo.yaml")
    backend = MultiRobotBackend(config)
    try:
        initial = backend.state
        assert backend.physics_dt == pytest.approx(0.025)
        assert backend.decimation == 8
        state = backend.step(
            {name: [0, 0] for name in backend.descriptors},
            markers=[dict(id=1, color=[0, 1, 0, 1], position=[8, -10, 1], scale=0.3)],
        )
        assert state["iterations"] - initial["iterations"] == 8
        assert state["sim_time"] - initial["sim_time"] == pytest.approx(0.2)
        for name, robot in state["robots"].items():
            assert not robot["contacts"]
            assert max(abs(v) for v in robot["rpy"][:2]) < 0.35
            assert (
                abs(
                    robot["position"][2]
                    - backend.maps.height(*robot["position"][:2])
                    - 0.4
                )
                < 0.2
            )
        assert (
            "lunar_rock0" in state["objects"] and "lunar_surface0" in state["objects"]
        )
        # Both source controllers must move forward in the common navigation
        # frame despite their different model axes, with independent topics.
        from space_robotics_nav.mosaic.run import create_mission

        mission, loc, health = create_mission(config, state, maps=backend.maps)
        for moving, stationary in [
            ("explorer_r2", "explorer_r2_science"),
            ("explorer_r2_science", "explorer_r2"),
        ]:
            start = loc.get(moving).pose
            commands = {moving: [0.15, 0], stationary: [0, 0]}
            for _ in range(15):
                state = backend.step(commands)
            loc.ingest(state)
            end = loc.get(moving).pose
            displacement = np.array([end.x - start.x, end.y - start.y])
            assert (
                displacement @ np.array([np.cos(start.yaw), np.sin(start.yaw)]) > 0.15
            )
            assert all(not r["contacts"] for r in state["robots"].values())
            assert abs(loc.get(moving).linear_speed) > 0.03
    finally:
        backend.close()


def test_default_continues_exploring_and_does_not_measure_from_a_distant_station(
    tmp_path,
):
    output = tmp_path / "moon"
    assert main(["--seconds", "320", "--output", str(output)]) == 2
    summary = json.loads((output / "summary.json").read_text())
    assert summary["exit_reason"] == "budget_exhausted"
    assert summary["exploration"]["enabled"]
    assert summary["exploration"]["generated"] >= 2
    rocks = [p for p in summary["pois"] if p["object_id"]]
    assert len(rocks) == 2
    for rock in rocks:
        assert rock["kind"] == "rock_measurement"
        assert rock["status"] == "succeeded"
        assert rock["measurement_radius"] == 3.5
        assert rock["measurement_approach"] is not None
        assert rock["result"]["robot"] == "explorer_r2_science"
        assert rock["result"]["target_distance"] <= 3.5
        assert rock["result"]["station_error"] <= 0.45
        assert rock["confirmation_result"]["confirmed_measurement"]
        assert [a["robot"] for a in rock["attempts"]] == [
            "explorer_r2",
            "explorer_r2_science",
        ]
    events = [json.loads(l) for l in (output / "events.jsonl").read_text().splitlines()]
    assert not any(e["event"] == "all_tasks_completed" for e in events)
    assert not any(e["event"] == "science_waiting_for_clearance" for e in events)
    assert any(
        e["event"] == "action_started" and e.get("kind") == "rock_measurement"
        for e in events
    )
    analyses = [e for e in events if e["event"] == "analysis_finished"]
    assert len(analyses) >= 3 and all(e["seconds"] >= 10 for e in analyses)
    assert len(analyses[0]["candidates"]) == 2
    empty = next(
        e
        for e in analyses
        if not e["candidates"] and e["stamp"] < summary["sim_time"] - 3
    )
    later_claims = [
        e
        for e in events
        if e["event"] == "task_claimed"
        and e["robot"] == "explorer_r2"
        and e["stamp"] > empty["stamp"]
    ]
    assert later_claims and later_claims[0]["kind"] == "explore"
    assert any(
        e["event"] == "task_claimed"
        and e["robot"] == "explorer_r2"
        and e["stamp"] > 200
        for e in events
    )
    generated = {e["poi"] for e in events if e["event"] == "exploration_poi_generated"}
    assert any(
        p["id"] in generated and p["status"] == "succeeded" for p in summary["pois"]
    )
    assert all(a["success"] for p in summary["pois"] for a in p["attempts"])
    view = json.loads((output / "mission_state.json").read_text())
    assert view["reason"] != "completed" and view["exploration"]["enabled"]
    for p in view["pois"]:
        if p["object_id"]:
            assert p["navigation_station"] is not None
            assert p["result"]["target_distance"] <= p["measurement_radius"]
    records = [
        json.loads(l) for l in (output / "telemetry.jsonl").read_text().splitlines()
    ]
    first_science = next(
        e
        for e in events
        if e["event"] == "task_claimed" and e.get("kind") == "rock_measurement"
    )
    assert any(
        first_science["stamp"] <= row["stamp"] <= first_science["stamp"] + 0.4
        and sum(abs(v) for v in row["velocities"]["explorer_r2_science"]) > 0.01
        for row in records
    )
    assert any(
        r["stamp"] > 200 and r["velocities"]["explorer_r2"][0] > 0.05 for r in records
    )
    for row in records:
        assert "wait_clearance" not in row["phases"].values()
        assert all(v[0] >= 0 for v in row["velocities"].values())
        if row["phases"].get("explorer_r2") == "analyze":
            assert row["velocities"]["explorer_r2"] == [0, 0]
        assert all(
            not r["contacts"] for r in row["truth_for_evaluation"]["robots"].values()
        )
    for analysis in analyses:
        started = next(
            e["stamp"]
            for e in events
            if e["event"] == "analysis_started" and e["poi"] == analysis["poi"]
        )
        window = [
            r["poses"]["explorer_r2"]["pose"]
            for r in records
            if started <= r["stamp"] <= analysis["stamp"]
        ]
        points = np.array([[p["x"], p["y"]] for p in window])
        assert np.linalg.norm(points - points[0], axis=1).max() < 0.05
    scene = json.loads((output / "scene.json").read_text())
    assert scene["world"] == "moon" and scene["physics_dt"] == 0.025
    assert [v["model"] for v in scene["vehicles"]] == [
        "explorer_r2",
        "explorer_r2",
    ]


def test_original_procgen_four_vehicle_fleet_starts_with_exploration_then_measures(
    tmp_path,
):
    # Baking the original 8K material is a separate, expensive asset build.
    # Validate the actual generated 80m scene, rather than replacing it with
    # a cheap mesh or packing four robots into the old narrow 40m terrain.
    config = ROOT / "assets/generated/mosaic_lunar80_seed17_2s2m/mission.yaml"
    if not config.is_file():
        pytest.skip("Generate the original 80m / 2+2 scene first; see docs/MOSAIC.md")
    output = tmp_path / "fleet"
    assert (
        main(["--config", str(config), "--seconds", "600", "--output", str(output)])
        == 2
    )
    summary = json.loads((output / "summary.json").read_text())
    assert len(summary["modes"]) == 4
    assert summary["sim_time"] >= 600
    assert summary["exit_reason"] == "budget_exhausted"
    events = [
        json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()
    ]
    claims = [e for e in events if e["event"] == "task_claimed"]
    assert len({e["robot"] for e in claims if e["kind"] == "explore"}) == 2
    assert all(e["kind"] == "explore" for e in claims if e["stamp"] == 4)
    science = [
        p
        for p in summary["pois"]
        if p["kind"] == "rock_measurement" and p["status"] == "succeeded"
    ]
    assert science
    assert {p["result"]["robot"] for p in science} == {"scientist_01", "scientist_02"}
    for p in science:
        assert p["confirmation_result"]["confirmed_measurement"]
        assert p["attempts"][0]["kind"] == "rock_candidate"
        assert p["result"]["target_distance"] <= p["measurement_radius"]
    records = [
        json.loads(line)
        for line in (output / "telemetry.jsonl").read_text().splitlines()
    ]
    maps = known_map(load_config(config))
    health = GazeboHealth(maps)
    for row in records:
        truth = row["truth_for_evaluation"]
        assert len(truth["robots"]) == 4
        maps.ingest_objects(truth.get("objects", {}))
        health.ingest(truth)
        # Wheel contact with a traversable low rock is expected; body contact,
        # contact with a teammate or a non-traversable rock remains a failure.
        assert all(not health.get(name).collision for name in truth["robots"])
    first_confirmation = min(
        e["stamp"] for e in events if e["event"] == "candidate_confirmed"
    )
    assert all(
        row["velocities"][name] == [0, 0]
        for row in records
        if row["stamp"] < first_confirmation
        for name in summary["modes"]
        if name.startswith("scientist_")
    )


def test_recorded_four_vehicle_padding_deadlock_can_drive_out():
    from space_robotics_nav.mosaic.maps import traversable
    from space_robotics_nav.mosaic.contracts import Pose

    source = ROOT / "assets/generated/mosaic_lunar80_seed17_2s2m/mission.yaml"
    if not source.exists():
        pytest.skip("Requires the recorded original 80m scene")
    config = load_config(source)
    raw = json.loads(
        (ROOT / "tests/mosaic/fixtures/navigation_margin_deadlock.json").read_text()
    )["state"]
    config.update(targets=[], pois=[], exploration={"enabled": False})
    reference, localization, _ = create_mission(config, raw)
    for item in config["robots"]:
        robot = reference.robots[item["name"]]
        start = localization.get(robot.name).pose
        item["home"] = [start.x, start.y, start.yaw]
        grid = reference.local_maps.snapshot(robot, start, raw["sim_time"])
        yy, xx = np.nonzero(traversable(grid, robot.radius))
        xy = np.c_[
            grid.origin[0] + xx * grid.resolution, grid.origin[1] + yy * grid.resolution
        ]
        distances = np.linalg.norm(xy - [start.x, start.y], axis=1)
        goal = None
        for index in np.argsort(distances):
            if not 2 <= distances[index] <= 3:
                continue
            candidate = Pose(*xy[index], start.yaw)
            if reference.planner.plan(start, candidate, grid, robot.radius):
                goal = candidate
                break
        assert goal is not None
        config["pois"].append(
            dict(
                id=f"escape_{robot.name}",
                kind="move",
                pose=[goal.x, goal.y, goal.yaw],
                assigned_to=robot.name,
            )
        )
    backend = MultiRobotBackend(config)
    try:
        mission, loc, health = create_mission(config, backend.state, maps=backend.maps)
        origins = {
            n: np.array(r["position"][:2]) for n, r in backend.state["robots"].items()
        }
        moved = {n: 0.0 for n in origins}
        for _ in range(300):
            state = backend.state
            loc.ingest(state)
            health.ingest(state)
            for name, r in state["robots"].items():
                assert not health.get(name).collision
                moved[name] = max(
                    moved[name],
                    float(np.linalg.norm(np.array(r["position"][:2]) - origins[name])),
                )
            if min(moved.values()) >= 0.3:
                break
            backend.step(mission.tick(state["sim_time"]))
        assert min(moved.values()) >= 0.3, moved
    finally:
        backend.close()
