"""Original-world preservation and mission visualization contracts."""

import json
from pathlib import Path
import copy
import os
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from space_robotics_nav.config import ROOT
from space_robotics_nav.mosaic.backend import runtime_world
from space_robotics_nav.mosaic.maps import traversable
from space_robotics_nav.mosaic.run import create_mission, load_config
from space_robotics_nav.mosaic.scene import ProjectMap
from space_robotics_nav.mosaic.visualization import MissionView
from space_robotics_nav.mosaic.vehicles import configured_vehicle
from space_robotics_nav.mosaic.providers import GazeboGroundTruth
from space_robotics_nav.mosaic.contracts import Kind, Pose
from space_robotics_nav.mosaic.poi import POI


@pytest.fixture(scope="module")
def lunar_map():
    if not (ROOT / "assets/cache/lunar_surface0/model.sdf").exists():
        pytest.skip("Run project asset preparation first")
    config = load_config(ROOT / "configs/mosaic/demo.yaml")
    return config, ProjectMap(config)


def canonical(element):
    return ET.canonicalize(ET.tostring(element, encoding="unicode"), strip_text=True)


def test_runtime_keeps_original_lunar_geometry_and_physics(lunar_map, tmp_path):
    config, maps = lunar_map
    config = {**config, "scene_objects_static": False}
    runtime = ET.parse(runtime_world(config, tmp_path, maps)).getroot().find("world")
    original = ET.parse(ROOT / "worlds/moon.sdf").getroot().find("world")
    for key in ("physics", "gravity", "scene", "light"):
        assert [canonical(e) for e in runtime.findall(key)] == [
            canonical(e) for e in original.findall(key)
        ]
    objects = [m for m in maps.spec["models"] if "controller" not in m]
    actual = runtime.findall("include")
    assert len(actual) == len(objects) + 2
    assert runtime.find("model") is None, "No substitute plane or box rocks"
    for include, item in zip(actual, objects):
        assert include.findtext("uri") == item["path"]
        assert include.findtext("pose") == item["pose"]
    for robot, include in zip(config["robots"], actual[-2:]):
        pose = np.fromstring(include.findtext("pose"), sep=" ")
        spec = configured_vehicle(robot, config)
        assert pose[2] == pytest.approx(
            maps.height(*robot["home"][:2]) + spec.spawn_height
        )
        native = ET.parse(spec.path).getroot().find("model")
        adapted = ET.parse(include.findtext("uri")).getroot().find("model")
        assert [canonical(j) for j in native.findall("joint")] == [
            canonical(j) for j in adapted.findall("joint")
        ]
        for before, after in zip(native.findall("link"), adapted.findall("link")):
            if before.find("inertial") is not None:
                assert canonical(before.find("inertial")) == canonical(
                    after.find("inertial")
                )
            for a, b in zip(
                before.findall("collision/geometry"),
                after.findall("collision/geometry"),
            ):
                original_geometry = copy.deepcopy(a)
                for uri in original_geometry.iter("uri"):
                    if not Path(uri.text).is_absolute():
                        uri.text = str((spec.path.parent / uri.text).resolve())
                assert canonical(original_geometry) == canonical(b)
        assert all(e.text == "0.0" for e in adapted.findall(".//friction/ode/slip1"))
        controller = include.find("plugin[@name='gz::sim::systems::DiffDrive']")
        assert controller.findtext("topic") == f"/model/{robot['name']}/cmd_vel"
        controller.find("topic").text = "/cmd_vel"
        if spec.wheel_separation is not None:
            assert float(controller.findtext("wheel_separation")) == pytest.approx(
                0.726
            )
            controller.find("wheel_separation").text = ET.fromstring(
                spec.controller
            ).findtext("wheel_separation")
        assert canonical(controller) == canonical(ET.fromstring(spec.controller))
        assert robot["radius"] >= spec.minimum_radius
    assert runtime.find("gui/plugin[@filename='WorldControl']") is None


def test_navigation_grid_uses_project_heights_slopes_and_original_rocks(lunar_map):
    config, maps = lunar_map
    grid = maps.snapshot(0)
    assert np.ptp(maps.heights) > 2
    assert grid.blocked.any() and (~grid.blocked).any()
    assert "moon" in grid.source
    for robot in config["robots"]:
        free = traversable(grid, robot["radius"])
        for position in [
            robot["home"],
            *[p["pose"] for p in config["pois"] if p["kind"] in robot["capabilities"]],
        ]:
            from space_robotics_nav.mosaic.contracts import Pose

            x, y = grid.cell(Pose(*position))
            assert free[y, x], position
    assert "lunar_rock0" in maps.objects
    assert maps.target_position("lunar_rock0") == pytest.approx([0.05, -3])


def test_perseverance_navigation_frame_matches_drive_direction_on_sloped_spawn(
    lunar_map,
):
    from scipy.spatial.transform import Rotation

    config, maps = lunar_map
    robot = {**config["robots"][1], "model": "nasa_perseverance"}
    spec = configured_vehicle(robot, config)
    pose = spec.spawn(maps, robot["home"])
    navigation = Rotation.from_euler("xyz", maps.spawn(*robot["home"])[3:])
    model = Rotation.from_euler("xyz", pose[3:])
    assert model.apply([0, -1, 0]) == pytest.approx(navigation.apply([1, 0, 0]))
    loc = GazeboGroundTruth({robot["name"]: spec.heading_offset})
    loc.ingest(
        dict(
            sim_time=4,
            robots={
                robot["name"]: dict(
                    position=pose[:3],
                    rpy=pose[3:],
                    linear_velocity=[0.2, 0, 0],
                    angular_velocity=[0, 0, 0.1],
                )
            },
        )
    )
    state = loc.get(robot["name"])
    assert state.pose.yaw == pytest.approx(0)
    assert state.linear_speed == pytest.approx(0.2)
    assert state.yaw_rate == pytest.approx(0.1)


def test_science_vehicle_cannot_use_an_undersized_footprint(lunar_map, tmp_path):
    import copy
    import yaml

    config = copy.deepcopy(lunar_map[0])
    config["robots"][1]["radius"] = 0.4
    path = tmp_path / "undersized_scientist.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="native collision radius"):
        load_config(path)


def test_recorded_moon_layout_replans_edge_station_with_arrival_range_margin(lunar_map):
    case = json.loads(
        (Path(__file__).parent / "fixtures/science_arrival_range.json").read_text()
    )
    config = copy.deepcopy(lunar_map[0])
    config["targets"], config["pois"] = [], []
    config["exploration"]["enabled"] = False
    mission, loc, _ = create_mission(config, case["state"])
    robot = mission.robots["explorer_r2_science"]
    target, old_station, old_stop = (
        Pose(**case[key]) for key in ("target", "old_station", "old_stop")
    )
    limit = case["maximum_distance"]
    # The recorded navigator legitimately accepted the station tolerance,
    # while the recorded physical stop was outside the instrument range.
    assert old_station.distance(target) < limit < old_stop.distance(target)
    assert old_stop.distance(old_station) < robot.position_tolerance
    poi = POI(
        "recorded_rock",
        Kind.ROCK_MEASUREMENT,
        target,
        old_station,
        measurement_approach=old_station,
        measurement_radius=limit,
    )
    mission.store.add(poi)
    mission.tick(case["state"]["sim_time"])
    mission.sequence_planner.future.result(timeout=10)
    mission.tick(case["state"]["sim_time"])
    assert poi.claimed_by == robot.name
    assert poi.measurement_approach != old_station
    assert poi.approach.distance(target) + robot.position_tolerance + 0.05 <= limit
    assert mission.navigators[robot.name].path


def test_second_recorded_measurement_can_start_outside_preferred_tracking_band(
    lunar_map,
):
    case = json.loads(
        (Path(__file__).parent / "fixtures/science_departure_margin.json").read_text()
    )
    config = copy.deepcopy(lunar_map[0])
    # Retain the original strict obstacle band for this historical departure
    # regression; normal scenes now allow rolling over these small rocks.
    config["map"]["max_step_height"] = 0
    config["targets"], config["pois"] = [], []
    config["exploration"]["enabled"] = False
    m, loc, _ = create_mission(config, case["state"])
    robot = m.robots["explorer_r2_science"]
    start = loc.get(robot.name).pose
    grid = m.maps.snapshot(0)
    x, y = grid.cell(start)
    assert traversable(grid, robot.radius)[y, x]
    assert not traversable(grid, robot.radius + config["measurement_clearance"])[y, x]
    data = case["poi"]
    p = POI(
        data["id"],
        Kind.ROCK_MEASUREMENT,
        Pose(**data["target"]),
        start,
        measurement_radius=data["measurement_radius"],
    )
    m.store.add(p)
    now = case["state"]["sim_time"]
    m.tick(now)
    m.sequence_planner.future.result(timeout=10)
    m.tick(now)
    assert p.claimed_by == robot.name and p.blocked_reason == ""
    assert p.measurement_approach.distance(p.target) <= 3.0


def view_fixture(tmp_path):
    config = load_config(ROOT / "configs/mosaic/flat_regression.yaml")
    raw = dict(
        sim_time=4,
        robots={
            r["name"]: dict(
                position=[*r["home"][:2], 0.4],
                rpy=[0, 0, r["home"][2]],
                linear_velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                contacts=[],
            )
            for r in config["robots"]
        },
    )
    mission, loc, health = create_mission(config, raw)
    mission.tick(4)
    view = MissionView(mission, tmp_path, config)
    state = view.snapshot([dict(event="test_ready", stamp=4)])
    return mission, view, state


def test_view_publishes_real_paths_states_and_stable_nonphysical_markers(tmp_path):
    mission, view, state = view_fixture(tmp_path)
    assert (tmp_path / "map.png").is_file()
    assert json.loads((tmp_path / "mission_state.json").read_text()) == json.loads(
        json.dumps(state)
    )
    assert state["robots"][0]["path"]
    assert all(r["phase"] == "navigate" for r in state["robots"])
    assert "truth_for_evaluation" not in state
    markers = view.markers(state)
    assert any(m.get("type") == "line" for m in markers)
    assert [m["id"] for m in markers] == [m["id"] for m in view.markers(state)]
    mission.command(dict(op="remove_poi", poi=state["pois"][0]["id"]), 4)
    assert mission.store.items[state["pois"][0]["id"]].status == "rejected"
    assert len(mission.executions) == 1


@pytest.mark.skipif(
    os.environ.get("RUN_MOSAIC_GUI_TESTS") != "1", reason="Requires desktop Tk/X11"
)
def test_control_panel_renders_and_sends_operator_commands(tmp_path):
    import tkinter as tk
    from space_robotics_nav.mosaic.control import ControlPanel

    mission, view, state = view_fixture(tmp_path)
    root = tk.Tk()
    try:
        panel = ControlPanel(root, tmp_path)
        root.update()
        panel.poll()
        root.update()
        assert len(panel.table.get_children()) == 2
        assert panel.canvas.find_all()
        panel.robot_command("pause")
        command = json.loads(
            (tmp_path / "control_commands.jsonl").read_text().splitlines()[-1]
        )
        assert command == dict(op="pause", robot="lunar_hauler")
        robot = panel.state["robots"][1]
        x, y = panel.screen(robot["pose"]["x"], robot["pose"]["y"])
        panel.map_click(SimpleNamespace(x=x, y=y))
        assert panel.robot_name.get() == "lunar_excavator"
    finally:
        root.destroy()
