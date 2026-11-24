"""Seeded physical maps, discovery-only initialization, and fleet task supply."""

import copy
import json
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from space_robotics_nav.mosaic.generate import generate, GeneratedMap
from space_robotics_nav.mosaic.run import load_config, create_mission
from space_robotics_nav.mosaic.backend import runtime_world
from space_robotics_nav.mosaic.contracts import Kind, Pose
from space_robotics_nav.terrain import collision_mesh
from space_robotics_nav.config import ROOT
from space_robotics_nav.raycast import Raycaster
from space_robotics_nav.mosaic.maps import traversable

TERRAIN = ROOT / "assets/cache/lunar_surface0/model.sdf"


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    path = generate(
        tmp_path_factory.mktemp("random_a"),
        seed=11,
        size=40,
        scouts=2,
        scientists=2,
        rocks=8,
        terrain_model=TERRAIN,
    )
    config = load_config(path)
    return path, config, GeneratedMap(config)


def test_seed_reproduces_height_rocks_and_fleet_and_different_seed_changes_layout(
    generated, tmp_path
):
    path, config, maps = generated
    repeat = load_config(
        generate(
            tmp_path / "same",
            seed=11,
            size=40,
            scouts=2,
            scientists=2,
            rocks=8,
            terrain_model=TERRAIN,
        )
    )
    other = load_config(
        generate(
            tmp_path / "other",
            seed=12,
            size=40,
            scouts=2,
            scientists=2,
            rocks=8,
            terrain_model=TERRAIN,
        )
    )
    repeat_maps, other_maps = GeneratedMap(repeat), GeneratedMap(other)
    assert np.array_equal(maps.heights, repeat_maps.heights)
    assert maps.poses == repeat_maps.poses and config["robots"] == repeat["robots"]
    # Reusing one original terrain preserves its geometry. Layout is reseeded.
    assert np.array_equal(maps.heights, other_maps.heights)
    assert maps.poses != other_maps.poses
    with pytest.raises(ValueError, match="empty"):
        generate(path.parent, seed=11, size=40, scouts=2, scientists=2, rocks=8)


def test_generated_mesh_map_and_runtime_share_geometry_and_unique_controls(
    generated, tmp_path
):
    scenario_path, config, maps = generated
    manifest = json.loads(open(config["scene_manifest"]).read())
    terrain = collision_mesh(manifest["spec"]["models"][0]["path"])
    assert np.ptp(terrain.vertices[:, 2]) > 0.1
    assert terrain.bounds[0, :2] == pytest.approx([-20, -20], abs=0.5)
    assert terrain.bounds[1, :2] == pytest.approx([20, 20], abs=0.5)
    # Navigation samples the full original collision mesh, never a replacement
    # low-resolution surface. Original GLB geometry and PBR bytes are retained.
    original = collision_mesh(TERRAIN)
    assert np.array_equal(terrain.vertices, original.vertices)
    assert np.array_equal(terrain.faces, original.faces)
    copied_mesh = next((scenario_path.parent / "terrain_asset/meshes").glob("*.glb"))
    assert (
        copied_mesh.read_bytes()
        == next((TERRAIN.parent / "meshes").glob("*.glb")).read_bytes()
    )
    xy = maps.coordinates()[100:-100:137]
    assert np.array([maps.height(*p) for p in xy]) == pytest.approx(
        Raycaster(terrain).heights(xy), abs=1e-5
    )
    assert not (scenario_path.parent / "terrain.dae").exists()
    path = runtime_world(config, tmp_path, maps)
    world = ET.parse(path).getroot().find("world")
    assert float(world.findtext("physics/max_step_size")) == 0.025
    includes = world.findall("include")
    assert len(includes) == 1 + 8 + 4
    topics = [
        e.findtext("plugin/topic") for e in includes if e.find("plugin") is not None
    ]
    assert len(set(topics)) == 4
    assert all(
        r["model"] == "explorer_r2" and r["radius"] == 0.9 for r in config["robots"]
    )
    assert all(m["static"] for m in manifest["spec"]["models"])


def test_initial_tasks_are_only_exploration_and_supply_scales_with_scout_count(
    generated,
):
    _, config, maps = generated
    raw = dict(
        sim_time=0,
        objects={},
        robots={
            r["name"]: dict(
                position=maps.spawn(*r["home"])[:3],
                rpy=maps.spawn(*r["home"])[3:],
                linear_velocity=[0, 0, 0],
                angular_velocity=[0, 0, 0],
                contacts=[],
            )
            for r in config["robots"]
        },
    )
    mission, loc, _ = create_mission(config, raw, maps=maps)
    assert all(p.kind == Kind.EXPLORE for p in mission.store.items.values())
    commands = mission.tick(0)
    mission.sequence_planner.future.result(timeout=10)
    commands = mission.tick(0)
    assert len(mission.store.items) == 2 * config["exploration"]["queue_size"]
    assert len(mission.executions) == 2
    assert all(
        mission.robots[name].capabilities.issuperset({Kind.EXPLORE})
        for name in mission.executions
    )
    assert all(
        commands[r["name"]] == (0, 0)
        for r in config["robots"]
        if r["role"] == "Scientist"
    )
    assert all(p.object_id is None for p in mission.store.items.values())
    mission.close()


def test_generated_grid_rejects_configuration_with_different_bounds(generated):
    _, config, _ = generated
    invalid = copy.deepcopy(config)
    invalid["map"]["bounds"] = [-10, -10, 10, 10]
    with pytest.raises(ValueError, match="match"):
        GeneratedMap(invalid)


def test_embedded_collision_rocks_survive_navigation_grid_sampling(generated):
    _, config, maps = generated
    manifest = json.loads(open(config["scene_manifest"]).read())
    assert manifest["collision_sample_resolution"] <= 0.1
    # Native four-vehicle prechecks hit these small protrusions in the original
    # terrain itself, independent of the separately placed science rocks.
    free = traversable(maps.grid, 0.9)
    for position in [(-13.29, 5.94), (-8.11, -11.37)]:
        x, y = maps.grid.cell(Pose(*position))
        assert not free[y, x]


def test_step_limit_reclassifies_cached_heights_without_rebuilding_physical_scene(
    generated,
):
    _, config, maps = generated
    limited = copy.deepcopy(config)
    limited["map"]["max_step_height"] = 0.03
    other = GeneratedMap(limited)
    assert np.array_equal(maps.heights, other.heights)
    assert np.count_nonzero(other.terrain_blocked) > np.count_nonzero(
        maps.terrain_blocked
    )


def test_legacy_binary_mask_is_replaced_using_physical_height_evidence(
    generated, tmp_path
):
    _, config, maps = generated
    old = json.loads(open(config["scene_manifest"]).read())
    old.pop("fine_heights_file")
    fake = tmp_path / "old_blocks.npy"
    np.save(fake, np.ones_like(maps.grid.blocked))
    old["terrain_blocked_file"] = str(fake)
    manifest = tmp_path / "scene_manifest.json"
    manifest.write_text(json.dumps(old))
    updated = GeneratedMap({**config, "scene_manifest": str(manifest)})
    assert not updated.grid.blocked.all()
    assert np.array_equal(updated.terrain_blocked, maps.terrain_blocked)
    assert (tmp_path / "terrain_heights_fine.npy").is_file()
