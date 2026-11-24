"""Small rocks remain scientific objects without becoming navigation walls."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np

from space_robotics_nav.mosaic.traversability import surface, pool
from space_robotics_nav.mosaic.scene import ProjectMap
from space_robotics_nav.mosaic.maps import KnownMap, traversable
from space_robotics_nav.mosaic.navigation import AStarPlanner
from space_robotics_nav.mosaic.contracts import Pose
from space_robotics_nav.mosaic.providers import GazeboHealth


def test_rollable_pebbles_are_free_but_large_steps_slopes_and_unknown_remain_blocked():
    axis = np.linspace(-3, 3, 121)
    xx, yy = np.meshgrid(axis, axis)
    low = 0.1 * np.exp(-(xx**2 + yy**2) / 0.1**2)
    _, small = surface(low, 0.05, {})
    assert not small.any()
    _, large = surface(low * 3, 0.05, {})
    assert large[60, 60]
    _, slope = surface(0.5 * xx, 0.05, {})
    assert slope[30:-30, 30:-30].all()
    unknown = low.copy()
    unknown[60, 60] = np.nan
    _, blocked = surface(unknown, 0.05, {})
    assert blocked[60, 60]
    # The same evidence can be reclassified when another vehicle has lower ability.
    _, limited = surface(low, 0.05, {"max_step_height": 0.04})
    assert limited[60, 60]


def test_small_rock_slope_does_not_destroy_a_route_after_footprint_inflation():
    axis = np.linspace(-3, 3, 121)
    xx, yy = np.meshgrid(axis, axis)
    # A row of sharp but low rocks across the travel direction.
    height = sum(
        0.09 * np.exp(-((xx**2 + (yy - y) ** 2) / 0.08**2))
        for y in np.arange(-3, 3.1, 0.3)
    )
    _, blocked = surface(height, 0.05, {})
    maps = KnownMap([-3, -3, 3, 3], 0.25)
    maps.grid = replace(maps.grid, blocked=pool(blocked, 5))
    assert AStarPlanner().plan(Pose(-1.5, 0), Pose(1.5, 0), maps.grid, 0.9)
    _, blocked = surface(height * 4, 0.05, {})
    maps.grid = replace(maps.grid, blocked=pool(blocked, 5))
    assert AStarPlanner().plan(Pose(-1.5, 0), Pose(1.5, 0), maps.grid, 0.9) is None


def test_low_scene_object_stays_discoverable_and_becomes_blocked_if_it_is_lifted():
    maps = ProjectMap.__new__(ProjectMap)
    KnownMap.__init__(maps, [-5, -5, 5, 5], 0.1)
    maps.world = "test"
    maps.terrain_parameters = {"max_step_height": 0.12}
    maps.terrain_blocked = np.zeros_like(maps.grid.blocked)
    maps.manual_blocks = np.zeros_like(maps.grid.blocked)
    maps.heights = np.zeros_like(maps.grid.blocked, dtype=float)
    maps.poses = {"rock": [0, 0, 0, 0, 0, 0]}
    maps.objects = {"rock": {}}
    maps.object_corners = {
        "rock": np.array(
            [[x, y, z] for x in [-0.1, 0.1] for y in [-0.1, 0.1] for z in [0, 0.1]]
        )
    }
    maps._rebuild()
    assert maps.traversable_objects == {"rock"} and not maps.grid.blocked.any()
    assert maps.target_position("rock") == [0, 0]
    maps.ingest_objects({"rock": [0, 0, 0.4, 0, 0, 0]})
    assert not maps.traversable_objects and maps.grid.blocked.any()


def test_only_low_rock_wheel_contacts_are_allowed_body_contacts_and_peers_still_stop():
    health = GazeboHealth(SimpleNamespace(traversable_objects={"small_rock"}))
    item = dict(rpy=[0, 0, 0], contacts=["small_rock"], body_contacts=[])
    health.ingest({"robots": {"r2": item}})
    assert not health.get("r2").collision
    for change in [
        {"body_contacts": ["small_rock"]},
        {"contacts": ["other_robot"]},
        {"contacts": ["big_rock"]},
        {"body_contacts": ["ground"]},
    ]:
        health.ingest({"robots": {"r2": {**item, **change}}})
        assert health.get("r2").collision
    # Old workers cannot distinguish wheels: remain conservative.
    health.ingest(
        {"robots": {"r2": {k: v for k, v in item.items() if k != "body_contacts"}}}
    )
    assert health.get("r2").collision
