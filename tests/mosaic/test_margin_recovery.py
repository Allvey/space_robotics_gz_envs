"""A physically clear stop inside planning padding must be able to leave it."""

import json
from pathlib import Path

import numpy as np
import pytest

from space_robotics_nav.config import ROOT
from space_robotics_nav.mosaic.contracts import Localization, Pose, Robot
from space_robotics_nav.mosaic.maps import (
    KnownMap,
    footprint_clearances,
    segment_safe,
    traversable,
)
from space_robotics_nav.mosaic.navigation import (
    AStarPlanner,
    PathTracker,
    planning_start,
)
from space_robotics_nav.mosaic.run import create_mission, load_config


def test_padding_stop_can_reverse_out_without_crossing_a_wall_or_ignoring_a_real_overlap():
    grid = KnownMap([-5, -5, 5, 5], 0.25, [[-0.25, -5, 0.25, 5]]).grid
    start, goal = Pose(-1.55, 0), Pose(-3, 0)
    robot = Robot("r2", frozenset(), start, radius=0.9, allow_reverse=True)
    state = Localization(start, 0, "test")
    x, y = grid.cell(start)
    assert not traversable(grid, robot.radius)[y, x]
    path = AStarPlanner().plan(start, goal, grid, robot.radius)
    assert path and path[0] == start
    assert all(segment_safe(a, b, grid, robot.radius) for a, b in zip(path, path[1:]))
    command = PathTracker().command(state, path[1], goal, grid, robot)
    assert command[0] < 0 and PathTracker.safe(state, command, grid, robot.radius)
    assert AStarPlanner().plan(start, Pose(3, 0), grid, robot.radius) is None
    touching = Localization(Pose(-1, 0), 0, "test")
    assert not PathTracker.safe(touching, (-0.1, 0), grid, robot.radius)
    assert AStarPlanner().plan(touching.pose, goal, grid, robot.radius) is None


def test_peer_buffer_allows_moving_away_but_rejects_a_predicted_collision():
    grid = KnownMap([-5, -5, 5, 5], 0.25, [[2.4 - 1.2, -1.2, 2.4 + 1.2, 1.2]]).grid
    state = Localization(Pose(0, 0), 0, "test")
    assert not traversable(grid, 0.9)[
        grid.cell(state.pose)[1], grid.cell(state.pose)[0]
    ]
    assert PathTracker.safe(state, (-0.1, 0), grid, 0.9)
    assert not PathTracker.safe(state, (0.25, 0), grid, 0.9)


def test_all_four_recorded_stops_have_safe_connections_in_the_real_local_maps():
    config = ROOT / "assets/generated/mosaic_lunar80_seed17_2s2m/mission.yaml"
    if not config.exists():
        pytest.skip("Requires the recorded original 80m scene")
    raw = json.loads(
        (Path(__file__).parent / "fixtures/navigation_margin_deadlock.json").read_text()
    )["state"]
    mission, loc, _ = create_mission(load_config(config), raw)
    padded_stops = 0
    for name, robot in mission.robots.items():
        state = loc.get(name)
        grid = mission.local_maps.snapshot(robot, state.pose, raw["sim_time"])
        x, y = grid.cell(state.pose)
        padded_stops += not traversable(grid, robot.radius)[y, x]
        assert PathTracker.safe(state, (0, 0), grid, robot.radius)
        anchor = planning_start(state.pose, grid, robot.radius)
        assert anchor is not None
        goal = Pose(
            grid.origin[0] + anchor[0] * grid.resolution,
            grid.origin[1] + anchor[1] * grid.resolution,
        )
        path = AStarPlanner().plan(state.pose, goal, grid, robot.radius)
        assert path
        assert all(
            segment_safe(a, b, grid, robot.radius) for a, b in zip(path, path[1:])
        )
    assert padded_stops >= 3
