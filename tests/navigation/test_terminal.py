"""Planner geometry, obstacle rejection and hybrid control contracts."""

from types import SimpleNamespace as NS

import numpy as np
import pytest

from space_robotics_nav.planning.local_map import LocalMap
from space_robotics_nav.planning.paths import connectors, integrate, wrap
from space_robotics_nav.planning.terminal import TerminalConfig, TerminalController


def flat_terrain():
    return NS(
        cfg=NS(slope_limit=0.35),
        rays=NS(heights=lambda xy: np.zeros(len(xy))),
        bounds=np.array([[-100, -100], [100, 100]]),
        objects={},
        poses={},
    )


def test_rs_full_pose_and_reversing_in_all_quadrants():
    rng = np.random.default_rng(578906)
    for _ in range(100):
        start = rng.uniform([-5, -5, -np.pi], [5, 5, np.pi])
        goal = start + rng.uniform([-2, -2, -np.pi], [2, 2, np.pi])
        candidates = connectors(start, goal, family="rs")
        assert candidates
        for path in candidates:
            np.testing.assert_allclose(path.poses[0], start, atol=1e-7)
            error = path.poses[-1] - goal
            error[2] = wrap(error[2])
            assert np.linalg.norm(error) < 1e-6
            assert (
                np.linalg.norm(np.diff(path.poses[:, :2], axis=0), axis=1).max()
                <= 0.050001
            )
            assert all(abs(s.curvature) <= 1 / 0.6 + 1e-9 for s in path.segments)
    reverse = connectors([0, 0, 0], [-1, 0, 0], family="rs")[0]
    assert sum(abs(s.length) for s in reverse.segments) == pytest.approx(1)
    assert any(s.length < 0 for s in reverse.segments)


@pytest.mark.parametrize(
    "goal", [[0, 0, 0], [0, 0, np.pi], [1e-10, 0, -np.pi], [1, 0, 0]]
)
def test_singular_rs_poses(goal):
    paths = connectors([0, 0, 0], goal, family="rs")
    assert paths and all(np.isfinite(p.poses).all() for p in paths)


def test_turn_centre_connector_preserves_original_base_frame_goal():
    start, goal = np.array([0.0, 0.0, -1.0]), np.array([1.0, 0.5, 1.2])
    paths = connectors(start, goal, family="diff-drive", turn_offset=0.32)
    for path in paths:
        np.testing.assert_allclose(path.poses[0], start, atol=1e-9)
        np.testing.assert_allclose(path.poses[-1], goal, atol=1e-9)
        assert np.linalg.norm(path.poses[1, :2] - start[:2]) > 0
        assert np.linalg.norm(np.diff(path.poses[:, :2], axis=0), axis=1).max() < 0.051


def test_footprint_sweep_unknown_terrain_and_dynamic_rock():
    terrain = flat_terrain()
    terrain.objects["rock"] = {"bounds": np.array([[-0.2, -0.2, 0], [0.2, 0.2, 1]])}
    terrain.poses["rock"] = [0, 0, 0, 0, 0, 0]
    local = LocalMap(terrain, radius=0.5, extent=4)
    local.refresh([0, 0])
    assert not local.valid([[0, 0, 0]])
    assert not local.valid(
        [[0.65, 0, 0]]
    )  # Centre clears the rock, footprint does not.
    assert local.valid([[1.2, 0, 0]])
    assert not local.valid([[4.1, 0, 0]])
    straight = connectors([-2, 0, 0], [2, 0, 0], family="diff-drive")[0]
    assert local.valid([straight.poses[0], straight.poses[-1]])
    assert not local.valid(straight.poses)  # Endpoint-only checks would miss the rock.
    terrain.poses["rock"] = [0, 3, 0, 0, 0, 0]
    assert local.objects_changed()
    local.refresh([0, 0])
    assert local.valid(straight.poses)
    terrain.rays.heights = lambda xy: np.full(len(xy), np.nan)
    local.refresh([0, 0])
    assert not local.valid([[0, 0, 0]])


def fake_controller():
    # Use the real planner/controller with a small synthetic footprint so this
    # test checks switching and feedback independently of Gazebo contact physics.
    controller = TerminalController.__new__(TerminalController)
    terrain = flat_terrain()
    terrain.spec = {"heading_offset": 0.0}
    controller.cfg = TerminalConfig()
    controller.env = NS(
        state={
            "position": [0, 0, 0],
            "rpy": [0, 0, 0],
            "linear_velocity": [0, 0, 0],
            "angular_velocity": [0, 0, 0],
        },
        target=np.array([1.5, 0, 0]),
        target_heading=1.0,
        cfg=NS(success_distance=0.18),
        terrain=terrain,
    )
    controller.map = LocalMap(terrain, 0.3)
    controller.vmax, controller.wmax = 0.25, 0.6
    controller.reset()
    return controller


def test_handoff_hysteresis_fail_closed_and_reset():
    controller = fake_controller()
    rl = np.array([0.4, -0.2])
    controller.env.target[0] = 2.1
    np.testing.assert_array_equal(controller.action(rl), rl.astype(np.float32))
    assert not controller.active
    controller.env.target[0] = 1.9
    assert np.isfinite(controller.action(rl)).all() and controller.active
    controller.env.target[0] = 2.2
    controller.action(rl)
    assert controller.active  # Above entry distance, below exit distance.
    controller.env.target[0] = 3.1
    np.testing.assert_array_equal(controller.action(rl), [0, 0])
    assert not controller.active and controller.stats["fallbacks"] == 1
    controller.env.target[0] = 1.5
    controller.reset()
    controller.env.terrain.rays.heights = lambda xy: np.full(len(xy), np.nan)
    np.testing.assert_array_equal(controller.action(rl), rl.astype(np.float32))
    assert not controller.active and controller.stats["planning_failures"] == 1
    np.testing.assert_array_equal(controller.action(rl), rl.astype(np.float32))
    controller.reset()
    assert controller.stats["handoffs"] == 0 and controller.path is None


def test_local_map_physical_slope_and_large_steps():
    terrain = flat_terrain()
    local = LocalMap(terrain, radius=0.3)
    terrain.rays.heights = lambda xy: xy[:, 0] * 0.3
    local.refresh([0, 0])
    assert local.valid([[0, 0, 0]])
    terrain.rays.heights = lambda xy: xy[:, 0] * 1.2
    local.refresh([0, 0])
    assert not local.valid([[0, 0, 0]])
    terrain.rays.heights = lambda xy: np.where(xy[:, 0] > 0, 0.5, 0)
    local.refresh([0, 0])
    assert not local.valid([[0, 0, 0]])


def test_braking_and_new_obstacle_during_tracking():
    controller = fake_controller()
    controller.env.state["linear_velocity"] = [0.35, 0, 0]
    controller.env.state["angular_velocity"] = [0, 0, 0.4]
    np.testing.assert_array_equal(controller.action([0.4, 0.4]), [0, 0])
    assert controller.active and controller.reason == "terminal_braking"
    controller.env.state["linear_velocity"] = [0, 0, 0]
    controller.env.state["angular_velocity"] = [0, 0, 0]
    controller.action([0.4, 0.4])
    assert not controller.braking
    controller.env.terrain.objects["new_rock"] = {
        "bounds": np.array([[-0.1, -0.1, 0], [0.1, 0.1, 1]])
    }
    controller.env.terrain.poses["new_rock"] = [0, 0, 0, 0, 0, 0]
    np.testing.assert_array_equal(controller.action([0.4, 0.4]), [0, 0])
    assert not controller.active and controller.stats["safety_stops"] == 1


@pytest.mark.parametrize("start", [[0, 0, 2.8], [1.6, 0.1, -2.5], [0, 0, -0.5]])
def test_closed_loop_pose_convergence_with_reduced_yaw_response(start):
    controller = fake_controller()
    pose = np.array(start, dtype=float)
    for _ in range(750):
        controller.env.state.update(position=[*pose[:2], 0], rpy=[0, 0, pose[2]])
        action = controller.action([0, 0])
        assert abs(action[0]) <= 0.250001 and abs(action[1]) <= 0.600001
        # Deliberately imperfect yaw response; feedback must correct the error.
        pose = integrate(
            pose,
            action[0] * 0.2,
            action[1] * 0.45 / action[0] if abs(action[0]) > 1e-8 else 0,
            action[1] * 0.45 * 0.2 if abs(action[0]) <= 1e-8 else 0,
        )
        if (
            np.linalg.norm(pose[:2] - controller.env.target[:2]) < 0.18
            and abs(wrap(pose[2] - 1)) < 0.08
        ):
            break
    else:
        pytest.fail(f"Did not converge: {pose}, {controller.report()}")


def test_turn_centre_feedback_converges_without_changing_goal():
    controller = fake_controller()
    controller.map.radius = 0.5
    pose = np.array([0.0, 0.0, -0.5])
    offset, dt = 0.32, 0.2
    for _ in range(750):
        controller.env.state.update(position=[*pose[:2], 0], rpy=[0, 0, pose[2]])
        action = controller.action([0, 0])
        axle = pose.copy()
        axle[:2] -= offset * np.array([np.cos(pose[2]), np.sin(pose[2])])
        v, w = action[0], 0.45 * action[1]
        axle = integrate(
            axle, v * dt, w / v if abs(v) > 1e-8 else 0, w * dt if abs(v) <= 1e-8 else 0
        )
        axle[:2] += offset * np.array([np.cos(axle[2]), np.sin(axle[2])])
        pose = axle
        velocity = v * np.array(
            [np.cos(pose[2]), np.sin(pose[2])]
        ) + offset * w * np.array([-np.sin(pose[2]), np.cos(pose[2])])
        controller.env.state.update(
            linear_velocity=[*velocity, 0], angular_velocity=[0, 0, w]
        )
        if (
            np.linalg.norm(pose[:2] - controller.env.target[:2]) < 0.18
            and abs(wrap(pose[2] - 1)) < 0.1
        ):
            assert controller.turn_offset == pytest.approx(offset, abs=0.08)
            break
    else:
        pytest.fail(f"Did not converge with offset: {pose}, {controller.report()}")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"enter_distance": 3},
        {"linear_speed": 0},
        {"turning_radius": np.nan},
        {"family": "dubins"},
        {"retry_steps": 0},
    ],
)
def test_reject_invalid_terminal_configuration(kwargs):
    with pytest.raises(ValueError):
        TerminalConfig(**kwargs)
