"""Opt-in integration tests against the original project robots and real Gazebo."""

import os
import time
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import torch

from space_robotics_nav.config import NavigationConfig
from space_robotics_nav.env import NavigationEnv

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_GAZEBO_TESTS") != "1",
    reason="Requires native assets and space-robotics-rl:ionic",
)


def canonical(node):
    return ET.canonicalize(ET.tostring(node, encoding="unicode"), strip_text=True)


@pytest.mark.parametrize("world", ["moon", "mars"])
def test_original_physics_observations_control_and_reset(world):
    cfg = NavigationConfig(world=world)
    env = NavigationEnv(cfg)
    try:
        original = ET.parse(env.backend.spec["source"]).getroot().find("world")
        runtime = ET.parse(env.backend.path).getroot().find("world")
        for key in ("physics", "gravity", "scene", "light"):
            assert canonical(original.find(key)) == canonical(runtime.find(key))
        a = original.find("include/plugin[@name='gz::sim::systems::DiffDrive']")
        b = runtime.find("include/plugin[@name='gz::sim::systems::DiffDrive']")
        assert canonical(a) == canonical(b)
        assert "aau" not in env.backend.spec["robot"]["name"].lower()
        obs, _ = env.reset(seed=41)
        assert (
            obs.shape == (10206,) and obs.dtype == np.float32 and np.isfinite(obs).all()
        )
        target = env.target.copy()
        d = float(torch.linalg.vector_norm(env.command[0, :2]))
        b = float(torch.atan2(env.command[0, 1], env.command[0, 0]))
        a = abs(float(env.command[0, 3]))
        expected = (
            0.2
            / 750
            * (
                5 / (1 + 0.11 * d * d)
                - 0.05 * (3 * 0.2) ** 4
                - (1.5 * abs(b) if abs(b) > 2 else 0)
                + 5 / (1 + d) / (1 + a)
            )
        )
        for _ in range(3):
            obs, reward, done, timeout, info = env.step([0.2, 0])
            if _ == 0:
                assert reward == pytest.approx(expected, abs=1e-9)
            assert np.isfinite(obs).all() and np.isfinite(reward)
            if done or timeout:
                break
        env.reset(seed=41)
        np.testing.assert_equal(env.target, target)
        assert (
            env.action_manager.action.abs().sum() == 0
            and env.episode_length_buf[0] == 0
        )
        ground = env.terrain.rays.heights(np.array([[0.0, 0.0]]))[0]
        state = env.backend.reset([0, 0, float(ground) + 0.5, 0])
        start = np.array(state["position"])
        t = state["sim_time"]
        iterations = state["iterations"]
        yaw = state["rpy"][2] + env.backend.spec["heading_offset"]
        for _ in range(15):
            state = env.backend.step([0.3, 0])
        displacement = np.array(state["position"])[:2] - start[:2]
        assert displacement @ np.array([np.cos(yaw), np.sin(yaw)]) > 0.03
        assert state["sim_time"] - t == pytest.approx(3.0, abs=1e-6)
        assert state["iterations"] - iterations == 120
    finally:
        env.close()


def test_obstacle_contact_feedback_and_visual_annotation():
    from space_robotics_nav.backend import GazeboBackend

    cfg = NavigationConfig(world="moon", settle_steps=1)
    backend = GazeboBackend(cfg, gui=False)
    try:
        # Intersect the robot with the ORIGINAL stack of rocks, not a synthetic obstacle.
        state = backend.reset([0.05, -3.0, 0.5, 0])
        peak = np.linalg.norm(state["obstacle_forces"])
        for _ in range(40):
            state = backend.request({"op": "step", "velocity": [0.0, 0.0], "steps": 1})
            peak = max(peak, np.linalg.norm(state["obstacle_forces"]))
        assert peak > cfg.contact_threshold
        state = backend.reset([0, 0, 2, 0])
        assert np.linalg.norm(state["obstacle_forces"]) == 0
    finally:
        backend.close()


def test_vector_autoreset_timeout_with_real_ppo(tmp_path):
    from space_robotics_nav.learning import create_agent, load_config
    from space_robotics_nav.vector_env import GazeboVectorEnv

    torch.set_num_threads(2)
    # Shorten only the test's episode duration so the native timeout is exercised.
    env = GazeboVectorEnv(NavigationConfig(episode_seconds=0.4), num_envs=2)
    try:
        cfg = load_config()
        cfg["agent"].update(rollouts=2, mini_batches=1, learning_epochs=1)
        cfg["agent"]["experiment"].update(write_interval=0, checkpoint_interval=0)
        agent = create_agent(env, cfg, directory=tmp_path)
        agent.init()
        agent.enable_training_mode(True)
        obs, _ = env.reset_seeds([41000, 41001])
        for step in range(2):
            with torch.no_grad():
                actions, _ = agent.act(obs, None, timestep=step, timesteps=2)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            if step == 1:
                assert truncated.all() and not terminated.any()
                for index, item in enumerate(info["individual"]):
                    assert item["episode"]["length"] == 2
                    assert (
                        item["episode"]["min_distance"] <= item["episode"]["distance"]
                    )
                    assert not item["episode"]["position_reached"]
                    assert not np.allclose(
                        item["final_observation"], next_obs[index].numpy()
                    )
                assert "Reward terms/oscillation" in info["log"]
                assert "Action/linear limit exceeded fraction" in info["log"]
            agent.record_transition(
                observations=obs,
                states=None,
                actions=actions,
                rewards=reward,
                next_observations=next_obs,
                next_states=None,
                terminated=terminated,
                truncated=truncated,
                infos=info,
                timestep=step,
                timesteps=2,
            )
            obs = next_obs
        agent.enable_models_training_mode(True)
        assert abs(agent.rollout_kl()) < 1e-7
        agent.update(timestep=1, timesteps=2)
        assert agent.tracking_data["Optimization / Gradient steps"][-1] == 1
        assert all(torch.isfinite(p).all() for p in agent.policy.parameters())
    finally:
        env.close()


def test_terminal_controller_on_native_moon_pose_fixture():
    from space_robotics_nav.planning.paths import wrap
    from space_robotics_nav.planning.terminal import TerminalController

    env = NavigationEnv(NavigationConfig())
    try:
        env.reset(seed=30002)
        yaw = env.state["rpy"][2]
        # Short synthetic task ONLY for controller integration; performance
        # evaluation retains the original nine-metre targets and random headings.
        env.target[:2] = np.asarray(env.state["position"][:2]) + 0.5 * np.array(
            [np.cos(yaw), np.sin(yaw)]
        )
        env.target_heading = wrap(yaw + 0.2)
        env._command()
        controller = TerminalController(env)
        for _ in range(150):
            action = controller.action([0, 0])
            assert abs(action[0]) <= 0.250001 and abs(action[1]) <= 0.600001
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        assert controller.stats["handoffs"] > 0
        assert info["success"] and not info["collision"]
        assert info["distance"] < 0.18 and info["heading_error"] < 0.1
    finally:
        env.close()


def test_ten_continuous_goals_preserve_native_rover_and_object_state():
    from unittest.mock import patch

    from space_robotics_nav.playback import PlaybackEnv, advance_goal

    # Short task duration tests switching only, not learned navigation performance.
    env = PlaybackEnv(NavigationConfig(episode_seconds=0.4))
    targets = []
    try:
        with patch.object(env.backend, "reset", wraps=env.backend.reset) as reset:
            env.reset(seed=30002)
            for index in range(10):
                targets.append(env.target.copy())
                assert np.linalg.norm(env.target[:2] - env.origin[:2]) == pytest.approx(
                    9, abs=1e-5
                )
                # The first native reset settles physically and may drift a few
                # centimetres; continuous goals use the already settled pose.
                if index:
                    assert env.initial_distance == pytest.approx(9, abs=1e-5)
                for _ in range(2):
                    _, _, terminated, truncated, info = env.step([0, 0])
                assert truncated and not terminated
                if index == 9:
                    break
                position = np.array(env.state["position"])
                iterations = env.state["iterations"]
                obs, mode = advance_goal(
                    env, info, seed=30003 + index, previous_targets=targets
                )
                assert mode == "continue_from_current_pose" and obs.shape == (10206,)
                assert np.isfinite(obs).all() and not env.done
                assert env.state["iterations"] == iterations + env.backend.decimation
                assert np.linalg.norm(env.origin - position) < 0.2
                assert int(env.episode_length_buf[0]) == 0 and env.episode_return == 0
                assert reset.call_count == 1
            assert len(targets) == 10
            for i, target in enumerate(targets):
                assert all(
                    np.linalg.norm(target[:2] - other[:2]) >= 1 for other in targets[:i]
                )
    finally:
        env.close()
