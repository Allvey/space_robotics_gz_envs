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
