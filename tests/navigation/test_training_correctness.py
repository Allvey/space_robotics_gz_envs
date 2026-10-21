"""Behavioral regressions for PPO likelihoods and autoreset boundaries."""

from types import SimpleNamespace as NS

import gymnasium as gym
import numpy as np
import pytest
import torch
from skrl.agents.torch.ppo.ppo import compute_gae

from space_robotics_nav.config import ROOT
from space_robotics_nav.learning import (
    adaptation_version,
    create_agent,
    load_config,
    minibatch_count,
)


@pytest.fixture
def agent(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(71)
    env = NS(
        num_envs=3,
        device=torch.device("cpu"),
        observation_space=gym.spaces.Box(-np.inf, np.inf, (10206,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
    )
    cfg = load_config()
    cfg["agent"].update(rollouts=2, mini_batches=1, learning_epochs=1)
    cfg["agent"]["experiment"].update(write_interval=0, checkpoint_interval=0)
    agent = create_agent(env, cfg, directory=tmp_path)
    agent.init()
    agent.enable_training_mode(True)
    return agent


def test_source_config_is_explicit_baseline():
    baseline = load_config(ROOT / "configs/navigation/ppo.yaml")
    corrected = load_config()
    assert adaptation_version(baseline) == 1
    assert adaptation_version(corrected) == 2
    assert corrected["models"] == baseline["models"]
    assert corrected["agent"]["time_limit_bootstrap"] is True
    assert {
        k: v
        for k, v in corrected["agent"].items()
        if k not in ("time_limit_bootstrap", "mini_batches")
    } == {k: v for k, v in baseline["agent"].items() if k != "mini_batches"}
    assert minibatch_count(corrected, 128) == baseline["agent"]["mini_batches"]


@pytest.mark.parametrize("num_envs,expected", [(4, 2), (8, 4), (16, 8), (128, 60)])
def test_minibatch_adaptation_preserves_upstream_sample_count(num_envs, expected):
    cfg = load_config()
    assert minibatch_count(cfg, num_envs) == expected
    cfg["agent"]["mini_batches"] = 1
    assert minibatch_count(cfg, num_envs) == 1


def test_optional_native_exploration_std_is_explicit(agent, tmp_path):
    torch.testing.assert_close(agent.policy.log_std_parameter.exp(), torch.ones(2))
    cfg = load_config(ROOT / "configs/navigation/ppo_gazebo_native_std.yaml")
    env = NS(
        num_envs=3,
        device=agent.device,
        observation_space=agent.observation_space,
        action_space=agent.action_space,
    )
    native = create_agent(env, cfg, directory=tmp_path)
    torch.testing.assert_close(
        native.policy.log_std_parameter.exp(), torch.tensor([0.25, 0.5])
    )
    cfg["adaptation"]["initial_action_std"] = [0, 0.5]
    with pytest.raises(ValueError, match="finite positive"):
        create_agent(env, cfg, directory=tmp_path)


def test_reward_gain_does_not_hide_navigation_regression():
    from space_robotics_nav.evaluate import compare

    def report(reward, progress):
        return {
            "episodes": [
                {
                    "seed": i,
                    "return": reward,
                    "length": 100,
                    "distance_progress": progress,
                }
                for i in range(8)
            ],
            "metrics": {"success": 0, "collision": 0},
        }

    result = compare(report(-6, 0), report(-3, -2))
    assert result["return_improved"]
    assert result["distance_progress_gain"] == -2
    assert not result["distance_progress_improved"]


@pytest.mark.parametrize("role", ["policy", "value"])
def test_encoder_receives_only_complete_heightmap(agent, role):
    model = agent.models[role]
    obs = torch.arange(10206, dtype=torch.float32).repeat(3, 1) / 100
    received = []
    handle = model.encoder.register_forward_pre_hook(
        lambda _, args: received.append(args[0].clone())
    )
    try:
        model.compute({"observations": obs}, role=role)
        changed = obs.clone()
        changed[:, 4] = -123
        model.compute({"observations": changed}, role=role)
    finally:
        handle.remove()
    torch.testing.assert_close(received[0], obs[:, 5:], rtol=0, atol=0)
    torch.testing.assert_close(received[1], received[0], rtol=0, atol=0)


@pytest.mark.parametrize("role", ["policy", "value"])
def test_training_mode_keeps_outputs_and_bn_statistics_consistent(agent, role):
    model = agent.models[role]
    obs = torch.randn(3, 10206)
    obs[:, 5:] += 10
    model.eval()
    before, _ = model.compute({"observations": obs}, role=role)
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    model.train()
    after, _ = model.compute({"observations": obs}, role=role)
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    agent.optimizer.zero_grad()
    after.square().mean().backward()
    bn = next(
        layer for layer in model.modules() if isinstance(layer, torch.nn.BatchNorm2d)
    )
    assert bn.weight.grad is not None and bn.weight.grad.abs().sum() > 0
    assert model.encoder.encoder_layers[0].weight.grad.abs().sum() > 0


def record(agent, obs, next_obs, rewards, terminated, truncated, infos):
    with torch.no_grad():
        actions, _ = agent.act(obs, None, timestep=0, timesteps=2)
    agent.record_transition(
        observations=obs,
        states=None,
        actions=actions,
        rewards=rewards,
        next_observations=next_obs,
        next_states=None,
        terminated=terminated,
        truncated=truncated,
        infos=infos,
        timestep=0,
        timesteps=2,
    )


def test_timeout_uses_terminal_value_and_cuts_trace(agent):
    # Make values independently known; reset observation is intentionally huge.
    agent.value.act = lambda inputs, role: (inputs["observations"][:, :1], {})
    obs = torch.zeros(3, 10206)
    next_obs = torch.zeros_like(obs)
    next_obs[:, 0] = torch.tensor([100.0, 200.0, 3.0])
    terminal = np.zeros(10206, dtype=np.float32)
    terminal[0] = 5
    rewards = torch.ones(3, 1)
    record(
        agent,
        obs,
        next_obs,
        rewards,
        torch.tensor([[False], [True], [False]]),
        torch.tensor([[True], [True], [False]]),
        {"individual": [{"final_observation": terminal}, {}, {}]},
    )
    stored = agent.memory.get_tensor_by_name("rewards")[0].clone()
    torch.testing.assert_close(stored[:, 0], torch.tensor([1 + 0.99 * 5, 1.0, 1.0]))
    torch.testing.assert_close(rewards, torch.ones_like(rewards))
    assert agent._current_next_observations is next_obs
    # Change next-episode reward drastically: the timed-out return stays fixed.
    for next_reward in (1.0, 101.0):
        r = torch.stack([stored, torch.full_like(stored, next_reward)])
        ends = agent.memory.get_tensor_by_name("terminated").clone()
        cuts = agent.memory.get_tensor_by_name("truncated").clone()
        returns, _ = compute_gae(
            rewards=r,
            terminated=ends,
            truncated=cuts,
            values=torch.zeros_like(r),
            last_values=torch.zeros(3, 1),
            discount_factor=0.99,
            lambda_coefficient=0.95,
            time_limit_bootstrap=True,
        )
        assert returns[0, 0, 0] == pytest.approx(5.95)
        assert returns[0, 1, 0] == pytest.approx(1.0)


def test_timeout_without_terminal_observation_fails(agent):
    obs = torch.zeros(3, 10206)
    with pytest.raises(ValueError, match="final_observation"):
        record(
            agent,
            obs,
            obs,
            torch.zeros(3, 1),
            torch.zeros(3, 1, dtype=torch.bool),
            torch.ones(3, 1, dtype=torch.bool),
            {"individual": [{}, {}, {}]},
        )


def test_real_ppo_update_preserves_collection_likelihood_and_updates_weights(agent):
    obs = torch.randn(3, 10206)
    for _ in range(2):
        record(
            agent,
            obs,
            obs,
            torch.tensor([[0.1], [-0.2], [0.3]]),
            torch.zeros(3, 1, dtype=torch.bool),
            torch.zeros(3, 1, dtype=torch.bool),
            {"individual": [{}, {}, {}]},
        )
    before = agent.policy.mlp[-2].weight.detach().clone()
    agent.enable_models_training_mode(True)
    assert abs(agent.rollout_kl()) < 1e-7
    agent.update(timestep=1, timesteps=2)
    assert not torch.equal(before, agent.policy.mlp[-2].weight)
    assert agent.tracking_data["Optimization / Gradient steps"][-1] == 1
    assert agent.tracking_data["Optimization / Zero-update rollout"][-1] == 0
