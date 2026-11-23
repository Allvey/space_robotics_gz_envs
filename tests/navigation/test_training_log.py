"""Iteration aggregation, real PPO logging hooks and TensorBoard axis contracts."""

import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from space_robotics_nav import training_log
from space_robotics_nav.config import ROOT
from space_robotics_nav.training_log import SEPARATOR, TrainingLogger, training_plan


def sample(*, reward=1.0, distance=9.0, approach=0.1, speed=0.2):
    return dict(
        reward=reward,
        distance=distance,
        approach_speed=approach,
        linear_speed=speed,
        collision=False,
        success=False,
    )


def episode(*, success=False, collision=False):
    return dict(
        length=2,
        distance=0.1,
        min_distance=0.1,
        distance_progress=8.9,
        success=success,
        collision=collision,
        out_of_bounds=False,
        position_reached=success,
        reward_terms={"distance_to_target": 0.1},
    ) | {"return": 4.0}


@pytest.mark.parametrize(
    "options,expected",
    [
        ({"iterations": 834}, (834, 50040)),
        ({"steps": 50000}, (834, 50040)),
        ({"steps": 3000}, (50, 3000)),
        ({}, (50, 3000)),
    ],
)
def test_budget_resolves_to_complete_iterations(options, expected):
    assert training_plan(rollout_steps=60, default_steps=3000, **options) == expected


@pytest.mark.parametrize(
    "options",
    [{"iterations": 0}, {"steps": 59}, {"iterations": 2, "steps": 120}],
)
def test_invalid_budgets_are_rejected(options):
    with pytest.raises(ValueError):
        training_plan(rollout_steps=60, default_steps=3000, **options)


def test_fresh_iteration_metrics_without_finished_episodes(
    tmp_path, capsys, monkeypatch
):
    clock = NS(now=20.0)
    monkeypatch.setattr(training_log, "time", NS(monotonic=lambda: clock.now))
    logger = TrainingLogger(
        total_iterations=2, steps_per_iteration=2, output_dir=tmp_path, started_at=10.0
    )
    writer = Mock()
    try:
        logger.record_step([sample(reward=1, distance=9), sample(reward=3, distance=7)])
        logger.record_step([sample(reward=5, distance=5), sample(reward=7, distance=3)])
        assert capsys.readouterr().out == ""
        logger.finish_iteration(writer)
        output = capsys.readouterr().out
        assert output.count(SEPARATOR) == 2
        assert "iteration: 1/2\n" in output
        assert "progress: 50.00%\n" in output
        assert "estimated_remaining_time: 00:00:10\n" in output
        assert "reward_mean: 4.000000\n" in output
        assert "distance_to_goal_mean_m: 6.000000\n" in output
        assert "approach_speed_mean_mps: 0.100000\n" in output
        assert "linear_speed_mean_mps: 0.200000\n" in output
        assert "collision_step_rate: 0.00% (0/4)" in output
        for excluded in ("completed_episodes", "optimizer_steps", "n/a", "vector_step"):
            assert excluded not in output
        assert logger.episode_count == 0

        clock.now = 31.0
        for _ in range(2):
            logger.record_step([sample(reward=-1), sample(reward=-1)])
        logger.finish_iteration(writer)
        output = capsys.readouterr().out
        assert "iteration: 2/2" in output
        assert "progress: 100.00%" in output
        assert "estimated_remaining_time: 00:00:00" in output
        assert "reward_mean: -1.000000" in output
        records = [
            json.loads(line)
            for line in (tmp_path / "iterations.jsonl").read_text().splitlines()
        ]
        assert [r["reward_mean"] for r in records] == [4, -1]
        assert [r["sample_count"] for r in records] == [4, 4]
    finally:
        logger.close()


def test_collision_denominator_and_parallel_episode_records(tmp_path, capsys):
    logger = TrainingLogger(
        total_iterations=1, steps_per_iteration=1, output_dir=tmp_path
    )
    a, b = sample(), sample()
    a.update(collision=True, episode=episode(collision=True))
    b.update(success=True, episode=episode(success=True))
    try:
        logger.record_step([a, b])
        logger.finish_iteration(Mock())
        output = capsys.readouterr().out
        assert "collision_step_rate: 50.00% (1/2)" in output
        assert "success_count: 1" in output
        records = [
            json.loads(line)
            for line in (tmp_path / "episodes.jsonl").read_text().splitlines()
        ]
        assert [r["episode"] for r in records] == [1, 2]
        assert [r["env_id"] for r in records] == [0, 1]
    finally:
        logger.close()
    events = EventAccumulator(str(tmp_path / "episode_metrics")).Reload()
    rates = events.Scalars("Episode/collision_rate")
    assert [(v.step, v.value) for v in rates] == [(1, 1.0), (2, 0.5)]


def test_partial_sampling_is_not_reported_as_a_completed_iteration(capsys):
    logger = TrainingLogger(total_iterations=1, steps_per_iteration=2)
    logger.record_step([sample()])
    with pytest.raises(ValueError, match="incomplete"):
        logger.finish_iteration(Mock())
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("version", [1, 2])
def test_real_ppo_writes_after_update_on_iteration_axis(tmp_path, capsys, version):
    from space_robotics_nav.learning import create_agent, load_config

    torch.set_num_threads(2)
    torch.manual_seed(13)
    spaces = NS(
        num_envs=2,
        device=torch.device("cpu"),
        observation_space=gym.spaces.Box(-np.inf, np.inf, (10206,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, (2,), dtype=np.float32),
    )
    config = load_config(ROOT / "configs/navigation/ppo.yaml" if version == 1 else None)
    config["agent"].update(rollouts=2, mini_batches=1, learning_epochs=1)
    config["agent"]["experiment"].update(write_interval=1, checkpoint_interval=0)
    agent = create_agent(spaces, config, directory=tmp_path)
    agent.init()
    logger = TrainingLogger(
        total_iterations=2, steps_per_iteration=2, output_dir=tmp_path
    )
    agent.training_logger = logger
    agent.enable_training_mode(True)
    obs = torch.randn(2, 10206)
    try:
        for step in range(4):
            with torch.no_grad():
                actions, _ = agent.act(obs, None, timestep=step, timesteps=4)
            infos = [sample(reward=1 if step < 2 else 2) for _ in range(2)]
            if step == 3:
                infos[0].update(success=True, episode=episode(success=True))
                agent.track_data("Episode/return", 4.0)
            next_obs = torch.randn_like(obs)
            agent.record_transition(
                observations=obs,
                states=None,
                actions=actions,
                rewards=torch.tensor(
                    [[info["reward"]] for info in infos], dtype=torch.float32
                ),
                next_observations=next_obs,
                next_states=None,
                terminated=torch.tensor([[info["success"]] for info in infos]),
                truncated=torch.zeros(2, 1, dtype=torch.bool),
                infos={"individual": infos},
                timestep=step,
                timesteps=4,
            )
            assert logger.iteration == step // 2
            agent.post_interaction(timestep=step, timesteps=4)
            assert logger.iteration == (step + 1) // 2
            obs = next_obs
    finally:
        agent.training_logger = None
        agent.writer.close()
        logger.close()
    output = capsys.readouterr().out
    assert output.count(SEPARATOR) == 4
    events = EventAccumulator(str(tmp_path / "ppo")).Reload()
    assert [(e.step, e.value) for e in events.Scalars("Train/reward_mean")] == [
        (1, 1.0),
        (2, 2.0),
    ]
    assert [e.step for e in events.Scalars("Loss / Policy loss")] == [1, 2]
    assert not any(tag.startswith("Episode/") for tag in events.Tags()["scalars"])
    if version == 2:
        assert [
            e.step for e in events.Scalars("Policy / Collection KL after update")
        ] == [1, 2]
    episodes = EventAccumulator(str(tmp_path / "episode_metrics")).Reload()
    assert [(e.step, e.value) for e in episodes.Scalars("Episode/success")] == [
        (1, 1.0)
    ]


@pytest.mark.parametrize(
    "heading_offset,expected_speed", [(0.0, 0.5), (-np.pi / 2, 0.25)]
)
def test_environment_measures_same_goal_motion_across_resets(
    monkeypatch, heading_offset, expected_speed
):
    from space_robotics_nav import env as env_module

    monkeypatch.setattr(env_module, "REWARDS", {})
    for name in ("is_success", "collision_with_obstacles", "far_from_target"):
        monkeypatch.setattr(
            env_module.terminations, name, lambda *a, **kw: torch.tensor([False])
        )
    env = env_module.NavigationEnv.__new__(env_module.NavigationEnv)
    env.cfg = NS(control_dt=0.2, success_distance=0.18, contact_threshold=1)
    env.done = False
    env.target = np.array([10.0, 0.0, 0.0])
    env.target_heading = 0.0
    env.state = {"position": [0, 0, 0]}
    state = dict(
        position=[1.0, 0, 0],
        rpy=[0, 0, np.pi / 2],
        linear_velocity=[0.25, 0.5, 0],
        obstacle_forces=[[0, 0, 0]],
        sim_time=0.2,
        iterations=8,
    )
    env.backend = NS(step=lambda action: state)
    env.terrain = NS(
        spec={"heading_offset": heading_offset},
        update=lambda state: None,
        contains=lambda p: True,
    )
    env.action_manager = NS(action=torch.zeros(1, 2), prev_action=torch.zeros(1, 2))
    env.episode_length_buf = torch.zeros(1, dtype=torch.long)
    env.max_episode_length = 750
    env.scene = NS(sensors={"contact_sensor": NS(data=NS(force_matrix_w=NS()))})
    env.command = torch.zeros(1, 4)
    env.episode_return = 0.0
    env.reward_totals = {}
    env.initial_distance = env.min_distance = 10.0
    env._observation = lambda: np.zeros(10206)
    _, reward, _, _, info = env.step([0.5, 1.0])
    assert info["reward"] == reward == 0
    assert info["approach_speed"] == pytest.approx(5.0)
    assert info["linear_speed"] == pytest.approx(expected_speed)
    # Simulate an autoreset to a new location and a completely different goal.
    env.state = {"position": [20, 0, 0]}
    env.target = np.array([21.0, 0.0, 0.0])
    state["position"] = [20.1, 0, 0]
    _, _, _, _, info = env.step([0, 0])
    assert info["approach_speed"] == pytest.approx(0.5, abs=1e-5)
