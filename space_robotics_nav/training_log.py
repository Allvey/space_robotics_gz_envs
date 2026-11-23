"""Per-iteration training metrics, with separate complete-episode records."""

import json
import math
import time
from pathlib import Path

from skrl.agents.torch.ppo import PPO
from skrl.utils.tensorboard import SummaryWriter

SEPARATOR = "—" * 72


def training_plan(*, iterations=None, steps=None, rollout_steps, default_steps):
    """Resolve the budget into complete sampling/update cycles."""
    if rollout_steps < 1:
        raise ValueError("rollouts must be positive")
    if iterations is not None and steps is not None:
        raise ValueError("Choose either --iterations or --steps")
    if iterations is not None:
        if iterations < 1:
            raise ValueError("iterations must be positive")
        return iterations, iterations * rollout_steps
    steps = default_steps if steps is None else steps
    if steps < rollout_steps:
        raise ValueError("Require at least one complete rollout")
    iterations = (steps + rollout_steps - 1) // rollout_steps
    return iterations, iterations * rollout_steps


def duration(seconds):
    hours, seconds = divmod(math.ceil(max(0, seconds)), 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingLogger:
    """Aggregate raw samples until PPO has finished the corresponding update."""

    def __init__(
        self, *, total_iterations, steps_per_iteration, output_dir=None, started_at=None
    ):
        if total_iterations < 1 or steps_per_iteration < 1:
            raise ValueError("Iterations and sampling length must be positive")
        self.total_iterations = total_iterations
        self.steps_per_iteration = steps_per_iteration
        self.started_at = time.monotonic() if started_at is None else started_at
        self.iteration = 0
        self.episode_count = self.successes = self.collisions = 0
        self.iterations_file = self.episodes_file = self.episode_writer = None
        self.closed = False
        self._reset_samples()
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            try:
                self.iterations_file = (output_dir / "iterations.jsonl").open(
                    "a", buffering=1
                )
                self.episodes_file = (output_dir / "episodes.jsonl").open(
                    "a", buffering=1
                )
                self.episode_writer = SummaryWriter(str(output_dir / "episode_metrics"))
            except BaseException:
                self.close()
                raise

    def _reset_samples(self):
        self.vector_steps = self.sample_count = self.collision_samples = 0
        self.success_count = 0
        self.sums = dict(
            reward_mean=0.0,
            distance_to_goal_mean_m=0.0,
            approach_speed_mean_mps=0.0,
            linear_speed_mean_mps=0.0,
        )

    def record_step(self, infos):
        if not infos or self.vector_steps >= self.steps_per_iteration:
            raise ValueError("Expected samples for the current incomplete iteration")
        self.vector_steps += 1
        self.sample_count += len(infos)
        for env_id, info in enumerate(infos):
            self.sums["reward_mean"] += info["reward"]
            self.sums["distance_to_goal_mean_m"] += info["distance"]
            self.sums["approach_speed_mean_mps"] += info["approach_speed"]
            self.sums["linear_speed_mean_mps"] += info["linear_speed"]
            self.collision_samples += int(info["collision"])
            self.success_count += int(info["success"])
            if "episode" in info:
                self._record_episode(info["episode"], env_id)

    def _record_episode(self, episode, env_id):
        self.episode_count += 1
        self.successes += int(episode["success"])
        self.collisions += int(episode["collision"])
        if self.episodes_file is not None:
            record = {
                **episode,
                "episode": self.episode_count,
                "iteration": self.iteration + 1,
                "env_id": env_id,
            }
            self.episodes_file.write(json.dumps(record, allow_nan=False) + "\n")
        if self.episode_writer is not None:
            for key, value in episode.items():
                if isinstance(value, (int, float, bool)):
                    self.episode_writer.add_scalar(
                        tag=f"Episode/{key}", value=value, timestep=self.episode_count
                    )
            for key, value in episode["reward_terms"].items():
                self.episode_writer.add_scalar(
                    tag=f"Episode reward terms/{key}",
                    value=value,
                    timestep=self.episode_count,
                )
            for key, count in (
                ("success_rate", self.successes),
                ("collision_rate", self.collisions),
            ):
                self.episode_writer.add_scalar(
                    tag=f"Episode/{key}",
                    value=count / self.episode_count,
                    timestep=self.episode_count,
                )

    def finish_iteration(self, writer):
        if self.vector_steps != self.steps_per_iteration:
            raise ValueError("Cannot report an incomplete PPO iteration")
        if self.iteration >= self.total_iterations:
            raise ValueError("Training exceeded its iteration budget")
        self.iteration += 1
        elapsed = max(0.0, time.monotonic() - self.started_at)
        remaining = elapsed / self.iteration * (self.total_iterations - self.iteration)
        record = {
            "iteration": self.iteration,
            "total_iterations": self.total_iterations,
            "progress": self.iteration / self.total_iterations,
            "elapsed_seconds": elapsed,
            "estimated_remaining_seconds": remaining,
            **{key: value / self.sample_count for key, value in self.sums.items()},
            "collision_step_rate": self.collision_samples / self.sample_count,
            "success_count": self.success_count,
            "collision_samples": self.collision_samples,
            "sample_count": self.sample_count,
        }
        metrics = {
            "iteration": f"{self.iteration}/{self.total_iterations}",
            "progress": f"{record['progress']:.2%}",
            "elapsed_time": duration(elapsed),
            "estimated_remaining_time": duration(remaining),
            **{key: f"{record[key]:.6f}" for key in self.sums},
            "collision_step_rate": (
                f"{record['collision_step_rate']:.2%} "
                f"({self.collision_samples}/{self.sample_count})"
            ),
            "success_count": self.success_count,
        }
        print(
            "\n".join(
                [SEPARATOR]
                + [f"        {key}: {value}" for key, value in metrics.items()]
                + [SEPARATOR]
            ),
            flush=True,
        )
        if self.iterations_file is not None:
            self.iterations_file.write(json.dumps(record, allow_nan=False) + "\n")
        for key in (*self.sums, "collision_step_rate", "success_count"):
            writer.add_scalar(
                tag=f"Train/{key}", value=record[key], timestep=self.iteration
            )
        for key in ("progress", "elapsed_seconds", "estimated_remaining_seconds"):
            writer.add_scalar(
                tag=f"Progress/{key}", value=record[key], timestep=self.iteration
            )
        if self.episode_writer is not None:
            self.episode_writer.flush()
        self._reset_samples()

    def close(self):
        if self.closed:
            return
        self.closed = True
        for resource in (self.iterations_file, self.episodes_file, self.episode_writer):
            if resource is not None:
                resource.close()


class TrainingPPO(PPO):
    """Write iteration metrics after the full PPO update, including diagnostics."""

    training_logger = None

    def record_transition(self, **transition):
        super().record_transition(**transition)
        if self.training and self.training_logger is not None:
            # Environment info carries raw rewards, before timeout bootstrapping.
            self.training_logger.record_step(transition["infos"]["individual"])

    def write_tracking_data(self, *, timestep, timesteps):
        if self.training_logger is None:
            super().write_tracking_data(timestep=timestep, timesteps=timesteps)

    def post_interaction(self, *, timestep, timesteps):
        super().post_interaction(timestep=timestep, timesteps=timesteps)
        if (
            self.training
            and self.training_logger is not None
            and not self._rollout % self.cfg.rollouts
            and timestep >= self.cfg.learning_starts
        ):
            self.training_logger.finish_iteration(self.writer)
            # Complete-episode metrics have their own writer and episode axis.
            for key in list(self.tracking_data):
                if key.startswith(("Episode/", "Episode /", "Reward / Total reward")):
                    del self.tracking_data[key]
            super().write_tracking_data(
                timestep=self.training_logger.iteration,
                timesteps=self.training_logger.total_iterations,
            )
            self.writer.flush()
