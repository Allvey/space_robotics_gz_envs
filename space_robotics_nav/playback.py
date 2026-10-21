"""Continuous goals for visualization; training/evaluation resets are unchanged."""

import json
import secrets
import subprocess

import numpy as np
import torch

from .env import NavigationEnv


class NoValidTargetError(RuntimeError):
    pass


def follow_camera(backend):
    """Activate the original Gazebo GUI camera plugin in this worker's partition."""
    try:
        result = subprocess.run(
            [
                "docker",
                "exec",
                backend.name,
                "gz",
                "service",
                "-s",
                "/gui/follow",
                "--reqtype",
                "gz.msgs.StringMsg",
                "--reptype",
                "gz.msgs.Boolean",
                "--timeout",
                "500",
                "--req",
                "data: " + json.dumps(backend.spec["robot"]["name"]),
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.returncode == 0 and "data: true" in result.stdout
    except subprocess.TimeoutExpired:
        return False


def playback_seed(seed=None):
    if seed is None:
        return secrets.randbelow(2**31)
    if not 0 <= seed < 2**63 - 12345:
        raise ValueError("Seed must be between 0 and 2**63 - 12346")
    return seed


def sample_next_target(terrain, position, rng, previous_targets=()):
    """Keep the native goal distance, terrain mask and random final heading."""
    recent = np.asarray(previous_targets, dtype=float).reshape(-1, 3)[-32:, :2]
    separation = min(1.0, terrain.target_distance * 0.2)
    for _ in range(2000):
        theta = rng.uniform(-np.pi, np.pi)
        xy = np.asarray(position[:2]) + terrain.target_distance * np.array(
            [np.cos(theta), np.sin(theta)]
        )
        if len(recent) and (np.linalg.norm(recent - xy, axis=1) < separation).any():
            continue
        _, invalid = terrain.check_if_target_is_valid(
            torch.tensor([0]), torch.as_tensor(xy[None]), "cpu"
        )
        if invalid:
            continue
        z = (
            terrain._heightmap_manager.get_height_at(torch.as_tensor(xy[None])).item()
            + 0.5
        )
        if np.isfinite(z):
            return np.r_[xy, z], rng.uniform(-np.pi, np.pi)
    raise NoValidTargetError("No distinct valid target near the current rover position")


class PlaybackEnv(NavigationEnv):
    def next_goal(self, *, seed, previous_targets=()):
        """Start a new task at the actual pose without resetting Gazebo bodies."""
        if not self.done:
            raise RuntimeError("Finish the current goal before requesting another")
        # Brake through the original controller. No pose, joint, rock or physics reset.
        self.state = self.backend.step([0.0, 0.0])
        self.terrain.update(self.state)
        self.rng = np.random.default_rng(seed)
        target, heading = sample_next_target(
            self.terrain, self.state["position"], self.rng, previous_targets
        )
        self.origin = np.asarray(self.state["position"], dtype=float).copy()
        self.target, self.target_heading = target, heading
        self.action_manager.action.zero_()
        self.action_manager.prev_action.zero_()
        self.episode_length_buf.zero_()
        self.episode_return = 0.0
        self.reward_totals = {name: 0.0 for name in self.reward_totals}
        self._command()
        self.initial_distance = float(torch.linalg.vector_norm(self.command[0, :2]))
        self.min_distance = self.initial_distance
        self.done = False
        return self._observation(), {}


def advance_goal(env, info, *, seed, previous_targets=(), reset_between=False):
    """Recover after collisions/bounds failures; otherwise continue from the rover."""
    rpy = getattr(env, "state", {}).get("rpy", [0, 0, 0])
    tipped = bool(np.max(np.abs(rpy[:2])) > np.pi / 3)
    if reset_between or info["collision"] or info["out_of_bounds"] or tipped:
        obs, _ = env.reset(seed=seed)
        mode = (
            "reset_requested"
            if reset_between
            else (
                "reset_after_failure"
                if info["collision"] or info["out_of_bounds"]
                else "reset_tipped_rover"
            )
        )
    else:
        try:
            obs, _ = env.next_goal(seed=seed, previous_targets=previous_targets)
            return obs, "continue_from_current_pose"
        except NoValidTargetError:
            obs, _ = env.reset(seed=seed)
            mode = "reset_no_valid_local_goal"
    # A recovery's native sample could revisit an old target. Preserve the native
    # spawn, but replace a repeated target without a second physical reset.
    recent = np.asarray(previous_targets, dtype=float).reshape(-1, 3)[-32:, :2]
    if (
        len(recent)
        and (
            np.linalg.norm(recent - env.target[:2], axis=1)
            < min(1, env.terrain.target_distance * 0.2)
        ).any()
    ):
        env.done = True
        obs, _ = env.next_goal(seed=seed, previous_targets=previous_targets)
    return obs, mode
