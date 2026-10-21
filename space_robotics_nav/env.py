from types import SimpleNamespace as NS

import gymnasium as gym
import numpy as np
import torch

from .backend import GazeboBackend
from .config import NavigationConfig
from .terrain import Terrain
from .vendor import observations, rewards, terminations

# Exact function bodies and weights from upstream RewardsCfg.
REWARDS = {
    "distance_to_target": (
        rewards.distance_to_target_reward,
        5.0,
        {"command_name": "target_pose"},
    ),
    "reached_target": (
        rewards.reached_target,
        5.0,
        {"command_name": "target_pose", "threshold": 0.18},
    ),
    "oscillation": (rewards.oscillation_penalty, -0.05, {}),
    "angle_to_target": (
        rewards.angle_to_target_penalty,
        -1.5,
        {"command_name": "target_pose"},
    ),
    "heading_soft_contraint": (
        rewards.heading_soft_contraint,
        -0.5,
        {"asset_cfg": None},
    ),
    "collision": (
        rewards.collision_penalty,
        -3.0,
        {"sensor_cfg": NS(name="contact_sensor"), "threshold": 1.0},
    ),
    "far_from_target": (
        rewards.far_from_target_reward,
        -2.0,
        {"command_name": "target_pose"},
    ),
    "angle_diff": (rewards.angle_to_goal_reward, 5.0, {"command_name": "target_pose"}),
}


class NavigationEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg=None, *, backend=None):
        self.cfg = cfg or NavigationConfig()
        self.terrain = Terrain(self.cfg)
        self.backend = backend or GazeboBackend(self.cfg)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (10206,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.num_envs, self.device = 1, "cpu"
        self.max_episode_length = int(
            np.ceil(self.cfg.episode_seconds / self.cfg.control_dt)
        )
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.action_manager = NS(
            action=torch.zeros(1, 2), prev_action=torch.zeros(1, 2)
        )
        self.command = torch.zeros(1, 4)
        self.command_manager = NS(get_command=lambda _: self.command)
        self.scene = NS(
            terrain=self.terrain,
            sensors={
                "contact_sensor": NS(
                    data=NS(force_matrix_w=NS(torch=torch.zeros(1, 1, 1, 3)))
                ),
                "height_scanner": NS(data=NS()),
            },
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        self.done = True

    def _command(self):
        delta = self.target - np.asarray(self.state["position"])
        yaw = self.state["rpy"][2] + self.terrain.spec["heading_offset"]
        c, s = np.cos(yaw), np.sin(yaw)
        self.command[0] = torch.tensor(
            [
                c * delta[0] + s * delta[1],
                -s * delta[0] + c * delta[1],
                delta[2],
                (self.target_heading - yaw + np.pi) % (2 * np.pi) - np.pi,
            ]
        )

    def _observation(self):
        pos, hits = self.terrain.scan(self.state["position"], self.state["rpy"])
        data = self.scene.sensors["height_scanner"].data
        data.pos_w = NS(torch=torch.as_tensor(pos[None], dtype=torch.float32))
        data.ray_hits_w = NS(torch=torch.as_tensor(hits[None], dtype=torch.float32))
        scalar = torch.cat(
            [
                self.action_manager.action,
                observations.distance_to_target_euclidean(self, "target_pose") * 0.11,
                observations.angle_to_target_observation(self, "target_pose") / np.pi,
                observations.angle_diff(self, "target_pose") / np.pi,
            ],
            dim=1,
        )
        scan = observations.height_scan_rover(self, NS(name="height_scanner"))
        obs = torch.cat([scalar, scan], dim=1)[0].numpy().astype(np.float32)
        if not np.isfinite(obs).all():
            raise RuntimeError("Non-finite navigation observation")
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.origin, yaw, self.target, self.target_heading = self.terrain.sample(
            self.rng
        )
        self.state = self.backend.reset([*self.origin[:2], self.origin[2] + 0.5, yaw])
        self.terrain.update(self.state)
        self.action_manager.action.zero_()
        self.action_manager.prev_action.zero_()
        self.episode_length_buf.zero_()
        self.episode_return = 0.0
        self.reward_totals = {name: 0.0 for name in REWARDS}
        self._command()
        self.initial_distance = float(torch.linalg.vector_norm(self.command[0, :2]))
        self.min_distance = self.initial_distance
        self.done = False
        return self._observation(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("reset() is required after termination")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Expected finite [linear, angular] action")
        self.action_manager.prev_action.copy_(self.action_manager.action)
        self.action_manager.action.copy_(torch.from_numpy(action)[None])
        self.state = self.backend.step(action)
        self.terrain.update(self.state)
        self.episode_length_buf += 1
        self.scene.sensors["contact_sensor"].data.force_matrix_w.torch = torch.tensor(
            self.state["obstacle_forces"], dtype=torch.float32
        ).reshape(1, -1, 1, 3)
        # Isaac Lab evaluates reward/termination BEFORE updating the target command.
        terms = {}
        for name, (fn, weight, params) in REWARDS.items():
            params = params.copy()
            if name == "reached_target":
                params["threshold"] = self.cfg.success_distance
            if name == "collision":
                params["threshold"] = self.cfg.contact_threshold
            terms[name] = (
                float(fn(self, **params).item()) * weight * self.cfg.control_dt
            )
        reward = sum(terms.values())
        success = bool(
            terminations.is_success(
                self, "target_pose", self.cfg.success_distance
            ).item()
        )
        collision = bool(
            terminations.collision_with_obstacles(
                self, NS(name="contact_sensor"), self.cfg.contact_threshold
            ).item()
        )
        far = bool(terminations.far_from_target(self, "target_pose").item())
        out_of_bounds = not self.terrain.contains(self.state["position"])
        terminated = success or collision or far or out_of_bounds
        truncated = bool(self.episode_length_buf[0] >= self.max_episode_length)
        self.episode_return += reward
        for name, value in terms.items():
            self.reward_totals[name] += value
        self._command()
        distance = float(torch.linalg.vector_norm(self.command[0, :2]))
        self.min_distance = min(self.min_distance, distance)
        self.done = terminated or truncated
        info = {
            "success": success,
            "collision": collision,
            "far_from_target": far,
            "out_of_bounds": out_of_bounds,
            "distance": distance,
            "heading_error": abs(float(self.command[0, 3])),
            "reward_terms": terms,
            "sim_time": self.state["sim_time"],
            "iterations": self.state["iterations"],
        }
        if self.done:
            info["episode"] = {
                "return": self.episode_return,
                "length": int(self.episode_length_buf[0]),
                "success": success,
                "collision": collision,
                "out_of_bounds": out_of_bounds,
                "distance": distance,
                "distance_progress": self.initial_distance - distance,
                "min_distance": self.min_distance,
                "position_reached": self.min_distance < self.cfg.success_distance,
                "reward_terms": self.reward_totals.copy(),
            }
        return self._observation(), reward, terminated, truncated, info

    def close(self):
        self.backend.close()
