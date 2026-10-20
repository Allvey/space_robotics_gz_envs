from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
from skrl.envs.wrappers.torch.base import Wrapper

from .env import NavigationEnv


class GazeboVectorEnv(Wrapper):
    """One real Gazebo instance per environment, with Isaac-style same-step autoreset."""

    def __init__(self, cfg, num_envs=4):
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.pool = ThreadPoolExecutor(max_workers=num_envs)
        self.envs, self.episodes = [], []
        self.latest = None
        self.closed = False
        self.progress_callback = None
        try:
            for i in range(num_envs):
                # Separate mutable object poses; immutable mesh geometry is cached.
                env = NavigationEnv(cfg)
                env.rng = np.random.default_rng(cfg.seed + i)
                self.envs.append(env)
        except BaseException:
            self.close()
            raise
        super().__init__(
            SimpleNamespace(
                num_envs=num_envs,
                num_agents=1,
                device=cfg.device,
                observation_space=self.envs[0].observation_space,
                action_space=self.envs[0].action_space,
            )
        )

    def reset(self):
        if self.latest is None:
            obs = list(self.pool.map(lambda e: e.reset()[0], self.envs))
            self.latest = torch.as_tensor(np.stack(obs), device=self.device)
        return self.latest, {}

    def reset_seeds(self, seeds):
        if len(seeds) != self.num_envs:
            raise ValueError("One seed per environment required")
        obs = list(
            self.pool.map(
                lambda item: item[0].reset(seed=item[1])[0], zip(self.envs, seeds)
            )
        )
        self.latest = torch.as_tensor(np.stack(obs), device=self.device)
        return self.latest, {}

    def step(self, actions):
        def one(pair):
            env, action = pair
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                info["final_observation"] = obs.copy()
                obs, _ = env.reset()
            return obs, reward, terminated, truncated, info

        result = list(
            self.pool.map(one, zip(self.envs, actions.detach().cpu().numpy()))
        )
        obs, reward, terminated, truncated, infos = zip(*result)
        episodes = [i["episode"] for i in infos if "episode" in i]
        self.episodes.extend(episodes)
        self.latest = torch.as_tensor(np.stack(obs), device=self.device)
        log = {}
        if episodes:
            for name in (
                "return",
                "length",
                "success",
                "collision",
                "distance",
                "distance_progress",
                "out_of_bounds",
            ):
                log[f"Episode/{name}"] = torch.tensor(
                    np.mean([e[name] for e in episodes]), device=self.device
                )
        if self.progress_callback is not None:
            self.progress_callback()
        return (
            self.latest,
            torch.tensor(reward, device=self.device, dtype=torch.float32)[:, None],
            torch.tensor(terminated, device=self.device)[:, None],
            torch.tensor(truncated, device=self.device)[:, None],
            {"log": log, "individual": infos},
        )

    def state(self):
        return None

    def render(self, *args, **kwargs):
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        for env in self.envs:
            env.close()
        self.pool.shutdown(wait=True)
