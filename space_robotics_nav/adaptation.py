"""Explicit training corrections around the unmodified upstream implementation."""

import torch
from skrl.agents.torch.ppo import PPO

from .vendor.models import GaussianPolicyConv, ValueNetworkConv


class FrozenBatchNorm:
    """Keep normalization identical during collection, optimization and playback.

    Affine parameters still receive gradients. Running statistics stay fixed at
    their initialization (or at the statistics restored from a v2 checkpoint).
    This retains upstream layers and state-dict keys without batch-dependent PPO
    likelihoods or running-stat changes outside the optimizer.
    """

    def train(self, mode=True):
        super().train(mode)
        for module in self.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        return self

    def features(self, states):
        obs = states["observations"]
        if self.encoder_input_size is None:
            return obs
        return torch.cat(
            [
                obs[:, : self.mlp_input_size],
                self.encoder(obs[:, self.mlp_input_size :]),
            ],
            dim=1,
        )


class NavigationPolicy(FrozenBatchNorm, GaussianPolicyConv):
    def compute(self, states, role="policy"):
        x = self.features(states)
        for layer in self.mlp:
            x = layer(x)
        return x, {"log_std": self.log_std_parameter}


class NavigationValue(FrozenBatchNorm, ValueNetworkConv):
    def compute(self, states, role="value"):
        x = self.features(states)
        for layer in self.mlp:
            x = layer(x)
        return x, {}


class NavigationPPO(PPO):
    """Bootstrap timeouts from terminal observations and cut the GAE trace."""

    def record_transition(self, **transition):
        if not self.training:
            return super().record_transition(**transition)
        transition = transition.copy()
        next_obs = transition["next_observations"]
        # A simultaneous real termination takes precedence over the time limit.
        timeouts = transition["truncated"] & ~transition["terminated"]
        transition["truncated"] = timeouts
        transition["rewards"] = transition["rewards"].clone()
        if timeouts.any():
            terminal_obs = next_obs.clone()
            individual = transition["infos"].get("individual", [])
            for index in timeouts.flatten().nonzero().flatten().tolist():
                if (
                    index >= len(individual)
                    or "final_observation" not in individual[index]
                ):
                    raise ValueError("Timeout bootstrap requires final_observation")
                terminal_obs[index] = torch.as_tensor(
                    individual[index]["final_observation"], device=self.device
                )
            transition["next_observations"] = terminal_obs
        # skrl adds gamma * V(terminal_obs) to timeout rewards and, with
        # time_limit_bootstrap=True, cuts both termination and truncation traces.
        super().record_transition(**transition)
        self._current_next_observations = next_obs

    @torch.no_grad()
    def rollout_kl(self):
        """Compare current likelihoods with those actually used in collection."""
        obs, actions, old_log_prob = [
            self.memory.get_tensor_by_name(name).flatten(0, 1)
            for name in ("observations", "actions", "log_prob")
        ]
        divergences = []
        for start in range(0, len(obs), 128):
            means, output = self.policy.compute(
                {
                    "observations": self._observation_preprocessor(
                        obs[start : start + 128]
                    )
                },
                role="policy",
            )
            distribution = torch.distributions.Normal(
                means, output["log_std"].clamp(-20, 2).exp()
            )
            log_prob = distribution.log_prob(actions[start : start + 128]).sum(
                -1, keepdim=True
            )
            delta = log_prob - old_log_prob[start : start + 128]
            divergences.append((delta.exp() - 1 - delta).flatten())
        return torch.cat(divergences).mean().item()

    def update(self, *, timestep, timesteps):
        self.track_data("Policy / Collection KL before update", self.rollout_kl())
        steps = 0

        def count_step(*_):
            nonlocal steps
            steps += 1

        handle = self.optimizer.register_step_post_hook(count_step)
        try:
            super().update(timestep=timestep, timesteps=timesteps)
        finally:
            handle.remove()
        self.track_data("Policy / Collection KL after update", self.rollout_kl())
        self.track_data("Optimization / Gradient steps", steps)
        self.track_data(
            "Optimization / Gradient step fraction",
            steps / (self.cfg.learning_epochs * self.cfg.mini_batches),
        )
        self.track_data("Optimization / Zero-update rollout", float(steps == 0))
