import copy
import dataclasses
from pathlib import Path

import torch
import yaml
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory

from .adaptation import NavigationPolicy, NavigationPPO, NavigationValue
from .config import ROOT
from .vendor import models as _models  # noqa: F401
from .vendor.registry import MODEL_REGISTRY


def load_config(path=None):
    path = Path(path or ROOT / "configs/navigation/ppo_gazebo.yaml")
    result = yaml.safe_load(path.read_text())
    parent = result.pop("extends", None)
    if parent:
        base = load_config(path.parent / parent)
        for key, value in result.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key].update(value)
            else:
                base[key] = value
        return base
    return result


def adaptation_version(experiment):
    version = experiment.get("adaptation", {}).get("version", 1)
    if version not in (1, 2):
        raise ValueError(f"Unsupported navigation adaptation version: {version}")
    return version


def minibatch_count(experiment, num_envs):
    samples = experiment["agent"]["rollouts"] * num_envs
    count = experiment["agent"]["mini_batches"]
    if count == "auto":
        target = experiment["adaptation"]["target_minibatch_samples"]
        if not isinstance(target, int) or target < 1:
            raise ValueError("target_minibatch_samples must be a positive integer")
        count = max(1, round(samples / target))
    if not isinstance(count, int) or not 1 <= count <= samples:
        raise ValueError("mini_batches must fit the rollout sample count")
    return count


def create_agent(env, experiment, *, directory, evaluation=False):
    corrected = adaptation_version(experiment) == 2
    models = {}
    for name, model in experiment["models"].items():
        if not isinstance(model, dict):
            continue
        params = model["params"].copy()
        params["encoder_input_size"] = 10201
        model_class = MODEL_REGISTRY[model["type"]]
        if corrected:
            model_class = {
                "GaussianPolicyConv": NavigationPolicy,
                "ValueNetworkConv": NavigationValue,
            }[model["type"]]
        models[name] = model_class(
            env.observation_space, env.action_space, env.device, **params
        ).to(env.device)
        if corrected:
            models[name].train(False)
    initial_std = experiment.get("adaptation", {}).get("initial_action_std")
    if initial_std is not None:
        std = torch.as_tensor(initial_std, device=env.device, dtype=torch.float32)
        if (
            not corrected
            or std.shape != (2,)
            or not torch.isfinite(std).all()
            or (std <= 0).any()
        ):
            raise ValueError(
                "initial_action_std requires v2 and two finite positive values"
            )
        with torch.no_grad():
            models["policy"].log_std_parameter.copy_(std.log())
    cfg = dataclasses.asdict(PPO_CFG())
    cfg.update(copy.deepcopy(experiment["agent"]))
    cfg["mini_batches"] = minibatch_count(experiment, env.num_envs)
    if corrected:
        cfg["time_limit_bootstrap"] = True
    cfg["experiment"].update(
        directory=str(directory), experiment_name="ppo", wandb=False
    )
    if evaluation:
        cfg["rollouts"] = 1
        cfg["experiment"].update(write_interval=0, checkpoint_interval=0)
    return (NavigationPPO if corrected else PPO)(
        models=models,
        memory=RandomMemory(
            memory_size=cfg["rollouts"], num_envs=env.num_envs, device=env.device
        ),
        cfg=PPO_CFG(**cfg),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def policy_action(agent, observations, stochastic=False, *, generator=None):
    with torch.inference_mode():
        means, output = agent.policy.compute(
            {"observations": observations}, role="policy"
        )
        if stochastic:
            return (
                means
                + torch.randn(
                    means.shape,
                    dtype=means.dtype,
                    device=means.device,
                    generator=generator,
                )
                * output["log_std"].clamp(-20, 2).exp()
            )
        return means


def save_state(agent, path, metadata):
    agent.save(str(path))
    torch.save(
        {
            "metadata": metadata,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        },
        str(path) + ".state",
    )


def restore_state(agent, path):
    agent.load(str(path))
    sidecar = Path(str(path) + ".state")
    if sidecar.exists():
        state = torch.load(sidecar, map_location="cpu", weights_only=True)
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state["metadata"]
    return {}
