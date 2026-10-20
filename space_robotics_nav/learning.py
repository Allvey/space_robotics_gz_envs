import copy
import dataclasses
from pathlib import Path

import torch
import yaml
from skrl.agents.torch.ppo import PPO, PPO_CFG
from skrl.memories.torch import RandomMemory

from .config import ROOT
from .vendor import models as _models  # noqa: F401
from .vendor.registry import MODEL_REGISTRY


def load_config(path=None):
    return yaml.safe_load(
        Path(path or ROOT / "configs/navigation/ppo.yaml").read_text()
    )


def create_agent(env, experiment, *, directory, evaluation=False):
    models = {}
    for name, model in experiment["models"].items():
        if not isinstance(model, dict):
            continue
        params = model["params"].copy()
        params["encoder_input_size"] = 10201
        models[name] = MODEL_REGISTRY[model["type"]](
            env.observation_space, env.action_space, env.device, **params
        ).to(env.device)
    cfg = dataclasses.asdict(PPO_CFG())
    cfg.update(copy.deepcopy(experiment["agent"]))
    cfg["experiment"].update(
        directory=str(directory), experiment_name="ppo", wandb=False
    )
    if evaluation:
        cfg["rollouts"] = 1
        cfg["experiment"].update(write_interval=0, checkpoint_interval=0)
    return PPO(
        models=models,
        memory=RandomMemory(
            memory_size=cfg["rollouts"], num_envs=env.num_envs, device=env.device
        ),
        cfg=PPO_CFG(**cfg),
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=env.device,
    )


def policy_action(agent, observations, stochastic=False):
    with torch.inference_mode():
        means, output = agent.policy.compute(
            {"observations": observations}, role="policy"
        )
        if stochastic:
            return (
                means + torch.randn_like(means) * output["log_std"].clamp(-20, 2).exp()
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
