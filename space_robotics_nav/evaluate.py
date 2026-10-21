"""Paired held-out evaluation. Deterministic and sampled policy modes are separate."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import CONTRACT, NavigationConfig
from .learning import create_agent, policy_action
from .vector_env import GazeboVectorEnv


def evaluate(agent, env, seeds, *, stochastic=False, action_seed=12345):
    old = agent.policy.training
    agent.policy.eval()
    results = []
    devices = [env.device] if env.device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(action_seed)
            for start in range(0, len(seeds), env.num_envs):
                batch = seeds[start : start + env.num_envs]
                padded = batch + [batch[-1]] * (env.num_envs - len(batch))
                generators = [
                    torch.Generator(device=env.device).manual_seed(action_seed + s)
                    for s in padded
                ]
                obs, _ = env.reset_seeds(padded)
                active = set(range(len(batch)))
                while active:
                    if stochastic:
                        with torch.inference_mode():
                            means, output = agent.policy.compute(
                                {"observations": obs}, role="policy"
                            )
                            noise = torch.stack(
                                [
                                    torch.randn(2, device=env.device, generator=g)
                                    for g in generators
                                ]
                            )
                            actions = (
                                means + noise * output["log_std"].clamp(-20, 2).exp()
                            )
                    else:
                        actions = policy_action(agent, obs)
                    obs, _, _, _, info = env.step(actions)
                    for index in list(active):
                        if "episode" in info["individual"][index]:
                            results.append(
                                {
                                    "seed": batch[index],
                                    **info["individual"][index]["episode"],
                                }
                            )
                            active.remove(index)
    finally:
        agent.policy.train(old)
    metrics = {
        name: float(np.mean([r[name] for r in results]))
        for name in (
            "return",
            "success",
            "collision",
            "out_of_bounds",
            "length",
            "distance",
            "distance_progress",
            "min_distance",
            "position_reached",
        )
    }
    metrics["return_per_step"] = sum(r["return"] for r in results) / sum(
        r["length"] for r in results
    )
    return {
        "metrics": metrics,
        "episodes": results,
        "stochastic": stochastic,
        "action_seed": action_seed,
    }


def compare(before, after):
    pre = {r["seed"]: r for r in before["episodes"]}
    post = {r["seed"]: r for r in after["episodes"]}
    if pre.keys() != post.keys():
        raise ValueError("Evaluation seeds must match")
    rng = np.random.default_rng(8273)
    result = {}
    comparisons = [("return", "return", False), ("step_return", "return", True)]
    if all("distance_progress" in row for row in [*pre.values(), *post.values()]):
        comparisons.append(("distance_progress", "distance_progress", False))
    for key, field, scale in comparisons:
        delta = np.array(
            [
                post[s][field] / (post[s]["length"] if scale else 1)
                - pre[s][field] / (pre[s]["length"] if scale else 1)
                for s in sorted(pre)
            ]
        )
        interval = np.quantile(
            rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1),
            [0.025, 0.975],
        )
        result[key + "_gain"] = float(delta.mean())
        result[key + "_gain_95_ci"] = interval.tolist()
        result[key + "_improved"] = bool(interval[0] > 0)
    result["success_rate_change"] = (
        after["metrics"]["success"] - before["metrics"]["success"]
    )
    result["collision_rate_change"] = (
        after["metrics"]["collision"] - before["metrics"]["collision"]
    )
    return result


def read_run(checkpoint):
    for parent in [checkpoint.parent, *checkpoint.parents]:
        path = parent / "run.json"
        if path.is_file():
            data = json.loads(path.read_text())
            if data.get("contract") != CONTRACT:
                raise ValueError(
                    "Checkpoint is not from this native Gazebo navigation task"
                )
            return data
    raise FileNotFoundError("No run.json found alongside checkpoint")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--episodes", type=int, default=12)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    if args.episodes < 1 or args.num_envs < 1:
        p.error("Episode and environment counts must be positive")
    torch.set_num_threads(2)
    metadata = read_run(args.checkpoint)
    cfg = NavigationConfig(**{**metadata["environment"], "device": args.device})
    env = GazeboVectorEnv(cfg, args.num_envs)
    try:
        agent = create_agent(
            env,
            metadata["experiment"],
            directory=args.checkpoint.parent,
            evaluation=True,
        )
        agent.load(str(args.checkpoint))
        result = evaluate(
            agent,
            env,
            list(range(args.seed, args.seed + args.episodes)),
            stochastic=args.stochastic,
        )
        path = args.output or args.checkpoint.with_suffix(
            ".stochastic.json" if args.stochastic else ".deterministic.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["metrics"], indent=2), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
