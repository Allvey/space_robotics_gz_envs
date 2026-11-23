"""Train the upstream height-map PPO in this project's native Gazebo worlds."""

import argparse
import json
import signal
import time
from pathlib import Path

import torch
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from .config import CONTRACT, UPSTREAM_COMMIT, NavigationConfig
from .evaluate import compare, evaluate, read_run
from .learning import (
    adaptation_version,
    create_agent,
    load_config,
    minibatch_count,
    restore_state,
    save_state,
)
from .training_log import TrainingLogger, training_plan
from .vector_env import GazeboVectorEnv
from .world import describe


def main():
    def terminate(signum, frame):
        raise KeyboardInterrupt("Training interrupted")

    signal.signal(signal.SIGTERM, terminate)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--world", choices=["moon", "mars"], default="moon")
    budget = p.add_mutually_exclusive_group()
    budget.add_argument(
        "--iterations", type=int, help="Number of complete PPO sampling/update cycles"
    )
    budget.add_argument(
        "--steps",
        type=int,
        help="Legacy vector-step budget, rounded up to complete PPO iterations",
    )
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument(
        "--mini-batches",
        type=int,
        help="Override: v2 auto targets 128 samples per minibatch; v1 uses 60 minibatches",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=0,
        help="0 disables paired before/after evaluation",
    )
    p.add_argument("--eval-seed", type=int, default=10000)
    p.add_argument("--stochastic-eval", action="store_true")
    args = p.parse_args()
    experiment = load_config(args.config)
    rollouts = experiment["agent"]["rollouts"]
    try:
        iterations, steps = training_plan(
            iterations=args.iterations,
            steps=args.steps,
            rollout_steps=rollouts,
            default_steps=experiment["trainer"]["timesteps"],
        )
    except ValueError as exc:
        p.error(str(exc))
    if args.num_envs < 1 or args.eval_episodes < 0 or args.eval_episodes == 1:
        p.error("Require positive num-envs and either 0 or >=2 eval episodes")
    if experiment["agent"]["learning_starts"] >= rollouts:
        p.error("Iteration logging requires learning_starts before the first update")
    if (args.output / "run.json").exists():
        p.error("Output already contains a run; choose a new output directory")
    torch.set_num_threads(2)
    set_seed(args.seed)
    cfg = NavigationConfig(world=args.world, device=args.device, seed=args.seed)
    spec = describe(cfg.world, cfg.assets)
    experiment["agent"]["experiment"]["wandb"] = False
    experiment["agent"]["experiment"]["write_interval"] = rollouts
    if args.mini_batches is not None:
        if (
            not 1
            <= args.mini_batches
            <= experiment["agent"]["rollouts"] * args.num_envs
        ):
            p.error("mini-batches must fit the rollout sample count")
        experiment["agent"]["mini_batches"] = args.mini_batches
    experiment["agent"]["mini_batches"] = minibatch_count(experiment, args.num_envs)
    if args.checkpoint:
        previous = read_run(args.checkpoint)
        if adaptation_version(previous["experiment"]) != adaptation_version(experiment):
            p.error(
                "Resume requires the same adaptation version. Start v2 from scratch; "
                "use --config configs/navigation/ppo.yaml to resume the legacy baseline."
            )
        if (
            previous["environment"]["world"] != cfg.world
            or previous["world"]["sha256"] != spec["sha256"]
        ):
            p.error("Resume requires the same original world and robot")
    metadata = {
        "contract": CONTRACT,
        "upstream_commit": UPSTREAM_COMMIT,
        "environment": cfg.to_dict(),
        "world": spec,
        "experiment": experiment,
        "training_steps": steps,
        "training_iterations": iterations,
        "steps_per_iteration": rollouts,
        "requested_training_steps": (
            args.steps if args.steps is not None else experiment["trainer"]["timesteps"]
        )
        if args.iterations is None
        else None,
        "logging": {
            "training_axis": "iteration",
            "episode_axis": "completed_episode",
            "iteration_records": "iterations.jsonl",
            "episode_records": "episodes.jsonl",
        },
        "num_envs": args.num_envs,
        "action_interface": {
            "layout": ["linear_mps", "angular_radps"],
            "transport": "/cmd_vel",
            "limits": spec["robot"]["limits"],
        },
        "evaluation_seeds": list(
            range(args.eval_seed, args.eval_seed + args.eval_episodes)
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)

    def write_run():
        (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    write_run()
    env = GazeboVectorEnv(cfg, args.num_envs)
    agent = None
    training_logger = None
    try:
        agent = create_agent(env, experiment, directory=args.output)
        trainer = SequentialTrainer(
            env=env,
            agents=agent,
            cfg={
                "timesteps": steps,
                "headless": True,
                "disable_progressbar": True,
                "close_environment_at_exit": False,
                "environment_info": "log",
            },
        )
        if args.checkpoint:
            metadata["resumed_from"] = str(args.checkpoint)
            metadata["previous_run"] = restore_state(agent, args.checkpoint)
        save_state(agent, args.output / "initial.pt", metadata)
        if args.eval_episodes:
            print("Evaluating initial policy", flush=True)
            before = evaluate(
                agent,
                env,
                metadata["evaluation_seeds"],
                stochastic=args.stochastic_eval,
            )
            (args.output / "before.json").write_text(
                json.dumps(before, indent=2) + "\n"
            )
        env.latest = None
        env.episodes.clear()
        started = time.monotonic()
        training_logger = TrainingLogger(
            total_iterations=iterations,
            steps_per_iteration=rollouts,
            output_dir=args.output,
            started_at=started,
        )
        agent.training_logger = training_logger
        requested_steps = metadata["requested_training_steps"]
        if requested_steps is not None and requested_steps != steps:
            print(
                f"Rounded {requested_steps} requested vector steps to {iterations} "
                f"complete iterations ({steps} vector steps).",
                flush=True,
            )
        print(
            f"Training {iterations} PPO iterations in {cfg.world} with {args.num_envs} "
            "independent environments using the original robot and physics",
            flush=True,
        )
        trainer.train()
        agent.training_logger = None
        metadata["elapsed_training_seconds"] = time.monotonic() - started
        metadata["completed_training_iterations"] = training_logger.iteration
        save_state(agent, args.output / "final.pt", metadata)
        write_run()
        (args.output / "training_episodes.json").write_text(
            json.dumps(env.episodes, indent=2) + "\n"
        )
        if args.eval_episodes:
            print("Evaluating final policy", flush=True)
            after = evaluate(
                agent,
                env,
                metadata["evaluation_seeds"],
                stochastic=args.stochastic_eval,
            )
            result = {
                "before": before,
                "after": after,
                "comparison": compare(before, after),
            }
            (args.output / "comparison.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(json.dumps(result["comparison"], indent=2), flush=True)
    except KeyboardInterrupt:
        if agent is not None:
            metadata["interrupted"] = True
            if training_logger is not None:
                metadata["completed_training_iterations"] = training_logger.iteration
            save_state(agent, args.output / "interrupted.pt", metadata)
            write_run()
        print("Training interrupted; saved interrupted.pt", flush=True)
    finally:
        if training_logger is not None:
            training_logger.close()
        if agent is not None:
            agent.training_logger = None
            writer = getattr(agent, "writer", None)
            if writer is not None:
                writer.close()
        env.close()


if __name__ == "__main__":
    main()
