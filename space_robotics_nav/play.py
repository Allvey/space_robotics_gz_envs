"""Watch a policy in the original Gazebo world, using the original rover."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .backend import GazeboBackend
from .config import NavigationConfig
from .env import NavigationEnv
from .evaluate import read_run
from .learning import create_agent, policy_action
from .world import gui_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="0 loops until the window closes or Ctrl+C",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--software-rendering", action="store_true")
    p.add_argument("--max-steps", type=int)
    args = p.parse_args()
    if (
        args.speed <= 0
        or args.episodes < 0
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        p.error("Invalid speed, episode count or step limit")
    torch.set_num_threads(2)
    torch.manual_seed(12345)
    metadata = read_run(args.checkpoint)
    cfg = NavigationConfig(**{**metadata["environment"], "device": args.device})
    backend = GazeboBackend(cfg, gui=True, software=args.software_rendering)
    env = None
    try:
        env = NavigationEnv(cfg, backend=backend)
        spaces = SimpleNamespace(
            num_envs=1,
            device=torch.device(args.device),
            observation_space=env.observation_space,
            action_space=env.action_space,
        )
        agent = create_agent(
            spaces,
            metadata["experiment"],
            directory=args.checkpoint.parent,
            evaluation=True,
        )
        agent.load(str(args.checkpoint))
        agent.policy.eval()
        obs, _ = env.reset(seed=args.seed)
        backend.goal(env.target, env.target_heading)
        path = gui_config(
            cfg, backend.path.with_suffix(".gui.config"), env.origin, env.target
        )
        backend.start_gui(path)
        print(
            f"Loaded {args.checkpoint}\nNative world: {cfg.world}\nGUI log: {backend.gui_log.name}",
            flush=True,
        )
        print(
            "Green beacon = goal; orange bar = target heading. Close window or Ctrl+C to exit.",
            flush=True,
        )
        episode, total_steps = 0, 0
        while not args.episodes or episode < args.episodes:
            print(
                f"Episode {episode+1}, seed={args.seed+episode}, goal={env.target.tolist()}",
                flush=True,
            )
            while True:
                if backend.gui.poll() is not None:
                    if backend.gui.returncode:
                        raise RuntimeError(
                            f"Gazebo GUI failed; see {backend.gui_log.name}"
                        )
                    return
                deadline = time.monotonic() + cfg.control_dt / args.speed
                actions = policy_action(
                    agent,
                    torch.as_tensor(obs, device=args.device)[None],
                    args.stochastic,
                )
                obs, _, terminated, truncated, info = env.step(actions[0].cpu().numpy())
                total_steps += 1
                if total_steps % 5 == 0 or terminated or truncated:
                    print(
                        f"distance={info['distance']:.2f}m heading_error={info['heading_error']:.2f}rad return={env.episode_return:.4f}",
                        flush=True,
                    )
                while time.monotonic() < deadline and backend.gui.poll() is None:
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                if args.max_steps is not None and total_steps >= args.max_steps:
                    return
                if terminated or truncated:
                    print("Episode: " + json.dumps(info["episode"]), flush=True)
                    break
            episode += 1
            if args.episodes and episode >= args.episodes:
                break
            obs, _ = env.reset(seed=args.seed + episode)
            backend.goal(env.target, env.target_heading)
    except KeyboardInterrupt:
        print("Playback stopped.", flush=True)
    finally:
        if env is not None:
            env.close()
        else:
            backend.close()


if __name__ == "__main__":
    main()
