"""Watch a policy in the original Gazebo world, using the original rover."""

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from .backend import GazeboBackend
from .config import NavigationConfig
from .evaluate import read_run
from .learning import create_agent, policy_action
from .planning.terminal import (
    TerminalController,
    add_terminal_arguments,
    terminal_config,
)
from .playback import PlaybackEnv, advance_goal, follow_camera, playback_seed
from .world import gui_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint", type=Path)
    p.add_argument(
        "--seed", type=int, help="Fixed seed for replay; omitted = fresh random run"
    )
    p.add_argument(
        "--episodes",
        "--goals",
        type=int,
        default=10,
        help="Number of navigation goals (default 10); 0 loops until closed",
    )
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--software-rendering", action="store_true")
    p.add_argument("--max-steps", type=int)
    p.add_argument(
        "--reset-between-goals",
        action="store_true",
        help="Reset the rover between goals instead of continuing at its actual pose",
    )
    p.add_argument(
        "--output", type=Path, help="Save completed goals and run seed as JSON"
    )
    p.add_argument(
        "--fixed-camera",
        action="store_true",
        help="Keep the camera fixed instead of following the rover",
    )
    add_terminal_arguments(p)
    args = p.parse_args()
    try:
        terminal_cfg = terminal_config(args)
        run_seed = playback_seed(args.seed)
    except ValueError as exc:
        p.error(str(exc))
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
    results = []
    try:
        env = PlaybackEnv(cfg, backend=backend)
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
        obs, _ = env.reset(seed=run_seed)
        controller = TerminalController(env, terminal_cfg) if terminal_cfg else None
        backend.goal(env.target, env.target_heading)
        path = gui_config(
            cfg,
            backend.path.with_suffix(".gui.config"),
            env.origin,
            env.target,
            follow=not args.fixed_camera,
        )
        backend.start_gui(path)
        print(
            f"Loaded {args.checkpoint}\nNative world: {cfg.world}\nRun seed: {run_seed}\nGUI log: {backend.gui_log.name}",
            flush=True,
        )
        print(
            "Green beacon = goal; orange bar = target heading. Close window or Ctrl+C to exit.",
            flush=True,
        )
        episode, total_steps = 0, 0
        targets = [env.target.copy()]
        transition = "initial_reset"
        camera_following = args.fixed_camera
        camera_attempts, next_camera_attempt = 0, time.monotonic()
        while not args.episodes or episode < args.episodes:
            seed = run_seed + episode
            generator = torch.Generator(device=args.device).manual_seed(12345 + seed)
            goal = env.target.copy().tolist()
            heading = float(env.target_heading)
            start_pose = list(env.state["position"])
            print(
                f"Goal {episode+1}/{args.episodes or 'unlimited'}, seed={seed}, goal={goal}, heading={heading:.3f}, start={transition}",
                flush=True,
            )
            while True:
                if backend.gui.poll() is not None:
                    if backend.gui.returncode:
                        raise RuntimeError(
                            f"Gazebo GUI failed; see {backend.gui_log.name}"
                        )
                    return
                if (
                    not camera_following
                    and camera_attempts < 10
                    and time.monotonic() >= next_camera_attempt
                ):
                    camera_following = follow_camera(backend)
                    camera_attempts += 1
                    next_camera_attempt = time.monotonic() + 2
                    if camera_following:
                        print("Camera following the rover.", flush=True)
                deadline = time.monotonic() + cfg.control_dt / args.speed
                actions = policy_action(
                    agent,
                    torch.as_tensor(obs, device=args.device)[None],
                    args.stochastic,
                    generator=generator,
                )
                action = actions[0].cpu().numpy()
                if controller is not None:
                    action = controller.action(action)
                obs, _, terminated, truncated, info = env.step(action)
                total_steps += 1
                if total_steps % 5 == 0 or terminated or truncated:
                    print(
                        f"distance={info['distance']:.2f}m heading_error={info['heading_error']:.2f}rad return={env.episode_return:.4f} mode={controller.reason if controller else 'rl'}",
                        flush=True,
                    )
                while time.monotonic() < deadline and backend.gui.poll() is None:
                    time.sleep(min(0.1, max(0, deadline - time.monotonic())))
                if terminated or truncated:
                    print("Episode: " + json.dumps(info["episode"]), flush=True)
                    results.append(
                        {
                            "index": episode + 1,
                            "seed": seed,
                            "target": goal,
                            "target_heading": heading,
                            "start_position": start_pose,
                            "end_position": list(env.state["position"]),
                            "transition": transition,
                            **info["episode"],
                            "heading_error": info["heading_error"],
                        }
                    )
                    if controller:
                        print(
                            "Terminal: " + json.dumps(controller.report()), flush=True
                        )
                if args.max_steps is not None and total_steps >= args.max_steps:
                    return
                if terminated or truncated:
                    break
            episode += 1
            if args.episodes and episode >= args.episodes:
                break
            obs, transition = advance_goal(
                env,
                info,
                seed=run_seed + episode,
                previous_targets=targets,
                reset_between=args.reset_between_goals,
            )
            targets.append(env.target.copy())
            if controller:
                controller.reset()
            backend.goal(env.target, env.target_heading)
    except KeyboardInterrupt:
        print("Playback stopped.", flush=True)
    finally:
        if env is not None:
            env.close()
        else:
            backend.close()
        print(
            f"Finished {len(results)} goals; successes={sum(r['success'] for r in results)}",
            flush=True,
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.checkpoint),
                        "run_seed": run_seed,
                        "requested_goals": args.episodes,
                        "continuous": not args.reset_between_goals,
                        "terminal_planner": args.terminal_planner,
                        "goals": results,
                    },
                    indent=2,
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
