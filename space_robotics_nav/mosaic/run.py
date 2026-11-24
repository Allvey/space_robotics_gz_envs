"""Run the minimal MOSAIC mission on actual Gazebo vehicles (no surrogate physics)."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
import subprocess
import sys
from collections import deque

import yaml

from ..config import ROOT
from .backend import MultiRobotBackend
from .contracts import Kind, Pose, Robot
from .maps import KnownMap, KnownMapCrop
from .scene import MeasurementStations, known_map, nearest_approach
from .visualization import MissionView
from .vehicles import SUPPORTED_MODELS, configured_vehicle
from .mission import Mission
from .exploration import LayoutExploration
from .navigation import AStarPlanner, GridNavigator, PathTracker
from .poi import POI, POIStore, Status
from .providers import (
    GazeboGroundTruth,
    GazeboHealth,
    SimulatedDetector,
    SimulatedInstrument,
)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    if "poi_planning" in config:
        from .poi_planning import PlanningSettings

        try:
            PlanningSettings(**config["poi_planning"])
        except TypeError as error:
            raise ValueError(f"Unknown POI planning setting: {error}") from error
    expected = dict(
        localization="gazebo_ground_truth",
        global_map="generated_collision_geometry"
        if config.get("scene_manifest")
        else "project_collision_geometry"
        if config.get("world") in {"moon", "mars"}
        else "known_layout",
        local_map="known_map_crop",
        detection="simulated_range_detector",
    )
    if config.get("providers") != expected:
        raise ValueError(
            "Unknown provider configuration; register an adapter instead of silently falling back"
        )
    if config.get("navigation") != dict(
        global_planner="grid_astar", local_controller="path_tracker"
    ):
        raise ValueError("Unknown navigation configuration")
    names = [r["name"] for r in config["robots"]]
    if not names or len(set(names)) != len(names):
        raise ValueError("Robot names must be unique")
    obstacles = config["obstacles"]
    if len({o["id"] for o in obstacles}) != len(obstacles):
        raise ValueError("Obstacle IDs must be unique")
    for r in config["robots"]:
        if r["model"] not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported mission vehicle: {r['model']}")
        spec = configured_vehicle(r, config)
        if r.get("radius", 1.7) < spec.minimum_radius:
            raise ValueError(
                f"{r['name']} footprint must cover its native collision radius {spec.minimum_radius:.3f} m"
            )
        Pose(*r["home"])
        if r.get("max_linear", 0.25) > 0.35 or r.get("max_angular", 0.45) > 0.7:
            raise ValueError(
                "Configured speeds exceed the source vehicle controller limits"
            )
    x0, y0, x1, y1 = config["map"]["bounds"]
    if config.get("scene_manifest"):
        manifest = json.loads(Path(config["scene_manifest"]).read_text())
        if (
            manifest["bounds"] != config["map"]["bounds"]
            or not Path(config["scene_file"]).is_file()
        ):
            raise ValueError("Generated scene files and map bounds must match")
    elif not (-19 <= x0 < x1 <= 19 and -19 <= y0 < y1 <= 19):
        raise ValueError("Map must fit inside the source 40m ground plane")
    for obstacle in obstacles:
        a, b, c, d = obstacle["bounds"]
        if (
            not all(math.isfinite(v) for v in (a, b, c, d))
            or a >= c
            or b >= d
            or obstacle.get("height", 1) <= 0
        ):
            raise ValueError(
                "Obstacles require nonempty finite bounds and positive height"
            )
    target_ids = [t["id"] for t in config["targets"]]
    if len(set(target_ids)) != len(target_ids):
        raise ValueError("Target IDs must be unique")
    return config


def create_mission(config, snapshot, *, emit=None, auto_approve=False, maps=None):
    localization = GazeboGroundTruth(
        {
            r["name"]: configured_vehicle(r, config).heading_offset
            for r in config["robots"]
        }
    )
    localization.ingest(snapshot)
    robots = {
        r["name"]: Robot(
            r["name"],
            frozenset(Kind(k) for k in r["capabilities"]),
            Pose(*r["home"]),
            r.get("radius", 1.7),
            r.get("max_linear", 0.25),
            r.get("max_angular", 0.45),
            {Kind(k): float(v) for k, v in r.get("preferences", {}).items()},
            r.get("turn_offset", 0.0),
            r.get("yaw_rate_damping", 0.0),
            r.get("position_tolerance", 0.32),
            r.get("allow_reverse", True),
        )
        for r in config["robots"]
    }
    maps = maps if maps is not None else known_map(config)
    if hasattr(maps, "ingest_objects"):
        maps.ingest_objects(snapshot.get("objects", {}))
    health = GazeboHealth(maps)
    health.ingest(snapshot)
    local_maps = KnownMapCrop(maps, localization, robots)
    targets = []
    for item in config["targets"]:
        target = dict(item)
        if "object" in item:
            target["target"] = [*maps.target_position(item["object"]), 0.0]
        if item.get("approach_selection") == "nearest_reachable":
            # Candidate stations are planned lazily at the observing scout's
            # current pose. Scene objects are not initial mission POIs.
            target["measurement_approach"] = None
        target["measurement_radius"] = item.get("measurement_max_distance", 3.5)
        targets.append(target)
        target.setdefault(
            "confirmation_radius", item.get("inspection_max_distance", 3.0)
        )
    detector = SimulatedDetector(
        targets,
        config.get("detection_radius", 4.0),
        maps=maps,
        localization=localization,
    )
    store = POIStore()
    for item in config["pois"]:
        target = Pose(*item.get("target", item["pose"]))
        approach = Pose(*item["pose"])
        store.add(
            POI(
                item["id"],
                Kind(item["kind"]),
                target,
                approach,
                assigned_to=item.get("assigned_to"),
            )
        )
    planner = AStarPlanner()
    navigators = {
        name: GridNavigator(
            r,
            maps,
            local_maps,
            planner,
            PathTracker(),
            tracking_margin=next(
                c.get("tracking_margin", 0.4)
                for c in config["robots"]
                if c["name"] == name
            ),
        )
        for name, r in robots.items()
    }
    mission = Mission(
        robots,
        store,
        localization,
        maps,
        local_maps,
        detector,
        health,
        navigators,
        planner,
        emit=emit,
        auto_approve=auto_approve,
        instrument=SimulatedInstrument(
            requires_heading=config.get("instrument_requires_heading", True)
        ),
        exploration_seconds=config.get("exploration_seconds", 10.0),
        exploration=LayoutExploration(
            planner,
            local_maps,
            localization,
            robots,
            spacing=config.get("exploration", {}).get("spacing", 6.0),
            clearance=config.get("exploration", {}).get("clearance", 0.6),
        )
        if config.get("exploration", {}).get("enabled", False)
        else None,
        exploration_queue=config.get("exploration", {}).get("queue_size", 2),
        measurement_stations=MeasurementStations(
            maps,
            localization,
            planner,
            clearance=config.get("measurement_clearance", 0.2),
            robots=robots if config.get("scene_manifest") else None,
        ),
        poi_planning=config.get("poi_planning"),
    )
    return mission, localization, health


class CommandFile:
    """Append-only JSONL commands; a partial final line waits for the next tick."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.offset, self.pending = 0, ""

    def read(self):
        if self.path is None or not self.path.exists():
            return []
        if self.path.stat().st_size < self.offset:
            raise ValueError("Command file was truncated; use an append-only file")
        with self.path.open() as file:
            file.seek(self.offset)
            self.pending += file.read()
            self.offset = file.tell()
        lines = self.pending.split("\n")
        self.pending = lines.pop()
        results = []
        for line in lines:
            if line.strip():
                try:
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise ValueError("Command must be a JSON object")
                    results.append(message)
                except (ValueError, TypeError) as error:
                    results.append({"op": "invalid", "error": str(error)})
        return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/mosaic/demo.yaml"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "logs/mosaic/demo")
    parser.add_argument(
        "--seconds",
        type=float,
        default=360,
        help="Simulation-time budget after initial settling",
    )
    parser.add_argument("--gui", action="store_true")
    parser.add_argument(
        "--observe-seconds",
        type=float,
        default=0,
        help="Keep stepping a headless mission after completion to observe parking",
    )
    parser.add_argument(
        "--control",
        action="store_true",
        help="Open the mission control panel without the Gazebo 3D window",
    )
    parser.add_argument(
        "--no-control", action="store_true", help="Show only Gazebo when using --gui"
    )
    parser.add_argument("--software", action="store_true")
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="Pace the CLI loop for live operator input",
    )
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Legacy option; close-range robot confirmation now releases science tasks automatically",
    )
    parser.add_argument(
        "--commands", type=Path, help="Append-only operator-command JSONL file"
    )
    parser.add_argument(
        "--script",
        type=Path,
        help="YAML list of commands with 'at' in seconds from mission start",
    )
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not math.isfinite(args.observe_seconds) or args.observe_seconds < 0:
        parser.error("--observe-seconds must be finite and nonnegative")
    config = load_config(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    for artifact in ("events.jsonl", "telemetry.jsonl", "summary.json", "run.json"):
        if (args.output / artifact).exists():
            parser.error("Use a fresh output directory to preserve previous runs")
    schedule = (
        sorted(yaml.safe_load(args.script.read_text()) or [], key=lambda x: x["at"])
        if args.script
        else []
    )
    commands = CommandFile(args.commands)
    control_commands = CommandFile(args.output / "control_commands.jsonl")
    interactive = args.control or (args.gui and not args.no_control)
    recent_events = deque(maxlen=80)
    run_info = dict(
        config=config,
        auto_approve=args.auto_approve,
        scheduled_commands=schedule,
        perception_is_simulated=True,
        observe_seconds=args.observe_seconds,
        seconds=args.seconds,
    )
    (args.output / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
    backend, mission, view, control_process, control_log = None, None, None, None, None
    exit_reason, error_text = "budget_exhausted", None
    with (args.output / "events.jsonl").open("w", buffering=1) as events, (
        args.output / "telemetry.jsonl"
    ).open("w", buffering=1) as telemetry:

        def emit(event):
            recent_events.append(event)
            events.write(json.dumps(event, allow_nan=False) + "\n")
            print(json.dumps(event, ensure_ascii=False), flush=True)

        try:
            backend = MultiRobotBackend(config, gui=args.gui, software=args.software)
            (args.output / "runtime_scene.sdf").write_bytes(backend.path.read_bytes())
            (args.output / "scene.json").write_text(
                json.dumps(
                    dict(
                        world=config.get("world", "mining_vehicles"),
                        source=str(
                            config.get(
                                "scene_file",
                                ROOT
                                / "worlds"
                                / f"{config.get('world','mining_vehicles')}.sdf",
                            )
                        ),
                        container=backend.name,
                        physics_dt=backend.physics_dt,
                        vehicles=[
                            dict(
                                name=r["name"],
                                model=r["model"],
                                path=str(configured_vehicle(r, config).path),
                                runtime_model=backend.vehicle_models[r["name"]],
                                body=configured_vehicle(r, config).body,
                                heading_offset=configured_vehicle(
                                    r, config
                                ).heading_offset,
                                radius=r["radius"],
                                drive_wheel_separation=configured_vehicle(
                                    r, config
                                ).wheel_separation,
                                wheel_slip=configured_vehicle(r, config).wheel_slip,
                            )
                            for r in config["robots"]
                        ],
                    ),
                    indent=2,
                )
                + "\n"
            )
            mission, localization, health = create_mission(
                config,
                backend.state,
                emit=emit,
                auto_approve=args.auto_approve,
                maps=backend.maps,
            )
            view = MissionView(mission, args.output, config)
            mission.now = backend.state["sim_time"]
            view.snapshot(recent_events)
            if interactive:
                control_log = (args.output / "control.log").open("w")
                control_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "space_robotics_nav.mosaic.control",
                        str(args.output.resolve()),
                    ],
                    stdout=control_log,
                    stderr=control_log,
                )
            started = backend.state["sim_time"]
            completed_at = None
            emit(
                dict(
                    event="mission_started",
                    stamp=started,
                    robots=list(mission.robots),
                    providers=config["providers"],
                )
            )
            while backend.state["sim_time"] - started < args.seconds:
                wall_started = time.monotonic()
                now = backend.state["sim_time"]
                localization.ingest(backend.state)
                if hasattr(mission.maps, "ingest_objects"):
                    mission.maps.ingest_objects(backend.state.get("objects", {}))
                health.ingest(backend.state)
                if control_process is not None and control_process.poll() is not None:
                    if control_process.returncode:
                        raise RuntimeError(
                            f"Mission control exited unexpectedly; see {args.output / 'control.log'}"
                        )
                    exit_reason = "interrupted"
                    break
                messages = commands.read() + control_commands.read()
                while schedule and schedule[0]["at"] <= now - started:
                    message = dict(schedule.pop(0))
                    message.pop("at")
                    messages.append(message)
                for message in messages:
                    try:
                        if message.get("op") == "stop":
                            exit_reason = "interrupted"
                            break
                        elif message.get("op") == "map_block":
                            mission.maps.block(
                                message["bounds"], message.get("blocked", True)
                            )
                            emit(
                                dict(
                                    event="adapter_map_update",
                                    stamp=now,
                                    command=message,
                                )
                            )
                        elif message.get("op") == "localization_valid":
                            name = message["robot"]
                            if name not in mission.robots or not isinstance(
                                message["valid"], bool
                            ):
                                raise ValueError(
                                    "Expected an existing robot and boolean valid"
                                )
                            localization.validity[name] = message["valid"]
                            localization.ingest(backend.state)
                            emit(
                                dict(
                                    event="adapter_fault_injection",
                                    stamp=now,
                                    command=message,
                                )
                            )
                        else:
                            mission.command(message, now)
                    except (ValueError, KeyError, TypeError) as error:
                        emit(
                            dict(
                                event="command_rejected",
                                stamp=now,
                                command=message,
                                reason=str(error),
                            )
                        )
                if exit_reason == "interrupted":
                    break
                decision_started = time.monotonic()
                actions = mission.tick(now)
                decision_elapsed = time.monotonic() - decision_started
                complete = mission.is_complete()
                visual = view.snapshot(
                    recent_events, reason="completed" if complete else "running"
                )
                telemetry.write(
                    json.dumps(
                        dict(
                            stamp=now,
                            poses={
                                name: asdict(localization.get(name))
                                for name in mission.robots
                            },
                            velocities=actions,
                            active=mission.summary()["active"],
                            phases={
                                **{
                                    name: task.phase
                                    for name, task in mission.executions.items()
                                },
                                **{name: "yield" for name in mission.yielding},
                            },
                            map_version=mission.maps.snapshot(now).version,
                            decision_wall_seconds=decision_elapsed,
                            plan_version=mission.plan_version,
                            future_owners=visual["poi_planning"]["future_owners"],
                            planned_sequences={
                                r["name"]: [s["poi_id"] for s in r["poi_plan"]]
                                for r in visual["robots"]
                            },
                            exploration=mission.exploration_status(),
                            truth_for_evaluation=backend.state,
                        ),
                        allow_nan=False,
                    )
                    + "\n"
                )
                backend.step(
                    actions, markers=view.markers(visual) if args.gui else None
                )
                if mission.is_complete():
                    exit_reason = "completed"
                    if completed_at is None:
                        completed_at = backend.state["sim_time"]
                        emit(dict(event="all_tasks_completed", stamp=now))
                    if (
                        not (interactive or args.gui)
                        and backend.state["sim_time"] - completed_at
                        >= args.observe_seconds
                    ):
                        break
                else:
                    exit_reason = "budget_exhausted"
                    completed_at = None
                if args.realtime or interactive or args.gui:
                    time.sleep(
                        max(0, backend.control_dt - (time.monotonic() - wall_started))
                    )
        except KeyboardInterrupt:
            exit_reason = "interrupted"
        except Exception as error:
            exit_reason, error_text = "error", str(error)
            raise
        finally:
            if mission is not None:
                mission.close()
            if view is not None:
                view.snapshot(recent_events, finished=True, reason=exit_reason)
            if backend is not None:
                try:
                    backend.step({name: [0.0, 0.0] for name in backend.descriptors})
                except Exception as stop_error:
                    emit(dict(event="shutdown_stop_failed", reason=str(stop_error)))
                finally:
                    backend.close()
            if control_process is not None and control_process.poll() is None:
                control_process.terminate()
                try:
                    control_process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    control_process.kill()
                    control_process.wait()
            if control_log is not None:
                control_log.close()
            summary = mission.summary() if mission else {}
            summary.update(exit_reason=exit_reason, error=error_text)
            (args.output / "summary.json").write_text(
                json.dumps(summary, indent=2) + "\n"
            )
    print(f"MOSAIC {exit_reason}: {args.output / 'summary.json'}", flush=True)
    return 0 if exit_reason == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
