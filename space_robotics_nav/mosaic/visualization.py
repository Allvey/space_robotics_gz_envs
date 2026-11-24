"""Mission-control read model and optional Gazebo geometry markers.

The control panel consumes snapshots and sends commands through an adapter. It
does not query simulation truth or mutate POIs directly.
"""

from collections import deque
from dataclasses import asdict
import json
import colorsys
from pathlib import Path

import cv2
import numpy as np


ROBOT_COLORS = ("#28c6ec", "#f5a742", "#c49bff", "#f179ad")
STATUS_COLORS = {
    "available": "#f0cc60",
    "claimed": "#40b9ed",
    "running": "#40b9ed",
    "waiting_approval": "#c995ff",
    "succeeded": "#648077",
    "rejected": "#687482",
    "needs_help": "#f16f6f",
}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(path)


class MissionView:
    def __init__(self, mission, output, config):
        self.mission, self.output, self.config = mission, Path(output), config
        self.version = None
        self.trails = {name: deque(maxlen=1500) for name in mission.robots}
        self.marker_ids = {}
        self.previous_ids = set()

    def _background(self, grid):
        maps = self.mission.maps
        heights = getattr(maps, "heights", np.zeros(grid.blocked.shape))
        finite = np.isfinite(heights)
        clean = np.where(finite, heights, 0)
        low, high = np.percentile(clean[finite], [2, 98]) if finite.any() else (0, 1)
        scale = np.clip((clean - low) / max(0.1, high - low), 0, 1)
        dy, dx = np.gradient(clean, grid.resolution)
        shade = np.clip(
            (1 - 0.6 * dx + 0.5 * dy) / np.sqrt(1 + dx * dx + dy * dy), 0.25, 1.2
        )
        intensity = np.clip((70 + 110 * scale) * shade, 25, 220)
        pixels = np.repeat(intensity[:, :, None], 3, axis=2)
        pixels[grid.blocked] = (
            pixels[grid.blocked] * 0.5 + np.array([120, 75, 55]) * 0.5
        )
        image = np.flipud(pixels.astype(np.uint8))
        # Fixed-size terrain raster; canvas overlays remain in map coordinates.
        size = (760, round(760 * image.shape[0] / image.shape[1]))
        image = cv2.resize(image, size, interpolation=cv2.INTER_NEAREST)
        temporary = self.output / "map.tmp.png"
        cv2.imwrite(str(temporary), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        temporary.replace(self.output / "map.png")

    def snapshot(self, events=(), *, finished=False, reason="running"):
        m, now = self.mission, self.mission.now
        grid = m.maps.snapshot(now)
        if grid.version != self.version:
            self._background(grid)
            self.version = grid.version
        robots = []
        descriptions = {r["name"]: r for r in self.config["robots"]}
        for index, (name, robot) in enumerate(m.robots.items()):
            state = m.localization.get(name)
            xy = (state.pose.x, state.pose.y)
            if (
                not self.trails[name]
                or np.linalg.norm(np.asarray(xy) - self.trails[name][-1]) > 0.08
            ):
                self.trails[name].append(xy)
            task, nav = m.executions.get(name), m.navigators[name]
            path = getattr(nav, "path", None) or []
            path = path[getattr(nav, "index", 0) :]
            local = m.local_maps.snapshot(robot, state.pose, now)
            yy, xx = np.nonzero(local.blocked)
            stride = max(1, len(xx) // 1800)
            local_cells = np.c_[
                local.origin[0] + xx[::stride] * local.resolution,
                local.origin[1] + yy[::stride] * local.resolution,
            ].tolist()
            phase = (
                task.phase
                if task
                else (
                    "yield"
                    if name in m.yielding
                    else ("drifting" if state.speed > 0.03 else "idle")
                )
            )
            robots.append(
                dict(
                    name=name,
                    label=descriptions[name].get("label", name),
                    role=descriptions[name].get("role", ""),
                    model=descriptions[name]["model"],
                    color=(
                        ROBOT_COLORS[index]
                        if index < len(ROBOT_COLORS)
                        else "#"
                        + "".join(
                            f"{round(c*255):02x}"
                            for c in colorsys.hsv_to_rgb(
                                (index * 0.61803398875) % 1, 0.65, 0.95
                            )
                        )
                    ),
                    pose=asdict(state.pose),
                    valid=state.usable(now),
                    mode=m.modes[name],
                    phase=phase,
                    poi=task.poi_id if task else None,
                    radius=robot.radius,
                    capabilities=sorted(robot.capabilities),
                    velocity=[state.linear_speed, state.yaw_rate],
                    lateral_speed=state.lateral_speed,
                    speed=state.speed,
                    waiting_reason=m.robot_wait_reason(robot),
                    analysis_remaining=max(
                        0, m.exploration_seconds - (now - task.analysis_started)
                    )
                    if task
                    and task.phase == "analyze"
                    and task.analysis_started is not None
                    else None,
                    path=[[p.x, p.y] for p in path],
                    trail=list(self.trails[name]),
                    local_map=dict(
                        source=local.source,
                        valid=local.usable(now),
                        blocked=local_cells,
                        origin=local.origin,
                        shape=local.blocked.shape,
                        resolution=local.resolution,
                    ),
                    local_replans=getattr(nav, "local_replans", 0),
                    navigation_wait_reason=getattr(nav, "waiting_reason", ""),
                    poi_plan=[s.record() for s in m.poi_plans[name]],
                )
            )
        value = dict(
            stamp=now,
            finished=finished,
            reason=reason,
            world=self.config.get("world", "mining_vehicles"),
            providers=self.config["providers"],
            candidate_confirmation="robot_proximity",
            exploration=m.exploration_status(),
            poi_planning=m.planning_status(),
            map=dict(
                origin=grid.origin,
                shape=grid.blocked.shape,
                resolution=grid.resolution,
                version=grid.version,
                source=grid.source,
                image="map.png",
            ),
            robots=robots,
            pois=[
                dict(
                    p,
                    wait_reason=m.poi_wait_reason(m.store.items[p["id"]]),
                    navigation_station=asdict(m.store.items[p["id"]].navigation_station)
                    if m.store.items[p["id"]].navigation_station is not None
                    else None,
                )
                for p in m.store.records()
            ],
            events=list(events)[-60:],
        )
        atomic_json(self.output / "mission_state.json", value)
        return value

    def markers(self, snapshot):
        result, ids = [], set()

        def marker(key, color, **fields):
            identifier = self.marker_ids.setdefault(key, len(self.marker_ids) + 1)
            ids.add(identifier)
            rgb = [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
            result.append(
                dict(id=identifier, color=[*rgb, fields.pop("alpha", 1.0)], **fields)
            )

        def height(x, y):
            return self.mission.maps.height(x, y) + 0.35

        for p in snapshot["pois"]:
            t, a = p["target"], p.get("navigation_station", p["approach"])
            color = STATUS_COLORS[p["status"]]
            alpha = 0.3 if p["status"] in {"succeeded", "rejected"} else 1.0
            marker(
                "poi:" + p["id"],
                color,
                position=[t["x"], t["y"], height(t["x"], t["y"]) + 0.4],
                scale=0.45,
                alpha=alpha,
            )
            if a is not None:
                marker(
                    "approach:" + p["id"],
                    color,
                    position=[a["x"], a["y"], height(a["x"], a["y"])],
                    scale=0.22,
                    alpha=alpha,
                )
        for robot in snapshot["robots"]:
            if robot["path"]:
                pose = robot["pose"]
                points = [[pose["x"], pose["y"], height(pose["x"], pose["y"])]]
                points += [[x, y, height(x, y)] for x, y in robot["path"]]
                marker(
                    "path:" + robot["name"],
                    robot["color"],
                    type="line",
                    points=points,
                    scale=1.0,
                )
        for identifier in self.previous_ids - ids:
            result.append(dict(id=identifier, delete=True, color=[0, 0, 0, 0]))
        self.previous_ids = ids
        return result
