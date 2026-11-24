"""Simulation adapters. Only this boundary consumes Gazebo's raw state."""

import math
from uuid import uuid4

from scipy.spatial.transform import Rotation

from .contracts import Detection, Health, InstrumentFeedback, Kind, Localization, Pose


class GazeboGroundTruth:
    def __init__(self, heading_offsets=None):
        self.states, self.validity = {}, {}
        self.heading_offsets = heading_offsets or {}

    def ingest(self, snapshot):
        for name, item in snapshot["robots"].items():
            offset = self.heading_offsets.get(name, 0.0)
            forward = Rotation.from_euler("xyz", item["rpy"]).apply(
                [math.cos(offset), math.sin(offset), 0]
            )
            yaw = math.atan2(forward[1], forward[0])
            vx, vy = item["linear_velocity"][:2]
            self.states[name] = Localization(
                Pose(*item["position"][:2], yaw),
                snapshot["sim_time"],
                "gazebo_ground_truth",
                self.validity.get(name, True),
                vx * math.cos(yaw) + vy * math.sin(yaw),
                item["angular_velocity"][2],
                lateral_speed=-vx * math.sin(yaw) + vy * math.cos(yaw),
            )

    def get(self, robot):
        return self.states[robot]


class GazeboHealth:
    def __init__(self, maps=None):
        self.states = {}
        self.maps = maps

    def ingest(self, snapshot):
        for name, item in snapshot["robots"].items():
            tipped = max(abs(x) for x in item["rpy"][:2]) > math.pi / 3
            contacts = set(item["contacts"])
            if "body_contacts" in item:
                contacts -= getattr(self.maps, "traversable_objects", set())
                contacts |= set(item["body_contacts"])
            self.states[name] = Health(
                not tipped, bool(contacts), "tipped" if tipped else ""
            )

    def get(self, robot):
        return self.states[robot]


class SimulatedDetector:
    """Range-gated scripted observations, no claim of image/point-cloud perception.

    No target is sent to the mission layer before an eligible robot comes within
    range. This placeholder does not model occlusion or sensor noise.
    """

    def __init__(self, targets, radius=4.0, *, maps=None, localization=None):
        self.targets, self.radius = targets, radius
        self.maps, self.localization = maps, localization

    def poll(self, robot, state, now):
        if Kind.ROCK_CANDIDATE not in robot.capabilities or not state.usable(now):
            return []
        observations = []
        for t in self.targets:
            target = (
                Pose(*self.maps.target_position(t["object"]))
                if "object" in t and self.maps is not None
                else Pose(*t["target"])
            )
            if state.pose.distance(target) > self.radius:
                continue
            if t.get("approach_selection") == "nearest_reachable":
                from .scene import nearest_approach

                if self.maps is None or self.localization is None:
                    raise ValueError(
                        "Scene detection requires map/localization adapters"
                    )
                try:
                    approach = Pose(
                        *nearest_approach(
                            self.maps,
                            target,
                            {robot.name: robot},
                            self.localization,
                            Kind.ROCK_CANDIDATE,
                            t["confirmation_radius"],
                            t.get("inspection_clearance", 0.6),
                        )
                    )
                except ValueError:
                    continue
            else:
                approach = Pose(*t["approach"])
            observations.append(
                Detection(
                    t["id"],
                    target,
                    approach,
                    now,
                    "simulated_range_detector",
                    measurement_approach=Pose(*t["measurement_approach"])
                    if t.get("measurement_approach") is not None
                    else None,
                    confirmation_radius=t.get("confirmation_radius", 3.0),
                    measurement_radius=t.get("measurement_radius"),
                )
            )
        return observations


class SimulatedInstrument:
    """Replace with an asynchronous inspection/scientific-instrument adapter."""

    def __init__(self, dwell_seconds=2.0, *, requires_heading=True):
        self.dwell_seconds, self.actions = dwell_seconds, {}
        self.requires_heading = requires_heading

    def start(self, robot, kind, target, now):
        handle = uuid4().hex
        self.actions[handle] = dict(
            robot=robot, kind=kind, target=target, stopped_since=None
        )
        return handle

    def tick(self, handle, state, now):
        action = self.actions[handle]
        if not state.usable(now):
            return InstrumentFeedback("failed", reason="localization_unavailable")
        if state.speed > 0.03 or abs(state.yaw_rate) > 0.04:
            action["stopped_since"] = None
            return InstrumentFeedback("running")
        if action["stopped_since"] is None:
            action["stopped_since"] = now
            action["station"] = state.pose
        if state.pose.distance(action["station"]) > 0.05:
            action["stopped_since"] = None
            return InstrumentFeedback("running")
        if now - action["stopped_since"] < self.dwell_seconds:
            return InstrumentFeedback("running")
        target = action["target"]
        result = dict(
            simulated=True,
            kind=action["kind"],
            robot=action["robot"],
            stamp=now,
            frame=state.pose.frame,
            target=[target.x, target.y, target.yaw],
            approach=[state.pose.x, state.pose.y, state.pose.yaw],
            source="simulated_inspection"
            if action["kind"] == Kind.ROCK_CANDIDATE
            else "simulated_measurement",
            settled_seconds=now - action["stopped_since"],
            position_drift=state.pose.distance(action["station"]),
            horizontal_speed=state.speed,
        )
        if action["kind"] == Kind.ROCK_CANDIDATE:
            result["confirmed_measurement"] = True
        self.actions.pop(handle)
        return InstrumentFeedback("succeeded", result)

    def cancel(self, handle):
        self.actions.pop(handle, None)
