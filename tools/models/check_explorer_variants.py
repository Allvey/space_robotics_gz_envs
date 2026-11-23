"""Finite Gazebo Ionic physics smoke test; no rendering or training.

Use the image's /usr/bin/python3 (gz.sim9). Results print as JSON.
"""

import json
import math
import os
from pathlib import Path
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET

# Import math first to register pybind Pose3d return types.
from gz.math8 import Pose3d  # noqa: F401
from gz.msgs11.double_pb2 import Double
from gz.msgs11.twist_pb2 import Twist
from gz.sim9 import Joint, Link, Model, TestFixture, World, world_entity
from gz.transport14 import Node

ROOT = Path(__file__).resolve().parents[2]
os.environ["GZ_PARTITION"] = "mining-check-" + uuid.uuid4().hex
os.environ["GZ_SIM_RESOURCE_PATH"] = str(ROOT / "assets/models")
NAMES = ["lunar_excavator", "lunar_hauler"]
TARGETS = {
    "lunar_excavator": {
        "shoulder_yaw": 0.5,
        "shoulder_pitch": -0.2,
        "elbow_pitch": -0.3,
        "bucket_pitch": 0.6,
    },
    "lunar_hauler": {"bed_tilt": -0.6, "tailgate_hinge": 1.2},
}


def main():
    root = ET.parse(ROOT / "worlds/mining_vehicles.sdf").getroot()
    world = root.find("world")
    for p in world.findall("plugin"):
        if p.get("name").endswith("Sensors"):
            world.remove(p)
    for el in world.findall("include/uri"):
        el.text = str(
            ROOT / "assets/models" / el.text.removeprefix("model://") / "model.sdf"
        )
    handles, state, maximum_tilt = {}, {}, {name: 0.0 for name in NAMES}

    def pre(_info, ecm):
        if handles:
            return
        w = World(world_entity(ecm))
        for name in NAMES:
            m = Model(w.model_by_name(ecm, name))
            body = Link(m.link_by_name(ecm, "base_link"))
            joints = {key: Joint(m.joint_by_name(ecm, key)) for key in TARGETS[name]}
            for j in joints.values():
                j.enable_position_check(ecm)
            handles[name] = (body, joints)

    def post(_info, ecm):
        for name, (body, joints) in handles.items():
            p = body.world_pose(ecm)
            angles = [p.rot().roll(), p.rot().pitch(), p.rot().yaw()]
            maximum_tilt[name] = max(maximum_tilt[name], abs(angles[0]), abs(angles[1]))
            state[name] = dict(
                position=[p.pos().x(), p.pos().y(), p.pos().z()],
                rpy=angles,
                joints={
                    key: j.position(ecm)[0]
                    for key, j in joints.items()
                    if j.position(ecm)
                },
            )

    with tempfile.TemporaryDirectory(prefix="mining-check-") as temp:
        path = Path(temp) / "world.sdf"
        ET.ElementTree(root).write(path)
        fixture = TestFixture(str(path))
        fixture.on_pre_update(pre)
        fixture.on_post_update(post)
        fixture.finalize()
        server = fixture.server()

        def run(steps):
            assert server.run(True, steps, False)
            for s in state.values():
                assert all(
                    math.isfinite(x)
                    for x in s["position"] + s["rpy"] + list(s["joints"].values())
                )
                assert 0.30 < s["position"][2] < 0.50, s
                assert max(abs(x) for x in s["rpy"][:2]) < 0.25, s

        run(800)
        settled = json.loads(json.dumps(state))
        node = Node()
        drives = {
            name: node.advertise(f"/model/{name}/cmd_vel", Twist) for name in NAMES
        }
        controls = {
            (name, key): node.advertise(f"/model/{name}/joint/{key}/0/cmd_pos", Double)
            for name in NAMES
            for key in TARGETS[name]
        }
        deadline = time.monotonic() + 5
        while not all(
            p.has_connections() for p in [*drives.values(), *controls.values()]
        ):
            if time.monotonic() > deadline:
                raise RuntimeError("Controller transport discovery timed out")
            time.sleep(0.05)
        twist = Twist()
        twist.linear.x = 0.2
        for pub in drives.values():
            assert pub.publish(twist)
        time.sleep(0.05)
        run(600)
        distance = {
            name: state[name]["position"][0] - settled[name]["position"][0]
            for name in NAMES
        }
        assert all(0.35 < d < 0.80 for d in distance.values()), distance
        twist.linear.x = 0
        for pub in drives.values():
            assert pub.publish(twist)
        time.sleep(0.05)
        run(200)
        yaw_before = {name: state[name]["rpy"][2] for name in NAMES}
        twist.angular.z = 0.25
        for pub in drives.values():
            assert pub.publish(twist)
        time.sleep(0.05)
        run(400)
        turn = {name: state[name]["rpy"][2] - yaw_before[name] for name in NAMES}
        assert all(0.15 < angle < 0.7 for angle in turn.values()), turn
        twist.angular.z = 0
        for pub in drives.values():
            assert pub.publish(twist)
        time.sleep(0.05)
        run(200)
        for (name, key), pub in controls.items():
            assert pub.publish(Double(data=TARGETS[name][key]))
        time.sleep(0.05)
        run(1800)
        errors = {
            name: {
                key: abs(state[name]["joints"][key] - target)
                for key, target in TARGETS[name].items()
            }
            for name in NAMES
        }
        assert all(
            error < 0.06 for values in errors.values() for error in values.values()
        ), errors
        # Return all payload joints home and verify they do not drift/run away.
        reached = json.loads(json.dumps(state))
        for pub in controls.values():
            assert pub.publish(Double(data=0))
        time.sleep(0.05)
        run(1800)
        assert all(
            abs(q) < 0.06 for s in state.values() for q in s["joints"].values()
        ), state
        result = dict(
            passed=True,
            simulated_seconds=29,
            settled=settled,
            travel_m=distance,
            turn_rad=turn,
            reached=reached,
            target_error_rad=errors,
            returned_home=state,
            maximum_tilt_rad=maximum_tilt,
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
