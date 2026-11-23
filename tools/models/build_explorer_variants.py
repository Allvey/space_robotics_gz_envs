"""Assemble mining variants from the unmodified Explorer R2 and payload assets.

Run with the project's Python; render_explorer_variants.py builds the adapter
meshes and renders the resulting SDF assemblies using Blender 4.3.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "assets/models"
SOURCE = (
    ROOT
    / "assets/cache/fuel/fuel.gazebosim.org/openrobotics/models/explorer_r2_sensor_config_1/17"
)
SCALE = 0.7


def child(parent, tag, text=None, **attrs):
    node = ET.SubElement(parent, tag, attrs)
    if text is not None:
        node.text = str(text)
    return node


def write_xml(root, path):
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def pose(node, value, relative_to=None):
    p = node.find("pose")
    if p is None:
        p = child(node, "pose")
    p.text = value
    p.attrib.clear()
    if relative_to:
        p.set("relative_to", relative_to)


def box_collision(link, name, xyz, size):
    c = child(link, "collision", name=name)
    pose(c, f"{xyz} 0 0 0")
    child(child(child(c, "geometry"), "box"), "size", size)


def adapter(model, kind):
    link = child(model, "link", name="payload_frame")
    # Rear rocker is rotated +90 deg about X in the source SDF. Undo that
    # rotation locally, so the payload's Z axis points up at the neutral pose.
    pose(link, "-0.382375 0.3554 0 -1.5707963267948966 0 0", "Rear_Rocker_Link")
    i = child(link, "inertial")
    child(i, "mass", 8)
    inertia = child(i, "inertia")
    # Conservative box approximation of the complete mount, in kg m^2.
    for name, val in dict(ixx=0.242, iyy=0.397, izz=0.626, ixy=0, ixz=0, iyz=0).items():
        child(inertia, name, val)
    v = child(link, "visual", name="adapter_visual")
    pose(v, "0 0 0 1.5707963267948966 0 0")
    child(child(child(v, "geometry"), "mesh"), "uri", "meshes/payload_frame.glb")
    for y in [-0.24, 0.24]:
        box_collision(link, f"rail_{y}", f"0 {y} 0", ".76 .055 .065")
    for x in [-0.29, 0, 0.29]:
        box_collision(link, f"crossmember_{x}", f"{x} 0 .01", ".055 .54 .045")
    if kind == "excavator":
        box_collision(link, "mount_plate", "0 0 .045", ".43 .43 .035")
    else:
        for y in [-0.21, 0.21]:
            box_collision(link, f"bed_support_{y}", f"0 {y} .055", ".80 .045 .08")
    camera_y = -0.47 if kind == "hauler" else 0
    if kind == "hauler":
        box_collision(link, "camera_outrigger", ".38 -.355 .045", ".05 .25 .04")
    box_collision(link, "work_camera_mast", f".38 {camera_y} .25", ".025 .03 .42")
    box_collision(link, "work_camera_housing", f".35 {camera_y} .46", ".075 .16 .055")
    # Move the existing rear RGB-D sensor to a physical mast above the bin.
    base = model.find("link[@name='base_link']")
    camera = base.find("sensor[@name='rs_back']")
    base.remove(camera)
    pose(camera, f".295 {camera_y} .46 0 .20 3.141592653589793")
    link.append(camera)
    j = child(model, "joint", name="payload_frame_fixed", type="fixed")
    child(j, "parent", "Rear_Rocker_Link")
    child(j, "child", "payload_frame")
    return link


def scaled_payload(kind):
    payload = ET.parse(Path(__file__).with_name("payloads.sdf")).find(
        f"model[@name='{kind}']"
    )
    for link in payload.findall("link"):
        for p in link.findall(".//pose"):
            values = [float(x) for x in p.text.split()]
            p.text = " ".join(
                f"{x * SCALE if k < 3 else x:.12g}" for k, x in enumerate(values)
            )
        for el in link.findall(".//collision/geometry/box/size"):
            el.text = " ".join(f"{float(x) * SCALE:.12g}" for x in el.text.split())
        for tag in ["radius", "length"]:
            for el in link.findall(f".//collision/geometry/cylinder/{tag}"):
                el.text = str(float(el.text) * SCALE)
        mass = link.find("inertial/mass")
        mass.text = str(float(mass.text) * SCALE**3)
        for el in link.findall("inertial/inertia/*"):
            el.text = str(float(el.text) * SCALE**5)
        child(link.find("visual/geometry/mesh"), "scale", f"{SCALE} {SCALE} {SCALE}")
    if kind == "excavator":
        pose(
            payload.find("link[@name='turret']"),
            "0 0 .13 0 0 3.141592653589793",
            "payload_frame",
        )
        payload.find("joint[@name='shoulder_yaw']/parent").text = "payload_frame"
        # Default arm faces away from the original sensor tower; usable rear
        # half-plane only, to avoid sweeping through it during loading.
        payload.find("joint[@name='shoulder_yaw']/axis/limit/lower").text = "-1.57"
        payload.find("joint[@name='shoulder_yaw']/axis/limit/upper").text = "1.57"
    else:
        pose(
            payload.find("link[@name='dump_bed']"), "-.45 0 .09 0 0 0", "payload_frame"
        )
        payload.find("joint[@name='bed_tilt']/parent").text = "payload_frame"
    # Rebuild controllers for the smaller payload; derivative gain stabilizes
    # the light links. Integral is bounded, removing gravity steady-state error.
    for old in payload.findall("plugin"):
        payload.remove(old)
    for j in payload.findall("joint"):
        p = child(
            payload,
            "plugin",
            filename="gz-sim-joint-position-controller-system",
            name="gz::sim::systems::JointPositionController",
        )
        gains = {
            "bucket_pitch": (25, 1.5),
            "tailgate_hinge": (20, 0.5),
            "bed_tilt": (150, 10),
        }.get(j.get("name"), (80, 8))
        for tag, val in dict(
            joint_name=j.get("name"),
            p_gain=gains[0],
            i_gain=8,
            d_gain=gains[1],
            i_min=-20,
            i_max=20,
            cmd_min=-100,
            cmd_max=100,
        ).items():
            child(p, tag, val)
    return payload


def main():
    common = MODELS / "explorer_r2_base"
    common.mkdir(parents=True, exist_ok=True)
    for directory in ["meshes", "materials"]:
        shutil.copytree(SOURCE / directory, common / directory, dirs_exist_ok=True)
    shutil.copy2(SOURCE / "model.sdf", common / "model.sdf")
    shutil.copy2(SOURCE / "model.config", common / "model.config")
    source_hashes = {
        str(p.relative_to(SOURCE)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in SOURCE.rglob("*")
        if p.is_file() and "thumbnails" not in p.parts
    }
    (common / "source_manifest.json").write_text(
        json.dumps(
            {
                "source": "https://fuel.gazebosim.org/1.0/OpenRobotics/models/EXPLORER_R2_SENSOR_CONFIG_1",
                "version": 17,
                "sha256": source_hashes,
            },
            indent=2,
        )
        + "\n"
    )
    for kind in ["excavator", "hauler"]:
        name = f"lunar_{kind}"
        dest = MODELS / name
        root = ET.parse(SOURCE / "model.sdf").getroot()
        root.set("version", "1.9")
        model = root.find("model")
        model.set("name", name)
        model.set("canonical_link", "base_link")
        # Include pose determines ground clearance. Model geometry is untouched.
        pose(model, "0 0 .43 0 0 0")
        for uri in model.findall(".//uri"):
            if uri.text.startswith(("meshes/", "materials/")):
                uri.text = "model://explorer_r2_base/" + uri.text
        for tag in ["albedo_map", "metalness_map", "roughness_map"]:
            for el in model.findall(f".//{tag}"):
                el.text = "model://explorer_r2_base/" + el.text
        # Convert deprecated SDF axis syntax to its modern, equivalent form.
        for axis in model.findall("joint/axis"):
            old = axis.find("use_parent_model_frame")
            if old is not None:
                axis.remove(old)
        for p in model.findall(".//pose"):
            p.attrib.pop("frame", None)
        for light in model.findall(".//light"):
            child(light, "visualize", "false")
            child(light, "light_on", "false")
            child(light, "intensity", "0")
        # These two upstream fields were placed inside friction/ode, where
        # libsdformat ignores them. Remove the invalid fields, not the contacts.
        for ode in model.findall(".//surface/friction/ode"):
            for tag in ["max_vel", "min_depth"]:
                el = ode.find(tag)
                if el is not None:
                    ode.remove(el)
        # Sensor topics are automatically namespaced by model in Gazebo.
        for sensor in model.findall(".//sensor"):
            for topic in sensor.findall("topic"):
                sensor.remove(topic)
        adapter(model, kind)
        for el in scaled_payload(kind):
            model.append(deepcopy(el))
        drive = child(
            model,
            "plugin",
            filename="gz-sim-diff-drive-system",
            name="gz::sim::systems::DiffDrive",
        )
        for side in ["left", "right"]:
            for end in ["front", "rear"]:
                child(drive, side + "_joint", f"{side}_{end}_wheel")
        # Use actual wheel-center spacing from the original link poses.
        for tag, value in dict(
            wheel_separation=0.726,
            wheel_radius=0.31,
            min_linear_velocity=-0.35,
            max_linear_velocity=0.35,
            min_angular_velocity=-0.7,
            max_angular_velocity=0.7,
            max_linear_acceleration=0.2,
            max_angular_acceleration=0.4,
        ).items():
            child(drive, tag, value)
        child(
            model,
            "plugin",
            filename="gz-sim-joint-state-publisher-system",
            name="gz::sim::systems::JointStatePublisher",
        )
        write_xml(root, dest / "model.sdf")
        cfg = ET.Element("model")
        child(cfg, "name", name)
        child(cfg, "version", "2.0")
        child(cfg, "sdf", "model.sdf", version="1.9")
        child(
            cfg,
            "description",
            f"Explorer R2 conversion: articulated lunar {kind}; original chassis and PBR materials, custom work equipment.",
        )
        write_xml(cfg, dest / "model.config")
        print(
            name,
            "mass",
            sum(float(i.text) for i in model.findall("link/inertial/mass")),
        )


if __name__ == "__main__":
    main()
