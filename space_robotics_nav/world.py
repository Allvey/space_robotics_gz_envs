"""Read the original SDF and resolve resource paths; never replace its assets or physics."""

import copy
import hashlib
import math
import uuid
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
from scipy.spatial.transform import Rotation

from .config import ROOT


def transform(text):
    p = list(map(float, (text or "0 0 0 0 0 0").split()))
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("xyz", p[3:]).as_matrix()
    t[:3, 3] = p[:3]
    return t


def resolve_model(uri, assets):
    uri = uri.strip()
    assets = Path(assets)
    if uri.startswith("model://"):
        path = assets / uri.removeprefix("model://") / "model.sdf"
        if path.is_file():
            return path.resolve()
    elif uri.startswith("http"):
        parts = unquote(urlparse(uri).path).strip("/").split("/")
        owner, name = parts[1].lower(), parts[3].lower()
        candidates = [
            p
            for p in (assets / "fuel").rglob("model.sdf")
            if p.parent.parent.name.lower() == name
            and p.parents[3].name.lower() == owner
        ]
        if candidates:
            return max(candidates, key=lambda p: int(p.parent.name)).resolve()
    else:
        path = Path(uri.removeprefix("file://"))
        if path.is_dir():
            path /= "model.sdf"
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError(
        f"Missing native project asset: {uri}. Run tools/navigation/setup.bash."
    )


@lru_cache(maxsize=4)
def describe(world, assets):
    source = ROOT / "worlds" / f"{world}.sdf"
    node = ET.parse(source).getroot().find("world")
    models = []
    for include in node.findall("include"):
        path = resolve_model(include.findtext("uri"), assets)
        model = ET.parse(path).getroot().find("model")
        item = {
            "name": include.findtext("name") or model.get("name"),
            "path": str(path),
            "pose": include.findtext("pose") or model.findtext("pose") or "0 0 0 0 0 0",
            "static": (
                include.findtext("static") or model.findtext("static") or "false"
            ).strip()
            == "true",
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        drive = include.find("plugin[@name='gz::sim::systems::DiffDrive']")
        if drive is not None:
            item["controller"] = ET.tostring(drive, encoding="unicode")
            item["body"] = model.get("canonical_link") or model.find("link").get("name")
            item["limits"] = {
                key: float(drive.findtext(key))
                for key in (
                    "max_linear_velocity",
                    "min_linear_velocity",
                    "max_angular_velocity",
                    "min_angular_velocity",
                )
            }
        models.append(item)
    return {
        "source": str(source),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "physics_dt": float(node.findtext("physics/max_step_size")),
        "gravity": list(map(float, node.findtext("gravity").split())),
        "models": models,
        "robot": next(m for m in models if "controller" in m),
        "heading_offset": -math.pi / 2 if world == "mars" else 0.0,
    }


def runtime_world(cfg, *, gui=False):
    spec = describe(cfg.world, cfg.assets)
    root = ET.parse(spec["source"]).getroot()
    world = root.find("world")
    for include, model in zip(world.findall("include"), spec["models"]):
        include.find("uri").text = model["path"]
    if gui:
        marker = ET.fromstring(
            """<model name="navigation_goal"><static>true</static>
          <pose>0 0 -100 0 0 0</pose><link name="annotation">
          <visual name="goal"><pose>0 0 0.7 0 0 0</pose>
          <geometry><cylinder><radius>0.05</radius><length>1.4</length></cylinder></geometry>
          <material><diffuse>0 1 0 1</diffuse><emissive>0 0.7 0 1</emissive></material></visual>
          <visual name="heading"><pose>0.4 0 0.15 0 0 0</pose>
          <geometry><box><size>0.8 0.08 0.08</size></box></geometry>
          <material><diffuse>1 0.4 0 1</diffuse><emissive>0.7 0.2 0 1</emissive></material></visual>
          </link></model>"""
        )
        world.append(marker)
    folder = ROOT / ".cache/navigation/worlds"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{uuid.uuid4().hex}.sdf"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def gui_config(cfg, path, origin, goal):
    # Reuse the original lighting/render settings; frame both rover and goal.
    original = (
        ET.parse(ROOT / "worlds" / f"{cfg.world}.sdf").getroot().find("world/gui")
    )
    nodes = [ET.Element("dialog", name="quick_start", show_again="false")]
    window = ET.Element("window")
    for tag, text in (
        ("width", "1280"),
        ("height", "850"),
        ("dialog_on_exit", "false"),
    ):
        ET.SubElement(window, tag).text = text
    nodes.append(window)
    scene = copy.deepcopy(original.find("plugin[@filename='MinimalScene']"))
    center = (np.asarray(origin) + np.asarray(goal)) / 2
    height = max(origin[2], goal[2]) + 10
    pitch = math.atan2(height - center[2], math.hypot(9, 10))
    scene.find("camera_pose").text = " ".join(
        map(str, [center[0] - 9, center[1] - 10, height, 0, pitch, math.atan2(10, 9)])
    )
    ET.SubElement(scene, "scene").text = "scene"
    nodes.append(scene)
    for name in (
        "GzSceneManager",
        "InteractiveViewControl",
        "CameraTracking",
        "EntityContextMenuPlugin",
        "SelectEntities",
    ):
        p = ET.Element("plugin", filename=name, name=name)
        g = ET.SubElement(p, "gz-gui")
        for key, kind, value in (
            ("state", "string", "floating"),
            ("showTitleBar", "bool", "false"),
            ("width", "double", "5"),
            ("height", "double", "5"),
        ):
            ET.SubElement(g, "property", key=key, type=kind).text = value
        nodes.append(p)
    path.write_text("\n".join(ET.tostring(n, encoding="unicode") for n in nodes) + "\n")
    return path
