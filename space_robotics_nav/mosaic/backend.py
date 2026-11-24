"""A single shared Gazebo process, separate from training episode/reset APIs."""

import json
import copy
import math
import os
from pathlib import Path
import selectors
import subprocess
import uuid
import xml.etree.ElementTree as ET

from ..backend import GazeboBackend
from ..config import ROOT
from .scene import known_map
from .vehicles import configured_vehicle


def runtime_world(config, folder, maps=None):
    maps = maps if maps is not None else known_map(config)
    source_world = config.get("world", "mining_vehicles")
    root = ET.parse(
        config.get("scene_file", ROOT / "worlds" / f"{source_world}.sdf")
    ).getroot()
    identifier = uuid.uuid4().hex
    world = root.find("world")
    world.find("gui/plugin[@filename='MinimalScene']/camera_pose").text = config.get(
        "camera_pose", "-14 -16 17 0 0.65 0.9"
    )
    # The mission worker owns the simulation clock; GUI controls must not pause
    # or reset the world underneath an atomic batch.
    control = world.find("gui/plugin[@filename='WorldControl']")
    if control is not None:
        world.find("gui").remove(control)
    for plugin in list(world.findall("gui/plugin")):
        if plugin.get("filename") in {"Teleop", "ResourceSpawner"}:
            world.find("gui").remove(plugin)
    if hasattr(maps, "spec"):
        # Remove only the original single rover. Retain every original terrain
        # and rock include, transform, static flag and collision/visual asset.
        for include, item in zip(world.findall("include"), maps.spec["models"]):
            if "controller" in item:
                world.remove(include)
            else:
                include.find("uri").text = item["path"]
                if config.get("scene_objects_static") and item["name"] in maps.objects:
                    # Geology targets are fixed at the terrain surface. Source
                    # files remain untouched; vehicles retain wheel physics.
                    from ..world import transform

                    pose = list(maps.poses[item["name"]])
                    rotation = transform(" ".join(map(str, [0, 0, 0, *pose[3:]])))[
                        :3, :3
                    ]
                    local = maps.object_corners[item["name"]] @ rotation.T
                    pose[2] = maps.height(*pose[:2]) - float(local[:, 2].min()) + 0.01
                    include.find("pose").text = " ".join(map(str, pose))
                    node = include.find("static")
                    if node is None:
                        node = ET.SubElement(include, "static")
                    node.text = "true"
                    maps.poses[item["name"]] = pose
        if config.get("scene_objects_static"):
            maps._rebuild()
    else:
        for include in world.findall("include"):
            world.remove(include)
    for plugin in world.findall("plugin"):
        if not config.get("sensors_enabled", False) and plugin.get("name", "").endswith(
            ("Sensors", "Imu")
        ):
            world.remove(plugin)
    if world.find("plugin[@name='gz::sim::systems::Contact']") is None:
        ET.SubElement(
            world,
            "plugin",
            filename="gz-sim-contact-system",
            name="gz::sim::systems::Contact",
        )
    for robot in config["robots"]:
        spec = configured_vehicle(robot, config)
        model = spec.path
        if not model.is_file():
            raise FileNotFoundError(model)
        if spec.controller:
            model = spec.runtime_model(folder / f"{identifier}.{robot['name']}.sdf")
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "uri").text = str(model)
        ET.SubElement(include, "name").text = robot["name"]
        ET.SubElement(include, "pose").text = " ".join(
            map(str, spec.spawn(maps, robot["home"]))
        )
        if spec.controller:
            drive = ET.fromstring(spec.controller)
            drive.find("topic").text = f"/model/{robot['name']}/cmd_vel"
            if spec.wheel_separation is not None:
                drive.find("wheel_separation").text = str(spec.wheel_separation)
            include.append(drive)
    for obstacle in config["obstacles"]:
        model = ET.SubElement(world, "model", name=obstacle["id"])
        ET.SubElement(model, "static").text = "true"
        x0, y0, x1, y1 = obstacle["bounds"]
        height = float(obstacle.get("height", 1.0))
        ET.SubElement(model, "pose").text = f"{(x0+x1)/2} {(y0+y1)/2} {height/2} 0 0 0"
        link = ET.SubElement(model, "link", name="body")
        for tag in ("collision", "visual"):
            item = ET.SubElement(link, tag, name=tag)
            geometry = ET.SubElement(item, "geometry")
            box = ET.SubElement(geometry, "box")
            ET.SubElement(box, "size").text = f"{x1-x0} {y1-y0} {height}"
            if tag == "visual":
                material = ET.SubElement(item, "material")
                ET.SubElement(material, "diffuse").text = "0.35 0.3 0.25 1"
    path = folder / f"{identifier}.sdf"
    ET.indent(root)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    return path


def mission_gui_config(world_path):
    """Load the mission camera explicitly; a GUI-only client ignores SDF GUI settings."""
    source = ET.parse(world_path).getroot().find("world/gui")
    nodes = [ET.Element("dialog", name="quick_start", show_again="false")]
    window = ET.Element("window")
    for tag, value in (
        ("width", "1280"),
        ("height", "850"),
        ("dialog_on_exit", "false"),
    ):
        ET.SubElement(window, tag).text = value
    nodes.append(window)
    scene = copy.deepcopy(source.find("plugin[@filename='MinimalScene']"))
    ET.SubElement(scene, "scene").text = "scene"
    nodes.append(scene)
    for name in (
        "GzSceneManager",
        "MarkerManager",
        "InteractiveViewControl",
        "CameraTracking",
        "EntityContextMenuPlugin",
        "SelectEntities",
    ):
        plugin = ET.Element("plugin", filename=name, name=name)
        gui = ET.SubElement(plugin, "gz-gui")
        for key, kind, value in (
            ("state", "string", "floating"),
            ("showTitleBar", "bool", "false"),
            ("width", "double", "5"),
            ("height", "double", "5"),
        ):
            ET.SubElement(gui, "property", key=key, type=kind).text = value
        nodes.append(plugin)
    path = world_path.with_suffix(".gui.config")
    path.write_text(
        "\n".join(ET.tostring(node, encoding="unicode") for node in nodes) + "\n"
    )
    return path


class MultiRobotBackend(GazeboBackend):
    def __init__(self, config, *, gui=False, software=False):
        self.closed = False
        self.process = self.selector = self.gui = self.gui_log = self.auth = None
        self.gui_config = None
        self.path = None
        folder = ROOT / ".cache/mosaic/worlds"
        folder.mkdir(parents=True, exist_ok=True)
        self.maps = known_map(config)
        self.path = runtime_world(config, folder, self.maps)
        world = ET.parse(self.path).getroot().find("world")
        self.vehicle_models = {
            include.findtext("name"): include.findtext("uri")
            for include in world.findall("include")
            if include.findtext("name") in {r["name"] for r in config["robots"]}
        }
        self.physics_dt = float(world.findtext("physics/max_step_size"))
        self.name = f"space-mosaic-{os.getpid()}-{self.path.stem[:8]}"
        self.log = self.path.with_suffix(".log").open("w")
        self.buffer = b""
        self.control_dt = float(config.get("control_dt", 0.2))
        self.decimation = round(self.control_dt / self.physics_dt)
        if self.decimation < 1 or not math.isclose(
            self.decimation * self.physics_dt, self.control_dt
        ):
            self.close()
            raise ValueError(
                "Control period must be a multiple of the source world physics step"
            )
        self.descriptors = {
            r["name"]: dict(
                body=configured_vehicle(r, config).body,
                max_linear=r.get("max_linear", 0.25),
                max_angular=r.get("max_angular", 0.45),
            )
            for r in config["robots"]
        }
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.name,
            "--network",
            "none",
            "--env",
            f"GZ_PARTITION={self.name}",
            "--env",
            "GZ_IP=127.0.0.1",
            "--env",
            f"MOSAIC_ROBOTS={json.dumps(self.descriptors)}",
            "--env",
            f"MOSAIC_SETTLE_STEPS={round(4 / self.physics_dt)}",
            "--env",
            f"MOSAIC_GROUND_MODELS={json.dumps([m['name'] for m in self.maps.spec['models'] if 'surface' in m['name']] if hasattr(self.maps, 'spec') else ['ground'])}",
            "--env",
            f"GZ_SIM_RESOURCE_PATH={ROOT / 'assets/models'}:{config.get('assets', ROOT / 'assets/cache')}",
            "--volume",
            f"{ROOT / 'assets/models'}:{ROOT / 'assets/models'}:ro",
            "--volume",
            f"{Path(config.get('assets', ROOT / 'assets/cache')).resolve()}:{Path(config.get('assets', ROOT / 'assets/cache')).resolve()}:ro",
            "--volume",
            f"{folder}:{folder}:ro",
            "--entrypoint",
            "/usr/local/bin/mosaic-worker",
        ]
        if config.get("scene_manifest"):
            scene_folder = Path(config["scene_manifest"]).resolve().parent
            command += ["--volume", f"{scene_folder}:{scene_folder}:ro"]
        try:
            if gui:
                command += self._display_options(software)
            self.process = subprocess.Popen(
                command
                + [config.get("image", "space-robotics-rl:ionic"), str(self.path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
            )
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            ready = self._receive(120)
            if not ready.get("ready"):
                raise RuntimeError("Multi-robot worker did not initialize")
            self.state = ready["state"]
            if gui:
                self.start_gui(mission_gui_config(self.path))
        except BaseException:
            self.close()
            raise

    def step(self, actions, *, markers=None):
        if set(actions) != set(self.descriptors):
            raise ValueError("A batch must contain exactly one action per robot")
        self.state = self.request(
            dict(
                op="step",
                velocities=actions,
                steps=self.decimation,
                markers=markers or [],
            )
        )
        return self.state

    def reset(self, *args, **kwargs):
        raise RuntimeError(
            "Mission execution has no episode reset; restart the world explicitly"
        )
