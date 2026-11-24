"""Vehicle adapters reuse the project's original rover assets and drive plugins."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import math
import xml.etree.ElementTree as ET

from scipy.spatial.transform import Rotation

from ..config import ROOT
from ..world import resolve_model


NATIVE_ROVERS = {"explorer_r2": "moon", "nasa_perseverance": "mars"}
SUPPORTED_MODELS = {*NATIVE_ROVERS, "lunar_hauler", "lunar_excavator"}


@dataclass(frozen=True)
class VehicleSpec:
    path: Path
    body: str = "base_link"
    controller: str | None = None
    # Navigation heading relative to the model's +X axis. Perseverance drives -Y.
    heading_offset: float = 0.0
    spawn_height: float = 0.48
    minimum_radius: float = 0.0
    wheel_separation: float | None = None
    wheel_slip: float | None = None

    def spawn(self, maps, home):
        pose = maps.spawn(*home)
        # First align the navigation frame to the slope, then rotate the model
        # around its local vertical; simply adding yaw would misalign the wheels.
        rotation = Rotation.from_euler("xyz", pose[3:]) * Rotation.from_euler(
            "z", -self.heading_offset
        )
        return [
            *pose[:2],
            pose[2] + self.spawn_height - 0.48,
            *rotation.as_euler("xyz"),
        ]

    def runtime_model(self, destination):
        """Keep native geometry/inertia; patch contact slip in a runtime copy.

        The upstream Perseverance tires have force-dependent lateral slip. With
        the 1025 kg chassis this produces creeping even with zero wheel speed.
        The minimal demo uses rigid Coulomb tire contact instead. No body pose,
        body velocity, gravity or terrain changes are applied.
        """
        tree = ET.parse(self.path)
        model = tree.getroot().find("model")
        for element in model.iter():
            if (
                element.tag
                in {"uri", "albedo_map", "normal_map", "metalness_map", "roughness_map"}
                and element.text
            ):
                value = element.text.strip()
                if "://" not in value and not Path(value).is_absolute():
                    resource = self.path.parent / value
                    if resource.exists():
                        element.text = str(resource.resolve())
        for visualize in model.findall(".//sensor/visualize"):
            visualize.text = "false"
        for light in model.findall(".//light"):
            visualize = light.find("visualize")
            if visualize is None:
                visualize = ET.SubElement(light, "visualize")
            visualize.text = "false"
        if self.wheel_slip is not None:
            for link in model.findall("link"):
                if "wheel" not in link.get("name", "").lower():
                    continue
                for friction in link.findall("collision/surface/friction/ode"):
                    for tag in ("slip1", "slip2"):
                        node = friction.find(tag)
                        if node is None:
                            node = ET.SubElement(friction, tag)
                        node.text = str(self.wheel_slip)
        ET.indent(tree)
        tree.write(destination, encoding="utf-8", xml_declaration=True)
        return destination


@lru_cache(maxsize=16)
def vehicle_spec(model, assets):
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported mission vehicle: {model}")
    if model not in NATIVE_ROVERS:
        return VehicleSpec(ROOT / "assets/models" / model / "model.sdf")
    source = ET.parse(ROOT / "worlds" / f"{NATIVE_ROVERS[model]}.sdf")
    include = source.find("world/include[plugin]")
    path = resolve_model(include.findtext("uri"), assets)
    node = ET.parse(path).getroot().find("model")
    drive = include.find("plugin[@name='gz::sim::systems::DiffDrive']")
    from ..planning.local_map import robot_radius

    return VehicleSpec(
        path,
        node.get("canonical_link") or node.find("link").get("name"),
        ET.tostring(drive, encoding="unicode"),
        -math.pi / 2 if model == "nasa_perseverance" else 0.0,
        0.25 if model == "nasa_perseverance" else 0.48,
        robot_radius(path),
        0.726 if model == "explorer_r2" else None,
        0.0,
    )


def configured_vehicle(robot, config):
    return vehicle_spec(
        robot["model"], str(Path(config.get("assets", ROOT / "assets/cache")).resolve())
    )
