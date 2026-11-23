"""Asset contracts that catch broken conversion, scale, cavity, and resource paths."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import trimesh

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "assets/models"


def matrix(pose):
    values = [float(x) for x in (pose or "0 0 0 0 0 0").split()]
    mat = np.eye(4)
    mat[:3, :3] = Rotation.from_euler("xyz", values[3:]).as_matrix()
    mat[:3, 3] = values[:3]
    return mat


def test_original_assets_preserved_byte_for_byte():
    common = MODELS / "explorer_r2_base"
    hashes = json.loads((common / "source_manifest.json").read_text())["sha256"]
    for name, digest in hashes.items():
        assert hashlib.sha256((common / name).read_bytes()).hexdigest() == digest


def test_hauler_camera_outside_bin_sweep():
    model = ET.parse(MODELS / "lunar_hauler/model.sdf").getroot().find("model")
    frame = model.find("link[@name='payload_frame']")
    # Bin tilts about Y; its lateral envelope is invariant at every angle.
    # Include the rolled lip and stiffeners, not only the thin collision wall.
    bin_outer_half_width = 0.37
    for name in ["work_camera_mast", "work_camera_housing"]:
        c = frame.find(f"collision[@name='{name}']")
        center = np.fromstring(c.findtext("pose"), sep=" ")[:3]
        half = np.fromstring(c.findtext("geometry/box/size"), sep=" ") / 2
        assert center[1] + half[1] < -bin_outer_half_width
    # The support crossing under the bed stays below its floor at rest;
    # negative pitch lifts the bed farther away from this support.
    support = frame.find("collision[@name='camera_outrigger']")
    z = np.fromstring(support.findtext("pose"), sep=" ")[2]
    half_z = np.fromstring(support.findtext("geometry/box/size"), sep=" ")[2] / 2
    assert z + half_z < 0.09 - 0.7 * 0.04 / 2


@pytest.mark.parametrize("kind", ["excavator", "hauler"])
def test_asset_physics_and_resources(kind):
    directory = MODELS / f"lunar_{kind}"
    model = ET.parse(directory / "model.sdf").getroot().find("model")
    base = ET.parse(MODELS / "explorer_r2_base/model.sdf").getroot().find("model")
    links = {l.get("name"): l for l in model.findall("link")}
    joints = {j.get("name"): j for j in model.findall("joint")}
    assert not links.keys() & joints.keys()
    # Original mass, COM and inertia must survive the conversion untouched.
    for original in base.findall("link"):
        converted = links[original.get("name")]
        for el in original.findall("inertial/*") + original.findall(
            "inertial/inertia/*"
        ):
            path = (
                "inertial/inertia/" + el.tag
                if el.tag.startswith("i") and el.tag != "inertia"
                else "inertial/" + el.tag
            )
            if list(el):
                continue
            assert np.allclose(
                np.fromstring(el.text, sep=" "),
                np.fromstring(converted.findtext(path), sep=" "),
            )
    for l in links.values():
        assert float(l.findtext("inertial/mass")) > 0
        i = l.find("inertial/inertia")
        tensor = np.array(
            [
                [float(i.findtext("i" + a + b, i.findtext("i" + b + a))) for b in "xyz"]
                for a in "xyz"
            ]
        )
        eigen = np.linalg.eigvalsh(tensor)
        assert np.isfinite(eigen).all() and eigen.min() > 0
        assert eigen.max() <= eigen.sum() - eigen.max() + 1e-8
    for j in joints.values():
        assert j.findtext("parent") in links and j.findtext("child") in links
    for uri in (
        model.findall(".//uri")
        + model.findall(".//albedo_map")
        + model.findall(".//metalness_map")
        + model.findall(".//roughness_map")
    ):
        path = (
            MODELS / uri.text.removeprefix("model://")
            if uri.text.startswith("model://")
            else directory / uri.text
        )
        assert path.exists(), path
    controls = model.findall(
        "plugin[@name='gz::sim::systems::JointPositionController']"
    )
    assert len(controls) == (4 if kind == "excavator" else 2)
    for control in controls:
        assert control.findtext("joint_name") in joints
        assert control.find("topic") is None
    # Every added moving visual must contain its collision geometry at the
    # same scale and origin. A missing glTF Y-up transform fails this check.
    for name in (
        ["turret", "boom", "forearm", "bucket"]
        if kind == "excavator"
        else ["dump_bed", "tailgate"]
    ):
        link = links[name]
        visual = link.find("visual")
        scene = trimesh.load(
            directory / visual.findtext("geometry/mesh/uri"), force="scene"
        )
        scale = np.eye(4)
        scale[:3, :3] = np.diag(
            np.fromstring(visual.findtext("geometry/mesh/scale"), sep=" ")
        )
        scene.apply_transform(matrix(visual.findtext("pose")) @ scale)
        bounds = scene.bounds
        for collision in link.findall("collision"):
            geo = collision.find("geometry")
            if geo.find("box") is not None:
                mesh = trimesh.creation.box(
                    np.fromstring(geo.findtext("box/size"), sep=" ")
                )
            else:
                mesh = trimesh.creation.cylinder(
                    float(geo.findtext("cylinder/radius")),
                    float(geo.findtext("cylinder/length")),
                    sections=64,
                )
            mesh.apply_transform(matrix(collision.findtext("pose")))
            assert (mesh.bounds[0] >= bounds[0] - 0.003).all(), name
            assert (mesh.bounds[1] <= bounds[1] + 0.003).all(), name


@pytest.mark.parametrize(
    "kind,link_name,point",
    [("excavator", "bucket", (0.15, 0, 0.07)), ("hauler", "dump_bed", (0.45, 0, 0.15))],
)
def test_payload_has_open_cavity(kind, link_name, point):
    model = ET.parse(MODELS / f"lunar_{kind}/model.sdf").getroot().find("model")
    link = model.find(f"link[@name='{link_name}']")
    for collision in link.findall("collision"):
        geo = collision.find("geometry")
        local = np.linalg.inv(matrix(collision.findtext("pose"))) @ np.r_[point, 1]
        assert geo.find("mesh") is None, "Do not close the cavity with a convex hull"
        if geo.find("box") is not None:
            half = np.fromstring(geo.findtext("box/size"), sep=" ") / 2
            assert not np.all(np.abs(local[:3]) < half)
        elif geo.find("cylinder") is not None:
            assert not (
                np.linalg.norm(local[:2]) < float(geo.findtext("cylinder/radius"))
                and abs(local[2]) < float(geo.findtext("cylinder/length")) / 2
            )
