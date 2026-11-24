"""Original srb_assets lunar terrain and native rocks for a MOSAIC fleet."""

import argparse
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.ndimage import label
import yaml

from ..config import ROOT
from ..terrain import collision_mesh
from ..world import describe
from ..raycast import Raycaster
from .contracts import Pose
from .maps import KnownMap, traversable
from .scene import ProjectMap
from .traversability import parameters, surface, pool


class GeneratedMap(ProjectMap):
    """Known-map adapter reading the same height grid/rock bounds as the SDF."""

    def __init__(self, config):
        manifest = json.loads(Path(config["scene_manifest"]).read_text())
        if manifest["bounds"] != config["map"]["bounds"] or not math.isclose(
            manifest["resolution"], config["map"]["resolution"]
        ):
            raise ValueError("Map grid must match the generated physical scene")
        self.world = config["world"]
        self.terrain_parameters = parameters(config["map"])
        self.spec = manifest["spec"]
        KnownMap.__init__(self, manifest["bounds"], manifest["resolution"])
        self.heights = np.load(manifest["heights_file"], allow_pickle=False)
        if (
            self.heights.shape != self.grid.blocked.shape
            or np.isnan(self.heights).any()
            or not np.isfinite(self.heights).any()
        ):
            raise ValueError("Invalid generated height grid")
        subdivisions = math.ceil(self.grid.resolution / 0.1)
        step = self.grid.resolution / subdivisions
        fine_path = Path(
            manifest.get(
                "fine_heights_file",
                Path(config["scene_manifest"]).parent / "terrain_heights_fine.npy",
            )
        )
        metadata = fine_path.with_suffix(".json")
        signature = dict(
            terrain=manifest.get("terrain_sha256"),
            bounds=manifest["bounds"],
            resolution=self.grid.resolution,
        )
        cached = manifest.get("fine_heights_file") or (
            fine_path.is_file()
            and metadata.is_file()
            and json.loads(metadata.read_text()) == signature
        )
        if cached:
            fine = np.load(fine_path, allow_pickle=False)
        else:
            model = next(m for m in self.spec["models"] if "surface" in m["name"])
            _, _, fine = _collision_details(
                Raycaster(collision_mesh(model["path"])),
                manifest["size"],
                self.grid.resolution,
                config,
            )
            # Old scenes contain only a binary steepness mask. Reconstruct
            # physical height evidence once; never reuse that old all-rock mask.
            try:
                with tempfile.NamedTemporaryFile(
                    dir=fine_path.parent, suffix=".npy", delete=False
                ) as temp:
                    np.save(temp, fine, allow_pickle=False)
                    temporary = Path(temp.name)
                temporary.replace(fine_path)
                with tempfile.NamedTemporaryFile(
                    dir=metadata.parent, mode="w", delete=False
                ) as temp:
                    json.dump(signature, temp)
                    temporary = Path(temp.name)
                temporary.replace(metadata)
            except PermissionError:
                pass
        expected = tuple((n - 1) * subdivisions + 1 for n in self.heights.shape)
        if (
            fine.shape != expected
            or np.isnan(fine).any()
            or not np.isfinite(fine).any()
        ):
            raise ValueError("Invalid fine terrain height evidence")
        support, fine_blocks = surface(fine, step, config["map"])
        self.terrain_blocked = pool(fine_blocks, subdivisions)
        support = support[::subdivisions, ::subdivisions]
        self.dy, self.dx = np.gradient(support, self.grid.resolution)
        self.objects = manifest["objects"]
        self.poses = {name: list(o["pose"]) for name, o in self.objects.items()}
        from ..planning.local_map import corners

        self.object_corners = {
            name: corners(np.asarray(o["bounds"])) for name, o in self.objects.items()
        }
        self.manual_blocks = np.zeros_like(self.terrain_blocked)
        self._rebuild()

    def _rebuild(self):
        super()._rebuild()
        self.grid = replace(
            self.grid, source=f"generated_collision_geometry:{self.world}"
        )


def _scaled_model(source, destination, scale):
    tree = ET.parse(source)
    model = tree.getroot().find("model")
    node = model.find("static")
    if node is None:
        node = ET.SubElement(model, "static")
    node.text = "true"
    for pose in model.findall(".//pose"):
        values = list(map(float, pose.text.split()))
        values[:3] = [v * scale for v in values[:3]]
        pose.text = " ".join(map(str, values))
    for mesh in model.findall(".//mesh"):
        uri = mesh.find("uri")
        value = Path(uri.text.removeprefix("file://"))
        if not value.is_absolute():
            value = Path(source).parent / value
        uri.text = str(value.resolve())
        node = mesh.find("scale")
        values = list(map(float, node.text.split())) if node is not None else [1, 1, 1]
        if node is None:
            node = ET.SubElement(mesh, "scale")
        node.text = " ".join(str(v * scale) for v in values)
    ET.indent(tree)
    tree.write(destination, encoding="utf-8", xml_declaration=True)


def _collision_details(rays, size, resolution, config):
    """Pool vehicle-scale traversability evidence from the unmodified mesh."""
    subdivisions = math.ceil(resolution / 0.1)
    step = resolution / subdivisions
    count = round(size / step) + 1
    axis = np.linspace(-size / 2, size / 2, count)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.c_[xx.ravel(), yy.ravel()]
    height = np.concatenate(
        [rays.heights(xy[i : i + 100000]) for i in range(0, len(xy), 100000)]
    ).reshape(count, count)
    _, blocked = surface(height, step, config["map"])
    return pool(blocked, subdivisions), step, height


def generate(
    output,
    *,
    seed=17,
    size=80.0,
    scouts=2,
    scientists=2,
    rocks=16,
    resolution=0.25,
    terrain_model=None,
    procgen_image="space_robotics_gz_envs:latest",
    texture_resolution=None,
    rock_density_factor=0.2,
    pebble_spacing=2.0,
):
    if (
        not isinstance(seed, int)
        or seed < 0
        or not 40 <= size <= 300
        or not 0.25 <= resolution <= 1
        or not isinstance(scouts, int)
        or not isinstance(scientists, int)
        or min(scouts, scientists) < 1
        or scouts + scientists > 64
        or not isinstance(rocks, int)
        or rocks < 1
        or not math.isfinite(rock_density_factor)
        or not 0 <= rock_density_factor <= 2
        or not math.isfinite(pebble_spacing)
        or not (pebble_spacing == 0 or 1 <= pebble_spacing <= size / 2)
        or not math.isclose(size / resolution, round(size / resolution))
    ):
        raise ValueError(
            "Require integer seed/counts, 40–300 m grid, 0.25–1 m resolution, 1+ scout/scientist and <=64 robots"
        )
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use an empty scenario directory")
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/mosaic/demo.yaml").read_text())
    rng = np.random.default_rng(seed + 1)
    if terrain_model is None:
        from .procgen import lunar_surface

        terrain_model, provenance = lunar_surface(
            output / "procgen",
            seed,
            size,
            image=procgen_image,
            texture_resolution=texture_resolution,
            rock_density_factor=rock_density_factor,
            pebble_spacing=pebble_spacing,
        )
    else:
        terrain_model = Path(terrain_model).resolve()
        provenance = dict(
            generator="existing_srb_assets_model", source=str(terrain_model)
        )
        generation_file = terrain_model.parent.parent / "generation.json"
        if generation_file.is_file():
            provenance.update(json.loads(generation_file.read_text()))
        # Keep every runtime resource under the scenario's mounted directory.
        copied = output / "terrain_asset"
        shutil.copytree(terrain_model.parent, copied)
        terrain_model = copied / terrain_model.name
    # Copy only the SDF wrapper; retain the original high-detail GLB, UVs,
    # embedded PBR textures and collision mesh without remeshing or scaling.
    ground_path = output / "terrain.sdf"
    _scaled_model(terrain_model, ground_path, 1)
    ground_mesh = collision_mesh(ground_path)
    # The authored terrain includes rocks protruding slightly past its grid.
    if not np.allclose(
        ground_mesh.bounds[:, :2],
        [[-size / 2, -size / 2], [size / 2, size / 2]],
        atol=0.5,
    ):
        raise ValueError("Original terrain model bounds do not match --size")
    rays = Raycaster(ground_mesh)
    n = round(size / resolution) + 1
    axis = np.linspace(-size / 2, size / 2, n)
    xx, yy = np.meshgrid(axis, axis)
    heights = rays.heights(np.c_[xx.ravel(), yy.ravel()]).reshape(n, n)
    # Mesh border rays can miss an exact floating point edge. Border cells
    # are outside the robot footprint envelope anyway; sample just inside.
    if not np.isfinite(heights).all():
        missing = ~np.isfinite(heights)
        xy = np.c_[xx[missing], yy[missing]]
        xy = np.clip(xy, -size / 2 + 0.001, size / 2 - 0.001)
        heights[missing] = rays.heights(xy)
    np.save(output / "heights.npy", heights, allow_pickle=False)
    collision_blocks, collision_step, fine_heights = _collision_details(
        rays, size, resolution, config
    )
    np.save(output / "terrain_blocked.npy", collision_blocks, allow_pickle=False)
    np.save(output / "terrain_heights_fine.npy", fine_heights, allow_pickle=False)
    source = describe("moon", str(ROOT / "assets/cache"))
    tree = ET.parse(source["source"])
    world = tree.getroot().find("world")
    world.set("name", "mosaic_random_moon")
    for include in list(world.findall("include")):
        world.remove(include)
    models = [
        dict(
            name="random_surface",
            path=str(ground_path),
            pose="0 0 0 0 0 0",
            static=True,
        )
    ]
    catalog = [m for m in source["models"] if m["name"].startswith("lunar_rock")]
    bounds = {m["name"]: collision_mesh(m["path"]).bounds for m in catalog}
    variants = {}
    objects = {}
    positions = []
    # A static seeded layout avoids random rolling of tiny source rocks.
    for i in range(rocks):
        for attempt in range(5000):
            x, y = rng.uniform(-size / 2 + 6, size / 2 - 6, 2)
            if all(math.hypot(x - a, y - b) >= 5 for a, b in positions):
                break
        else:
            raise ValueError(
                "Too many rocks for this map; increase size or reduce rocks"
            )
        positions.append((float(x), float(y)))
        item = catalog[int(rng.integers(len(catalog)))]
        scale = 1
        key = (item["name"], scale)
        if key not in variants:
            path = output / f"{key[0]}_scale{scale}.sdf"
            _scaled_model(item["path"], path, scale)
            variants[key] = path
        local_bounds = bounds[item["name"]] * scale
        z = float(rays.heights(np.array([[x, y]]))[0]) - local_bounds[0, 2] + 0.01
        pose = [
            float(x),
            float(y),
            float(z),
            0,
            0,
            float(rng.uniform(-math.pi, math.pi)),
        ]
        name = f"random_rock_{i:03d}"
        objects[name] = dict(bounds=local_bounds.tolist(), pose=pose)
        models.append(
            dict(
                name=name,
                path=str(variants[key]),
                pose=" ".join(map(str, pose)),
                static=True,
            )
        )
    for item in models:
        include = ET.SubElement(world, "include")
        ET.SubElement(include, "name").text = item["name"]
        ET.SubElement(include, "uri").text = item["path"]
        ET.SubElement(include, "pose").text = item["pose"]
        ET.SubElement(include, "static").text = "true"
    scene_path = output / "world.sdf"
    ET.indent(tree)
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)
    manifest = dict(
        terrain_provenance=provenance,
        terrain_blocked_file=str(output / "terrain_blocked.npy"),
        fine_heights_file=str(output / "terrain_heights_fine.npy"),
        collision_sample_resolution=collision_step,
        seed=seed,
        size=size,
        bounds=[-size / 2, -size / 2, size / 2, size / 2],
        resolution=resolution,
        heights_file=str(output / "heights.npy"),
        objects=objects,
        spec=dict(
            source=str(scene_path),
            physics_dt=source["physics_dt"],
            gravity=source["gravity"],
            models=models,
        ),
    )
    manifest_path = output / "scene_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    config.update(
        world="random_moon",
        scene_file=str(scene_path),
        scene_manifest=str(manifest_path),
        instrument_requires_heading=False,
    )
    config["providers"]["global_map"] = "generated_collision_geometry"
    config["map"].update(bounds=manifest["bounds"], resolution=resolution)
    config["exploration"].update(spacing=8.0, queue_size=3)
    maps = GeneratedMap(config)
    safe = traversable(maps.grid, 1.5)
    components, count = label(safe)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    if count == 0:
        raise ValueError("No connected safe terrain; use another seed")
    component = components == int(sizes.argmax())
    ys, xs = np.nonzero(component)
    sites = [Pose(float(axis[x]), float(axis[y])) for x, y in zip(xs, ys)]
    homes = []
    fleet = []
    roles = ["Scout", "Scientist"] * max(scouts, scientists)
    used = {"Scout": 0, "Scientist": 0}
    counts = {"Scout": scouts, "Scientist": scientists}
    for role in roles:
        if used[role] >= counts[role]:
            continue
        lane = (used[role] / (counts[role] - 1) - 0.5) if counts[role] > 1 else 0
        home_center = Pose(
            -size * (0.25 if role == "Scout" else 0.38), size * 0.4 * lane
        )
        home_candidates = sorted(sites, key=home_center.distance)
        home = next(
            (p for p in home_candidates if all(p.distance(q) >= 6 for q in homes)), None
        )
        if home is None:
            raise ValueError("Cannot fit requested fleet with safe spawn spacing")
        homes.append(home)
        used[role] += 1
        template = copy.deepcopy(config["robots"][0 if role == "Scout" else 1])
        template.update(
            name=f"{role.lower()}_{used[role]:02d}",
            label=f"R2 {role} {used[role]}",
            home=[home.x, home.y, 0],
            allow_reverse=True,
        )
        fleet.append(template)
    config["robots"] = fleet
    config["pois"] = []
    # Seed one reachable, separate exploration task per scout. Everything
    # scientific is created later by detection and close confirmation.
    from .navigation import AStarPlanner

    planner = AStarPlanner()
    chosen = []
    for robot in fleet:
        if robot["role"] != "Scout":
            continue
        start = Pose(*robot["home"])
        candidates = sorted(
            (
                p
                for p in sites
                if p.distance(start) >= 5
                and all(p.distance(q) >= 4 for q in homes)
                and all(p.distance(q) >= 6 for q in chosen)
            ),
            # Start scouts outwards, away from the parked science vehicles.
            key=lambda p: (p.x < start.x + 4, start.distance(p)),
        )
        station = next(
            (
                p
                for p in candidates
                if planner.plan(start, p, maps.grid, 1.5) is not None
            ),
            None,
        )
        if station is None:
            raise ValueError("Cannot seed exploration; try another seed/size")
        chosen.append(station)
        config["pois"].append(
            dict(
                id=f"explore_start_{len(chosen):02d}",
                kind="explore",
                pose=[station.x, station.y, 0],
            )
        )
    config["targets"] = [
        dict(
            id=name,
            object=name,
            approach_selection="nearest_reachable",
            inspection_max_distance=3.0,
            inspection_clearance=0.6,
            measurement_max_distance=3.5,
        )
        for name in objects
    ]
    config["camera_pose"] = f"0 {-size*.7} {size*.9} 0 0.9 1.57079632679"
    config_path = output / "mission.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    manifest["terrain_sha256"] = hashlib.sha256(
        ground_mesh.vertices.tobytes() + ground_mesh.faces.tobytes()
    ).hexdigest()
    manifest["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return config_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--size", type=float, default=80)
    parser.add_argument("--scouts", type=int, default=2)
    parser.add_argument("--scientists", type=int, default=2)
    parser.add_argument("--rocks", type=int, default=16)
    parser.add_argument(
        "--rock-density-factor",
        type=float,
        default=0.2,
        help="Embedded rock scatter multiplier for newly baked terrain; 1 keeps upstream density",
    )
    parser.add_argument("--resolution", type=float, default=0.25)
    parser.add_argument(
        "--pebble-spacing",
        type=float,
        default=2.0,
        help="Jittered low-pebble stratum size in metres for new terrain; 0 disables",
    )
    parser.add_argument(
        "--terrain-model",
        type=Path,
        help="Reuse an original procgen model.sdf of the requested size",
    )
    parser.add_argument("--procgen-image", default="space_robotics_gz_envs:latest")
    parser.add_argument(
        "--texture-resolution",
        type=int,
        help="PBR texture size; defaults to matching the original texels/metre",
    )
    args = parser.parse_args(argv)
    try:
        path = generate(**vars(args))
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
