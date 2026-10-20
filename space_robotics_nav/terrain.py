"""Navigation sampling and scans use only this project's collision geometry."""

import hashlib
import os
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation

from .raycast import Raycaster
from .vendor.terrain import HeightmapManager, TerrainOps
from .world import describe, transform


def collision_mesh(path):
    model = ET.parse(path).getroot().find("model")
    parts = []
    for link in model.findall("link"):
        for collision in link.findall("collision"):
            geom = collision.find("geometry")
            mesh = geom.find("mesh")
            if mesh is None:
                raise ValueError(
                    f"Expected the original procgen collision mesh in {path}"
                )
            uri = mesh.findtext("uri").strip()
            file = Path(uri.removeprefix("file://"))
            if not file.is_absolute():
                file = Path(path).parent / file
            shape = trimesh.load(file, force="mesh", process=False)
            shape.apply_scale(
                list(map(float, (mesh.findtext("scale") or "1 1 1").split()))
            )
            shape.apply_transform(
                transform(link.findtext("pose")) @ transform(collision.findtext("pose"))
            )
            parts.append(shape)
    return trimesh.util.concatenate(parts)


@lru_cache(maxsize=4)
def geometry(world, assets):
    spec = describe(world, assets)
    ground_model = next(m for m in spec["models"] if "surface" in m["name"])
    ground = collision_mesh(ground_model["path"])
    ground.apply_transform(transform(ground_model["pose"]))
    objects = {}
    for model in spec["models"]:
        if model is ground_model or "controller" in model:
            continue
        mesh = collision_mesh(model["path"])
        objects[model["name"]] = {
            "mesh": mesh,
            "rays": Raycaster(mesh),
            "bounds": mesh.bounds,
            "pose": list(map(float, model["pose"].split())),
        }
    return ground, Raycaster(ground), objects


class Terrain(TerrainOps):
    def __init__(self, cfg):
        self.cfg = cfg
        self.spec = describe(cfg.world, cfg.assets)
        self.target_distance = cfg.target_distance
        self.mesh, self.rays, self.objects = geometry(cfg.world, cfg.assets)
        self.poses = {name: obj["pose"] for name, obj in self.objects.items()}
        self.bounds = self.mesh.bounds[:, :2]
        # Isaac Lab GridPatternCfg defaults to torch.meshgrid(indexing='xy').
        axis = torch.arange(-2.5, 2.5 + 1e-9, 0.05)
        x, y = torch.meshgrid(axis, axis, indexing="xy")
        self.grid = torch.stack([x.flatten(), y.flatten()], dim=1).numpy()
        self.device = "cpu"
        self.resolution_in_m = 0.05
        self.target_distance_to_boundary = cfg.target_margin
        self.spawns = self._spawns()

    def _spawns(self):
        rocks = []
        for name, obj in self.objects.items():
            mesh = obj["mesh"].copy()
            mesh.apply_transform(transform(" ".join(map(str, obj["pose"]))))
            rocks.append(mesh)
        rocks = trimesh.util.concatenate(rocks)
        combined = trimesh.util.concatenate([self.mesh, rocks])
        signature = hashlib.sha256(
            combined.vertices.tobytes()
            + combined.faces.tobytes()
            + str(
                (
                    self.cfg.slope_limit,
                    self.cfg.spawn_margin,
                    self.cfg.obstacle_margin,
                    self.cfg.seed,
                )
            ).encode()
        ).hexdigest()[:16]
        path = (
            Path(self.cfg.assets)
            / f"rl-source-sampling-{self.cfg.world}-{signature}.npz"
        )
        if path.exists():
            with np.load(path) as data:
                h = HeightmapManager.__new__(HeightmapManager)
                h.heightmap = data["heightmap"]
                h.min_x, h.min_y, h.max_x, h.max_y = data["bounds"]
                h.device, h.resolution_in_m = "cpu", 0.05
                h._initialize_tensors()
                self._heightmap_manager = h
                self.safe_rock_mask_tensor = torch.from_numpy(
                    data["rock_mask"]
                ).unsqueeze(-1)
                return data["spawns"]
        self._heightmap_manager = HeightmapManager(
            0.05,
            np.asarray(combined.vertices, dtype=np.float32),
            np.asarray(combined.faces),
        )
        _, rock_mask = self.project_rocks_to_heightmap(
            np.asarray(rocks.vertices),
            np.asarray(rocks.faces),
            self.cfg.obstacle_margin,
        )
        ground = self.resize_terrain_heightmap_to_match_combined(
            np.asarray(self.mesh.vertices),
            np.asarray(self.mesh.faces),
            self._heightmap_manager,
        )
        _, gradient_mask = self.compute_gradient_masks(
            ground.heightmap, self.cfg.slope_limit, self.cfg.obstacle_margin
        )
        rng_state = np.random.get_state()
        try:
            spawns = self.random_rover_spawns(
                rock_mask | gradient_mask,
                self._heightmap_manager.heightmap,
                4096,
                self.cfg.spawn_margin,
                self.cfg.seed,
            )
        finally:
            np.random.set_state(rng_state)
        self.safe_rock_mask_tensor = torch.from_numpy(rock_mask).unsqueeze(-1)
        h = self._heightmap_manager
        temporary = path.with_suffix(f".{os.getpid()}.npz")
        np.savez_compressed(
            temporary,
            heightmap=h.heightmap,
            bounds=[h.min_x, h.min_y, h.max_x, h.max_y],
            rock_mask=rock_mask,
            spawns=spawns,
        )
        temporary.replace(path)
        return spawns

    def log(self, *args, **kwargs):
        pass

    def update(self, state):
        self.poses.update(state["objects"])

    def sample(self, rng):
        origin = self.spawns[rng.integers(len(self.spawns))].copy()
        for _ in range(20000):
            theta = rng.uniform(0, 2 * np.pi)
            xy = origin[:2] + self.target_distance * np.array(
                [np.cos(theta), np.sin(theta)]
            )
            _, invalid = self.check_if_target_is_valid(
                torch.tensor([0]), torch.as_tensor(xy[None]), "cpu"
            )
            if invalid:
                continue
            z = (
                self._heightmap_manager.get_height_at(torch.as_tensor(xy[None])).item()
                + 0.5
            )
            return (
                origin,
                rng.uniform(-np.pi, np.pi),
                np.r_[xy, z],
                rng.uniform(-np.pi, np.pi),
            )
        raise RuntimeError(
            "No valid target under the upstream terrain safety constraints"
        )

    def contains(self, position):
        xy = np.asarray(position[:2])
        return bool(
            np.all(xy >= self.bounds[0] + 3) and np.all(xy <= self.bounds[1] - 3)
        )

    def scan(self, position, rpy):
        sensor = np.asarray(position) + Rotation.from_euler("xyz", rpy).apply(
            [0, 0, 10]
        )
        yaw = rpy[2] + self.spec["heading_offset"]
        c, s = np.cos(yaw), np.sin(yaw)
        xy = self.grid @ np.array([[c, s], [-s, c]]) + sensor[:2]
        hit = self.rays.heights(xy, sensor[2])
        origins = np.column_stack([xy, np.full(len(xy), sensor[2])])
        for name, obj in self.objects.items():
            pose = np.asarray(self.poses[name])
            radius = np.linalg.norm(obj["bounds"][1] - obj["bounds"][0]) / 2
            if np.linalg.norm(pose[:2] - np.asarray(position[:2])) > radius + 4:
                continue
            rotation = Rotation.from_euler("xyz", pose[3:]).as_matrix()
            distances = obj["rays"].distances(
                (origins - pose[:3]) @ rotation, np.array([0, 0, -1]) @ rotation
            )
            valid = np.isfinite(distances)
            hit[valid] = np.maximum(hit[valid], sensor[2] - distances[valid])
        hit[~np.isfinite(hit)] = sensor[2] - 100
        return sensor, np.column_stack([xy, hit]).astype(np.float32)
