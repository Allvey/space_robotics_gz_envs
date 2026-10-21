"""Conservative local traversability from the original Gazebo collision assets.

This optional planner has an explicit map interface. It does not change the policy's
height scan, rewards or task, and its scores must be labelled as hybrid navigation.
"""

import itertools
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt, gaussian_filter

from ..world import transform


def corners(bounds):
    return np.array(list(itertools.product(*np.asarray(bounds).T)))


@lru_cache(maxsize=4)
def robot_radius(model_path):
    """Enclose the native collision geometries at their SDF rest poses in a disk.

    A disk also covers a full in-place rotation. Unknown geometry/pose frames fail
    explicitly rather than silently using a smaller footprint.
    """
    model_path = Path(model_path)
    model = ET.parse(model_path).getroot().find("model")
    points = []
    for link in model.findall("link"):
        for collision in link.findall("collision"):
            for pose in (link.find("pose"), collision.find("pose")):
                if pose is not None and pose.get("relative_to"):
                    raise ValueError("Planner footprint requires resolved SDF poses")
            geometry = collision.find("geometry")
            if (box := geometry.find("box")) is not None:
                half = np.fromstring(box.findtext("size"), sep=" ") / 2
                vertices = corners([-half, half])
            elif (cylinder := geometry.find("cylinder")) is not None:
                r = float(cylinder.findtext("radius"))
                half = np.array([r, r, float(cylinder.findtext("length")) / 2])
                vertices = corners([-half, half])
            elif (mesh := geometry.find("mesh")) is not None:
                uri = Path(mesh.findtext("uri").removeprefix("file://"))
                if not uri.is_absolute():
                    uri = model_path.parent / uri
                shape = trimesh.load(uri, force="mesh", process=False)
                scale = np.fromstring(mesh.findtext("scale") or "1 1 1", sep=" ")
                vertices = corners(shape.bounds) * scale
            else:
                raise ValueError("Unsupported original robot collision geometry")
            matrix = transform(link.findtext("pose")) @ transform(
                collision.findtext("pose")
            )
            points.append(vertices @ matrix[:3, :3].T + matrix[:3, 3])
    if not points:
        raise ValueError("No original robot collision footprint")
    return float(np.linalg.norm(np.concatenate(points)[:, :2], axis=1).max())


class LocalMap:
    def __init__(
        self,
        terrain,
        radius,
        *,
        extent=5.0,
        resolution=0.1,
        margin=0.1,
        max_slope=0.6,
        max_height_residual=0.16,
    ):
        self.terrain, self.radius = terrain, radius
        self.extent, self.resolution, self.margin = extent, resolution, margin
        self.max_slope, self.max_height_residual = max_slope, max_height_residual
        self.center = None
        self.clearance = None
        self.object_poses = {}

    def objects_changed(self):
        for name, pose in self.terrain.poses.items():
            previous = self.object_poses.get(name)
            if previous is None or np.linalg.norm(np.asarray(pose) - previous) > 0.03:
                return True
        return False

    def refresh(self, center):
        # Keep raster samples on a fixed world lattice while the rover moves.
        self.center = (
            np.floor(np.asarray(center[:2], dtype=float) / self.resolution)
            * self.resolution
        )
        axis = np.arange(
            -self.extent, self.extent + self.resolution / 2, self.resolution
        )
        x, y = np.meshgrid(axis, axis, indexing="xy")
        self.shape = x.shape
        self.lower = self.center - self.extent
        xy = np.stack([x.ravel(), y.ravel()], axis=1) + self.center
        height = self.terrain.rays.heights(xy).reshape(self.shape)
        finite = np.isfinite(height)
        # Missing rays and steep terrain both fail closed. No interpolation across voids.
        # The upstream spawn threshold is an UNNORMALIZED Sobel magnitude,
        # not a slope in m/m. Use a separate, explicit physical slope limit.
        # Wheel-scale smoothing separates terrain tilt from small mesh bumps;
        # retain a residual-height limit so smoothing cannot hide large steps.
        surface = gaussian_filter(np.where(finite, height, 0), 0.15 / self.resolution)
        dy, dx = np.gradient(surface, self.resolution)
        blocked = ~finite | (np.hypot(dx, dy) > self.max_slope)
        blocked |= np.abs(height - surface) > self.max_height_residual
        blocked |= (
            (xy < self.terrain.bounds[0] + 3).any(axis=1)
            | (xy > self.terrain.bounds[1] - 3).any(axis=1)
        ).reshape(self.shape)
        for name, obj in self.terrain.objects.items():
            matrix = transform(" ".join(map(str, self.terrain.poses[name])))
            world = corners(obj["bounds"]) @ matrix[:3, :3].T + matrix[:3, 3]
            lower, upper = world[:, :2].min(axis=0), world[:, :2].max(axis=0)
            # Conservative projected AABB; the distance field inflates it by the
            # full native robot radius below. Dynamic poses come from Gazebo.
            blocked |= (
                (xy >= lower - self.resolution).all(axis=1)
                & (xy <= upper + self.resolution).all(axis=1)
            ).reshape(self.shape)
        padded = np.pad(~blocked, 1, constant_values=False)
        self.clearance = distance_transform_edt(padded, sampling=self.resolution)[
            1:-1, 1:-1
        ]
        self.object_poses = {
            name: np.asarray(pose).copy() for name, pose in self.terrain.poses.items()
        }

    def valid(self, poses):
        if self.clearance is None:
            return False
        xy = np.atleast_2d(poses)[:, :2]
        if not np.isfinite(xy).all():
            return False
        indices = np.rint((xy - self.lower) / self.resolution).astype(int)
        if (indices < 0).any() or (indices >= np.array(self.shape[::-1])).any():
            return False
        # Extra diagonal accounts for rasterization and rounded query locations.
        required = self.radius + self.margin + np.sqrt(2) * self.resolution
        return bool((self.clearance[indices[:, 1], indices[:, 0]] > required).all())
