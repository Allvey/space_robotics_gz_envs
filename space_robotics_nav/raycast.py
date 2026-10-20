"""CPU mesh rays with the upstream 100 m range and yaw-aligned grid."""

from threading import Lock

import numpy as np
import warp as wp

MISS = wp.constant(float("inf"))


@wp.kernel
def cast(
    mesh: wp.uint64,
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    i = wp.tid()
    hit = wp.mesh_query_ray(mesh, origins[i], directions[i], 100.0)
    output[i] = hit.t if hit.result else MISS


class Raycaster:
    def __init__(self, mesh):
        wp.init()
        self.points = wp.array(
            np.asarray(mesh.vertices, dtype=np.float32), dtype=wp.vec3, device="cpu"
        )
        self.indices = wp.array(
            np.asarray(mesh.faces, dtype=np.int32).ravel(), dtype=wp.int32, device="cpu"
        )
        self.mesh = wp.Mesh(points=self.points, indices=self.indices)
        self.lock = Lock()
        wp.load_module(module=__name__, device="cpu")

    def distances(self, origins, direction):
        origins = np.asarray(origins, dtype=np.float32)
        with self.lock:
            points = wp.array(origins, dtype=wp.vec3, device="cpu")
            directions = wp.array(
                np.broadcast_to(direction, origins.shape).copy().astype(np.float32),
                dtype=wp.vec3,
                device="cpu",
            )
            output = wp.empty(len(origins), dtype=wp.float32, device="cpu")
            wp.launch(
                cast,
                dim=len(origins),
                inputs=[self.mesh.id, points, directions, output],
                device="cpu",
            )
            return output.numpy()

    def heights(self, xy, z=50.0):
        return z - self.distances(
            np.column_stack([xy, np.full(len(xy), z)]), [0, 0, -1]
        )
