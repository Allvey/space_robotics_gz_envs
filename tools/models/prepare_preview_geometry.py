"""Extract centered COLLADA submeshes, with original normals and UVs.

Gazebo centers submeshes after applying the COLLADA node matrix and unit scale.
This intermediate is only for Blender previews; Gazebo loads the original DAE.
"""

from pathlib import Path

import collada
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / "assets/models/explorer_r2_base/meshes/r2.dae"
    dest = ROOT / ".cache/models/explorer_conversion/r2_geometry.npz"
    c = collada.Collada(str(source))
    arrays = {}
    for geometry in c.scene.objects("geometry"):
        name = geometry.original.name.removesuffix("Mesh")
        primitive = next(geometry.primitives())
        v = np.asarray(primitive.vertex, dtype=float) * c.assetInfo.unitmeter
        v -= (v.max(0) + v.min(0)) / 2
        arrays[name + "_v"] = v
        arrays[name + "_f"] = primitive.vertex_index
        arrays[name + "_uv"] = primitive.texcoordset[0][primitive.texcoord_indexset[0]]
        arrays[name + "_n"] = primitive.normal[primitive.normal_index]
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dest, **arrays)


if __name__ == "__main__":
    main()
