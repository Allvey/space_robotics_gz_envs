"""Thin large rocks and add stratified, ground-following native lunar pebbles."""

import math
import os

import bpy

factor = float(os.environ["MOSAIC_ROCK_DENSITY_FACTOR"])
terrain = bpy.data.node_groups["LunarTerrain"]
scatter = terrain.nodes["Distribute Points on Faces"]
assert scatter.bl_idname == "GeometryNodeDistributePointsOnFaces"
original = float(scatter.inputs["Density Max"].default_value)
scatter.inputs["Density Max"].default_value = original * factor
print(f"MOSAIC rock scatter Density Max: {original} -> {original * factor}", flush=True)

# Keep the upstream terrain and its crater/large-rock masks intact. Small
# traversable rocks have their own XY strata, including the boundary strata;
# they do not inherit the upstream scatter's empty boundary/flat-area masks.
spacing = float(os.environ.get("MOSAIC_PEBBLE_SPACING", "2"))
if spacing > 0:
    size = float(os.environ["MOSAIC_TERRAIN_SIZE"])
    count = math.ceil(size / spacing)
    cell = size / count
    nodes, links = terrain.nodes, terrain.links

    def node(kind, name):
        result = nodes.new(kind)
        result.name = f"MOSAIC {name}"
        return result

    output = nodes["Group Output"]
    ground = output.inputs["Geometry"].links[0].from_socket
    seed = nodes["Group Input"].outputs["Seed"]
    grid = node("GeometryNodeMeshGrid", "pebble strata")
    grid.inputs["Size X"].default_value = size - cell
    grid.inputs["Size Y"].default_value = size - cell
    grid.inputs["Vertices X"].default_value = count
    grid.inputs["Vertices Y"].default_value = count

    def random_vector(name, lower, upper, stream):
        result = node("FunctionNodeRandomValue", name)
        result.data_type = "FLOAT_VECTOR"
        result.inputs["Min"].default_value = lower
        result.inputs["Max"].default_value = upper
        offset = node("ShaderNodeMath", f"{name} seed")
        offset.operation = "ADD"
        links.new(seed, offset.inputs[0])
        offset.inputs[1].default_value = stream
        links.new(offset.outputs[0], result.inputs["Seed"])
        return result.outputs["Value"]

    jitter = random_vector(
        "stratum jitter",
        (-0.35 * cell, -0.35 * cell, 0),
        (0.35 * cell, 0.35 * cell, 0),
        901,
    )
    positions = node("GeometryNodeSetPosition", "jittered strata")
    links.new(grid.outputs["Mesh"], positions.inputs["Geometry"])
    links.new(jitter, positions.inputs["Offset"])
    position = node("GeometryNodeInputPosition", "pebble XY")
    origin = node("ShaderNodeVectorMath", "ray origin")
    origin.operation = "ADD"
    links.new(position.outputs["Position"], origin.inputs[0])
    origin.inputs[1].default_value = (0, 0, 100)
    ray = node("GeometryNodeRaycast", "project onto original ground")
    links.new(ground, ray.inputs["Target Geometry"])
    links.new(origin.outputs[0], ray.inputs["Source Position"])
    ray.inputs["Ray Direction"].default_value = (0, 0, -1)
    ray.inputs["Ray Length"].default_value = 200
    projected = node("GeometryNodeSetPosition", "grounded strata")
    links.new(positions.outputs["Geometry"], projected.inputs["Geometry"])
    links.new(ray.outputs["Hit Position"], projected.inputs["Position"])

    # Native rock shape, normalized once so the height limit is explicit.
    rock = node("GeometryNodeGroup", "native pebble shape")
    rock.node_tree = bpy.data.node_groups["LunarRock"]
    rock.inputs["Subdivisions"].default_value = 2
    links.new(seed, rock.inputs["Seed"])
    bounds = node("GeometryNodeBoundBox", "native pebble bounds")
    links.new(rock.outputs["Geometry"], bounds.inputs["Geometry"])
    extent = node("ShaderNodeVectorMath", "native pebble extent")
    extent.operation = "SUBTRACT"
    links.new(bounds.outputs["Max"], extent.inputs[0])
    links.new(bounds.outputs["Min"], extent.inputs[1])
    local_position = node("GeometryNodeInputPosition", "native pebble position")
    relative = node("ShaderNodeVectorMath", "native pebble relative")
    relative.operation = "SUBTRACT"
    links.new(local_position.outputs["Position"], relative.inputs[0])
    links.new(bounds.outputs["Min"], relative.inputs[1])
    normalized = node("ShaderNodeVectorMath", "unit pebble")
    normalized.operation = "DIVIDE"
    links.new(relative.outputs[0], normalized.inputs[0])
    links.new(extent.outputs[0], normalized.inputs[1])
    centered = node("ShaderNodeVectorMath", "buried pebble base")
    centered.operation = "SUBTRACT"
    links.new(normalized.outputs[0], centered.inputs[0])
    centered.inputs[1].default_value = (0.5, 0.5, 0.15)
    shape = node("GeometryNodeSetPosition", "normalized native pebble")
    links.new(rock.outputs["Geometry"], shape.inputs["Geometry"])
    links.new(centered.outputs[0], shape.inputs["Position"])

    # Keep each rock's stratum ID for geometry validation before export.
    index = node("GeometryNodeInputIndex", "pebble cell index")
    identifier = node("ShaderNodeMath", "pebble ID")
    identifier.operation = "ADD"
    links.new(index.outputs["Index"], identifier.inputs[0])
    identifier.inputs[1].default_value = 1
    tagged = node("GeometryNodeStoreNamedAttribute", "pebble cell attribute")
    tagged.data_type = "INT"
    tagged.domain = "POINT"
    tagged.inputs["Name"].default_value = "mosaic_pebble_cell"
    links.new(projected.outputs["Geometry"], tagged.inputs["Geometry"])
    links.new(identifier.outputs[0], tagged.inputs["Value"])

    rotation = node("FunctionNodeAlignEulerToVector", "follow ground normal")
    rotation.axis = "Z"
    links.new(ray.outputs["Hit Normal"], rotation.inputs["Vector"])
    yaw = random_vector("pebble yaw", (0, 0, -math.pi), (0, 0, math.pi), 902)
    links.new(yaw, rotation.inputs["Rotation"])
    instances = node("GeometryNodeInstanceOnPoints", "surface pebbles")
    links.new(tagged.outputs["Geometry"], instances.inputs["Points"])
    links.new(ray.outputs["Is Hit"], instances.inputs["Selection"])
    links.new(shape.outputs["Geometry"], instances.inputs["Instance"])
    links.new(rotation.outputs["Rotation"], instances.inputs["Rotation"])
    scales = random_vector(
        "pebble dimensions",
        (0.12, 0.12, 0.03),
        (0.30, 0.30, 0.07),
        903,
    )
    links.new(scales, instances.inputs["Scale"])
    realized = node("GeometryNodeRealizeInstances", "physical pebbles")
    links.new(instances.outputs["Instances"], realized.inputs["Geometry"])
    joined = node("GeometryNodeJoinGeometry", "terrain with uniform pebbles")
    links.new(ground, joined.inputs["Geometry"])
    links.new(realized.outputs["Geometry"], joined.inputs["Geometry"])
    links.new(joined.outputs["Geometry"], output.inputs["Geometry"])
    print(
        f"MOSAIC pebbles: {count}x{count} XY strata, {cell:g} m spacing, "
        "35% XY jitter, native LunarRock, 3–7 cm height (15% buried)",
        flush=True,
    )
