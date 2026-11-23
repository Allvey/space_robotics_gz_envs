"""Build detailed mount meshes and preview the actual SDF vehicles in Blender.

blender --background --factory-startup --python tools/models/render_explorer_variants.py
The NPZ is derived from the original COLLADA scene by prepare_preview_geometry.py.
"""

from pathlib import Path
import math
import xml.etree.ElementTree as ET

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "assets/models"
GEOMETRY = ROOT / ".cache/models/explorer_conversion/r2_geometry.npz"
PI = math.pi


def material(name, color, metal=0.0, rough=0.4):
    m = bpy.data.materials.new(name)
    m.diffuse_color = (*color, 1)
    m.use_nodes = True
    p = m.node_tree.nodes.get("Principled BSDF")
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Metallic"].default_value = metal
    p.inputs["Roughness"].default_value = rough
    # Baked material parameters travel to GLB. Subtle roughness, not huge noise.
    return m


def finish(o, name, mat, bevel=0):
    o.name = name
    o.data.materials.append(mat)
    if bevel:
        m = o.modifiers.new("Machined edge radii", "BEVEL")
        m.width = bevel
        m.segments = 3
        m = o.modifiers.new("Weighted corner normals", "WEIGHTED_NORMAL")
        m.keep_sharp = True
    return o


def box(name, pos, size, mat, bevel=0.002):
    bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
    o = bpy.context.object
    o.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    return finish(o, name, mat, bevel)


def cylinder(name, pos, radius, depth, mat, axis="Z", vertices=48):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=pos
    )
    o = bpy.context.object
    if axis == "Y":
        o.rotation_euler.x = PI / 2
    elif axis == "X":
        o.rotation_euler.y = PI / 2
    for p in o.data.polygons:
        p.use_smooth = len(p.vertices) == 4
    return finish(o, name, mat, 0.0008)


def tube(name, points, radius, mat):
    c = bpy.data.curves.new(name, "CURVE")
    c.dimensions = "3D"
    c.bevel_depth = radius
    c.bevel_resolution = 3
    s = c.splines.new("BEZIER")
    s.bezier_points.add(len(points) - 1)
    for v, xyz in zip(s.bezier_points, points):
        v.co = xyz
        v.handle_left_type = v.handle_right_type = "AUTO"
    o = bpy.data.objects.new(name, c)
    bpy.context.collection.objects.link(o)
    o.data.materials.append(mat)
    return o


def label(body, pos, rotation, size, mat):
    c = bpy.data.curves.new(body, "FONT")
    c.body = body
    c.size = size
    c.align_x = "CENTER"
    c.align_y = "CENTER"
    c.extrude = 0.00015
    o = bpy.data.objects.new(body, c)
    bpy.context.collection.objects.link(o)
    o.location, o.rotation_euler = pos, rotation
    c.materials.append(mat)


def bolt(pos, axis="Z"):
    cylinder("M8 washer", pos, 0.009, 0.0015, steel, axis)
    direction = {"Z": (0, 0, 1), "Y": (0, 1, 0), "X": (1, 0, 0)}[axis]
    p = tuple(x + 0.003 * d for x, d in zip(pos, direction))
    cylinder("M8 hex head", p, 0.006, 0.004, steel, axis, 6)


def build_adapter(kind):
    bpy.ops.object.select_all(action="DESELECT")
    before = set(bpy.data.objects)
    for y in [-0.24, 0.24]:
        box("Anodized chassis rail", (0, y, 0), (0.76, 0.055, 0.065), charcoal)
        # Extrusion grooves, end caps and eight machined mounting feet.
        for z in [-0.012, 0.012]:
            box(
                "Extrusion slot",
                (0, y + (0.028 if y > 0 else -0.028), z),
                (0.73, 0.001, 0.004),
                black,
                0.0005,
            )
        for x in [-0.36, 0.36]:
            box("Rail end cap", (x, y, 0), (0.012, 0.058, 0.068), black)
        for x in [-0.29, 0.0, 0.29]:
            box("Mount foot", (x, y, -0.035), (0.085, 0.09, 0.012), steel)
            bolt((x - 0.025, y, -0.027))
            bolt((x + 0.025, y, -0.027))
    for x in [-0.29, 0, 0.29]:
        box("Crossmember", (x, 0, 0.01), (0.055, 0.54, 0.045), charcoal)
        for y in [-0.24, 0.24]:
            box("Angle bracket", (x, y, 0.036), (0.075, 0.075, 0.007), steel)
            bolt((x, y, 0.043))
    for y in [-0.278, 0.278]:
        box("Serial plate", (0.03, y, -0.002), (0.29, 0.003, 0.043), steel, 0.001)
        rot = (PI / 2, 0, 0) if y < 0 else (PI / 2, 0, PI)
        label(
            "R2 / EX-01" if kind == "excavator" else "R2 / HA-01",
            (0.03, y + (-0.002 if y < 0 else 0.002), 0.002),
            rot,
            0.022,
            black,
        )
        for x in [-0.10, 0.16]:
            cylinder("Plate rivet", (x, y, -0.002), 0.0025, 0.006, charcoal, "Y", 16)
    before_camera = set(bpy.data.objects)
    box("Rear work camera mast", (0.38, 0, 0.25), (0.025, 0.03, 0.42), charcoal)
    box(
        "Work camera weatherproof housing",
        (0.35, 0, 0.46),
        (0.075, 0.16, 0.055),
        steel,
        0.007,
    )
    glass = material("Camera coated optical glass", (0.018, 0.075, 0.12), 0.5, 0.10)
    for y in [-0.052, 0.052]:
        cylinder("Lens hood", (0.309, y, 0.46), 0.019, 0.014, black, "X")
        cylinder("Lens glass", (0.300, y, 0.46), 0.014, 0.002, glass, "X")
    cylinder("IR projector", (0.309, 0, 0.46), 0.009, 0.012, black, "X")
    for z in [0.10, 0.20, 0.30, 0.40]:
        box("Mast harness clamp", (0.397, 0, z), (0.007, 0.04, 0.009), steel, 0.001)
    tube(
        "Camera cable",
        [(0.25, 0.05, 0.05), (0.4, 0.04, 0.08), (0.4, 0.04, 0.42), (0.375, 0.04, 0.46)],
        0.003,
        black,
    )
    if kind == "hauler":
        for o in set(bpy.data.objects) - before_camera:
            o.location.y -= 0.47
        box("Camera outrigger", (0.38, -0.355, 0.045), (0.05, 0.25, 0.04), charcoal)
        for y in [-0.25, -0.45]:
            bolt((0.38, y, 0.069))
        tube(
            "Outrigger cable loom",
            [(0.25, -0.17, 0.05), (0.3, -0.25, 0.07), (0.3, -0.4, 0.05)],
            0.003,
            black,
        )
    if kind == "excavator":
        box("Turntable baseplate", (0, 0, 0.045), (0.43, 0.43, 0.035), steel, 0.008)
        cylinder("Load bearing pedestal", (0, 0, 0.078), 0.155, 0.055, charcoal)
        cylinder("Machined bearing flange", (0, 0, 0.104), 0.173, 0.012, steel)
        for k in range(12):
            a = k * 2 * PI / 12
            bolt((0.16 * math.cos(a), 0.16 * math.sin(a), 0.112))
        for x in [-0.188, 0.188]:
            for y in [-0.188, 0.188]:
                bolt((x, y, 0.065))
        box(
            "Sealed motor controller",
            (0.285, 0, 0.085),
            (0.13, 0.22, 0.10),
            charcoal,
            0.008,
        )
        for x in np.linspace(0.235, 0.335, 7):
            box("Controller cooling fin", (x, 0, 0.144), (0.006, 0.19, 0.018), steel)
        for side in [-1, 1]:
            tube(
                "Arm power harness",
                [
                    (0.24, side * 0.09, 0.08),
                    (0.19, side * 0.14, 0.08),
                    (0.14, side * 0.14, 0.10),
                ],
                0.006,
                black,
            )
            cylinder(
                "Sealed cable gland",
                (0.222, side * 0.09, 0.08),
                0.011,
                0.025,
                steel,
                "X",
            )
    else:
        for y in [-0.21, 0.21]:
            box("Raised bed support", (0, y, 0.055), (0.80, 0.045, 0.08), charcoal)
            for x in [-0.3, 0.0, 0.3]:
                box("Elastomer bed pad", (x, y, 0.097), (0.07, 0.06, 0.01), black)
            cylinder("Hinge bearing block", (-0.45, y, 0.09), 0.045, 0.075, steel, "Y")
            cylinder(
                "Sealed tip drive", (-0.45, y * 1.25, 0.09), 0.063, 0.075, charcoal, "Y"
            )
            for k in range(8):
                a = k * 2 * PI / 8
                cylinder(
                    "Gearcase fastener",
                    (
                        -0.45 + 0.05 * math.cos(a),
                        y * 1.25 + (-0.04 if y < 0 else 0.04),
                        0.09 + 0.05 * math.sin(a),
                    ),
                    0.0035,
                    0.008,
                    steel,
                    "Y",
                    6,
                )
        box("Bed power module", (0.26, 0, 0.048), (0.15, 0.30, 0.06), charcoal)
        tube(
            "Protected tilt drive cable",
            [(0.26, 0.12, 0.05), (0, 0.18, 0.04), (-0.38, 0.20, 0.06)],
            0.006,
            black,
        )
    objects = set(bpy.data.objects) - before
    for o in objects:
        o.select_set(True)
    bpy.context.view_layer.objects.active = next(o for o in objects if o.type == "MESH")
    bpy.ops.object.convert(target="MESH")
    bpy.ops.export_scene.gltf(
        filepath=str(MODELS / f"lunar_{kind}/meshes/payload_frame.glb"),
        export_format="GLB",
        use_selection=True,
        use_active_scene=True,
    )
    for o in objects:
        if o.name in bpy.data.objects:
            bpy.data.objects.remove(o, do_unlink=True)
    # Conversion may replace curve objects, so clear the temporary scene.
    for o in list(bpy.context.scene.objects):
        bpy.data.objects.remove(o, do_unlink=True)


def transform(text):
    a = [float(v) for v in (text or "0 0 0 0 0 0").split()]
    return Matrix.Translation(Vector(a[:3])) @ Euler(a[3:], "XYZ").to_matrix().to_4x4()


def original_material():
    m = material("EXPLORER R2 / ORIGINAL PBR", (1, 1, 1))
    p = m.node_tree.nodes.get("Principled BSDF")
    for name, socket in [
        ("Albedo", "Base Color"),
        ("Metalness", "Metallic"),
        ("Roughness", "Roughness"),
    ]:
        n = m.node_tree.nodes.new("ShaderNodeTexImage")
        n.image = bpy.data.images.load(
            str(MODELS / f"explorer_r2_base/materials/textures/R2_{name}.jpg"),
            check_existing=True,
        )
        if name != "Albedo":
            n.image.colorspace_settings.name = "Non-Color"
        m.node_tree.links.new(n.outputs["Color"], p.inputs[socket])
    return m


def assembly(name, offset, original=False, joint_values=None):
    model = ET.parse(MODELS / name / "model.sdf").getroot().find("model")
    matrices = {"__model__": transform(" ".join(str(v) for v in offset))}
    links = {l.get("name"): l for l in model.findall("link")}
    joint_values = joint_values or {}
    joints = {j.findtext("child"): j for j in model.findall("joint")}

    def link_matrix(key):
        if key in matrices:
            return matrices[key]
        l = links[key]
        p = l.find("pose")
        ref = p.get("relative_to", "__model__") if p is not None else "__model__"
        result = link_matrix(ref) @ transform(p.text if p is not None else None)
        j = joints.get(key)
        if j is not None and j.get("name") in joint_values:
            axis = Vector([float(x) for x in j.findtext("axis/xyz").split()])
            result = result @ Matrix.Rotation(joint_values[j.get("name")], 4, axis)
        matrices[key] = result
        return result

    root = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(root)
    for key, l in links.items():
        lm = link_matrix(key)
        for v in l.findall("visual"):
            geometry = v.find("geometry")
            mesh = geometry.find("mesh")
            vm = transform(v.findtext("pose"))
            if mesh is not None and mesh.find("submesh") is not None:
                sub = mesh.findtext("submesh/name")
                data = bpy.data.meshes.new(sub)
                data.from_pydata(r2[sub + "_v"].tolist(), [], r2[sub + "_f"].tolist())
                data.update()
                uv = data.uv_layers.new(name="Original atlas UV")
                uv.data.foreach_set("uv", r2[sub + "_uv"].reshape(-1))
                for p in data.polygons:
                    p.use_smooth = True
                data.normals_split_custom_set(r2[sub + "_n"].reshape(-1, 3).tolist())
                o = bpy.data.objects.new(f"{name}/{sub}", data)
                bpy.context.collection.objects.link(o)
                o.data.materials.append(r2_mat)
                o.matrix_world = lm @ vm
                o.parent = root
            elif mesh is not None:
                before = set(bpy.data.objects)
                bpy.ops.import_scene.gltf(
                    filepath=str(MODELS / name / mesh.findtext("uri"))
                )
                scale = [float(x) for x in mesh.findtext("scale", "1 1 1").split()]
                # Blender's importer already restores glTF Y-up to Z-up.
                matrix = lm @ Matrix.Diagonal((*scale, 1))
                for o in set(bpy.data.objects) - before:
                    if o.parent is None:
                        o.matrix_world = matrix @ o.matrix_world
                        o.parent = root
            elif geometry.find("cylinder") is not None:
                c = geometry.find("cylinder")
                o = cylinder(
                    f"{name}/central articulation shaft",
                    (0, 0, 0),
                    float(c.findtext("radius")),
                    float(c.findtext("length")),
                    steel,
                )
                o.matrix_world = lm @ vm
                o.parent = root
    return root


def camera_at(pos, target, scale):
    bpy.ops.object.camera_add(location=pos)
    c = bpy.context.object
    c.rotation_euler = (Vector(target) - c.location).to_track_quat("-Z", "Y").to_euler()
    c.data.type = "ORTHO"
    c.data.ortho_scale = scale
    c.data.lens = 45
    bpy.context.scene.camera = c
    return c


def area(pos, power, size, color):
    bpy.ops.object.light_add(type="AREA", location=pos)
    o = bpy.context.object
    o.data.energy, o.data.size, o.data.color = power, size, color
    o.rotation_euler = (
        (Vector((0, 0, 0.5)) - o.location).to_track_quat("-Z", "Y").to_euler()
    )


if __name__ == "__main__":
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    charcoal = material("Graphite anodized aluminum", (0.09, 0.105, 0.12), 0.75, 0.32)
    black = material("EPDM seals and cable jackets", (0.012, 0.016, 0.019), 0.0, 0.55)
    steel = material("Bead blasted stainless steel", (0.48, 0.52, 0.55), 0.85, 0.27)
    build_adapter("excavator")
    build_adapter("hauler")
    r2 = np.load(GEOMETRY)
    r2_mat = original_material()
    assembly("lunar_excavator", (0, -0.85, 0.405, 0, 0, 0))
    assembly("lunar_hauler", (0.25, 0.85, 0.405, 0, 0, 0))
    floor = material("Studio slate", (0.14, 0.17, 0.21), 0.05, 0.75)
    box("Studio floor", (0, 0, -0.025), (200, 200, 0.05), floor, 0)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 48
    scene.cycles.use_denoising = True
    scene.world.color = (0.24, 0.24, 0.24)
    scene.view_settings.view_transform = "AgX"
    area((-3, -4, 6), 1600, 5, (1, 0.91, 0.8))
    area((2, 3, 4), 1800, 4, (0.76, 0.86, 1))
    area((4, -1, 3), 900, 3, (1, 1, 1))
    camera = camera_at((-4.6, -6, 3.5), (-0.35, 0, 0.65), 4.5)
    scene.render.resolution_x, scene.render.resolution_y = 2000, 1400
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    bpy.ops.file.pack_all()
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(MODELS / "mining_vehicles.blend"))
    scene.render.filepath = str(MODELS / "mining_vehicles.png")
    bpy.ops.render.render(write_still=True)
    camera.location = (4.6, -5, 3.3)
    camera.rotation_euler = (
        (Vector((-0.25, 0, 0.65)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    scene.render.filepath = str(MODELS / "mining_vehicles_front.png")
    bpy.ops.render.render(write_still=True)
