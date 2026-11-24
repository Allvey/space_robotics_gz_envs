"""Run upstream srb_assets scripts, retaining mesh detail with tunable rock scatter."""

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def lunar_surface(
    output,
    seed,
    size,
    *,
    image,
    texture_resolution=None,
    rock_density_factor=0.2,
    pebble_spacing=2.0,
):
    if not math.isfinite(rock_density_factor) or not 0 <= rock_density_factor <= 2:
        raise ValueError("Rock density factor must be between 0 and 2")
    if not math.isfinite(pebble_spacing) or not (
        pebble_spacing == 0 or 1 <= pebble_spacing <= size / 2
    ):
        raise ValueError(
            "Pebble spacing must be zero (disabled) or between 1 m and half the map size"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Match scripts/procgen_assets.bash's 40 m / 4096 texel density.
    texture_resolution = texture_resolution or 2 ** math.ceil(
        math.log2(4096 * size / 40)
    )
    if texture_resolution < 4096 or texture_resolution & (texture_resolution - 1):
        raise ValueError("Texture resolution must be a power of two >=4096")
    source = "/root/ws/assets/srb_assets"
    scripts = [
        f"{source}/scripts/blender/procgen_assets.py",
        f"{source}/model/terrain/lunar_surface_procgen/geometry.py",
        f"{source}/model/terrain/lunar_surface_procgen/material.py",
    ]
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image], text=True
    ).strip()
    hashes = subprocess.check_output(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--entrypoint",
            "sha256sum",
            image,
            *scripts,
        ],
        text=True,
    )
    parameters = dict(
        density=0.08,
        scale=[size, size, 4.0],
        flat_area_size=2.0,
        rock_mesh_boolean=False,
    )
    profile = Path(__file__).with_name("lunar_profile.py").resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--network",
        "none",
        "--env",
        "PYTHONUNBUFFERED=1",
        "--env",
        f"MOSAIC_ROCK_DENSITY_FACTOR={rock_density_factor}",
        "--env",
        f"MOSAIC_TERRAIN_SIZE={size}",
        "--env",
        f"MOSAIC_PEBBLE_SPACING={pebble_spacing}",
        "--env",
        f"PROCGEN_UID={os.getuid()}",
        "--env",
        f"PROCGEN_GID={os.getgid()}",
        "--volume",
        f"{output}:/output",
        "--volume",
        f"{profile}:/mosaic-lunar-profile.py:ro",
        "--entrypoint",
        "bash",
        image,
        "-c",
        '/root/blender/blender "$@"; result=$?; chown -R "$PROCGEN_UID:$PROCGEN_GID" /output; exit "$result"',
        "mosaic-procgen",
        "--factory-startup",
        "--background",
        "--offline-mode",
        "--enable-autoexec",
        "--python-exit-code",
        "1",
        "--python",
        scripts[0],
        "--",
        "--autorun_scripts",
        scripts[1],
        "/mosaic-lunar-profile.py",
        scripts[2],
        "--name",
        "lunar_surface",
        "--ext",
        "sdf",
        "--overwrite_min_age",
        "-1",
        "--seed",
        str(seed),
        "--num_assets",
        "1",
        "--geometry_nodes",
        json.dumps({"LunarTerrain": parameters}),
        "--material",
        "LunarSurface",
        "--texture_resolution",
        str(texture_resolution),
        "--render_samples",
        "2",
        "--render_thumbnail",
        "--thumbnail_resolution",
        "1024",
        "--outdir",
        "/output",
    ]
    print(
        f"Generating original srb_assets lunar terrain; Blender log: {output / 'blender.log'}",
        file=sys.stderr,
        flush=True,
    )
    with (output / "blender.log").open("w") as log:
        try:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Original Blender generator failed; see {output / 'blender.log'}"
            ) from error
    model = output / f"lunar_surface{seed}" / "model.sdf"
    if not model.is_file():
        raise RuntimeError(
            f"Blender did not export {model}; see {output / 'blender.log'}"
        )
    metadata = dict(
        generator="srb_assets/LunarTerrain",
        image=image,
        image_id=image_id,
        source_sha256=hashes.splitlines(),
        geometry_nodes=parameters,
        rock_density_factor=rock_density_factor,
        pebbles=dict(
            distribution="jittered_xy_strata_projected_onto_original_mesh",
            spacing=pebble_spacing,
            strata_per_axis=math.ceil(size / pebble_spacing) if pebble_spacing else 0,
            jitter_fraction=0.35,
            diameter_range_m=[0.12, 0.30],
            height_range_m=[0.03, 0.07],
            buried_fraction=0.15,
            shape="srb_assets/LunarRock",
        ),
        profile_sha256=hashlib.sha256(profile.read_bytes()).hexdigest(),
        texture_resolution=texture_resolution,
        render_samples=2,
        seed=seed,
        command=command,
        model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
    )
    (output / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return model, metadata
