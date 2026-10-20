#!/usr/bin/env python3
"""Locate the ORIGINAL project's assets. No RLRoverLab simulation assets are used."""
import argparse
import json
import os
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from space_robotics_nav.world import describe, resolve_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default="andrejorsula/space_robotics_gz_envs:latest")
    args = p.parse_args()
    assets = ROOT / "assets/cache"
    assets.mkdir(parents=True, exist_ok=True)
    if not all(
        (assets / name / "model.sdf").exists()
        for name in ("lunar_surface0", "martian_surface0")
    ):
        container = subprocess.check_output(
            ["docker", "create", args.image], text=True
        ).strip()
        try:
            subprocess.run(
                ["docker", "cp", f"{container}:/root/ws/assets/cache/.", str(assets)],
                check=True,
            )
        finally:
            subprocess.run(
                ["docker", "rm", container], check=True, stdout=subprocess.DEVNULL
            )
    for world in ("moon", "mars"):
        uri = (
            ET.parse(ROOT / "worlds" / f"{world}.sdf")
            .getroot()
            .findtext("world/include/uri")
            .strip()
        )
        try:
            resolve_model(uri, assets)
            continue
        except FileNotFoundError:
            pass
        with urllib.request.urlopen(uri, timeout=60) as response:
            metadata = json.load(response)
        version = metadata["version"]
        name = metadata["name"]
        parts = unquote(urlparse(uri).path).strip("/").split("/")
        target = (
            assets
            / "fuel/fuel.gazebosim.org"
            / parts[1].lower()
            / "models"
            / name.lower()
            / str(version)
        )
        target.mkdir(parents=True, exist_ok=True)
        archive = assets / "fuel" / f"{name}-{version}.zip"
        print(
            f"Downloading the project's original {name}, Fuel version {version}",
            flush=True,
        )
        urllib.request.urlretrieve(f"{uri}/{version}/{name}.zip", archive)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                if (
                    not (target / entry.filename)
                    .resolve()
                    .is_relative_to(target.resolve())
                ):
                    raise ValueError("Unsafe archive member")
            z.extractall(target)
        archive.unlink()
        resolve_model(uri, assets)
    provenance = {w: describe(w, str(assets)) for w in ("moon", "mars")}
    (assets / "rl-native-assets.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    print("Original Gazebo project assets are ready.")


if __name__ == "__main__":
    main()
