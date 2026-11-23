"""Capture the real Ogre2 rendering with an isolated, finite Gazebo server.

Requires gz.sim9 / gz.transport14 and an EGL-capable rendering environment.
Only the review world is temporary; vehicle visual resources are loaded as-is.
"""

import os
from pathlib import Path
import signal
import struct
import subprocess
import tempfile
import threading
import time
import uuid
import xml.etree.ElementTree as ET
import zlib

from gz.msgs11.image_pb2 import Image
from gz.transport14 import Node

ROOT = Path(__file__).resolve().parents[2]


def png(path, msg):
    if msg.step != msg.width * 3:
        raise ValueError(f"Expected packed RGB8, got {msg.width=} {msg.step=}")
    data = bytes(msg.data)

    def chunk(tag, payload):
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload))
        )

    rows = b"".join(
        b"\0" + data[y * msg.step : (y + 1) * msg.step] for y in range(msg.height)
    )
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", msg.width, msg.height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


def main():
    os.environ["GZ_PARTITION"] = "mining-preview-" + uuid.uuid4().hex
    os.environ["GZ_SIM_RESOURCE_PATH"] = str(ROOT / "assets/models")
    result = ROOT / ".cache/models/explorer_conversion"
    result.mkdir(parents=True, exist_ok=True)
    root = ET.parse(ROOT / "worlds/mining_vehicles.sdf").getroot()
    world = root.find("world")
    # This is a material/assembly review at the authored rest pose. Static
    # copies avoid falling or moving while shaders compile; physics has its
    # own independent test. Remove sensor workloads from the temporary copies.
    for include in world.findall("include"):
        name = include.findtext("name")
        model = ET.parse(ROOT / "assets/models" / name / "model.sdf").getroot()[0]
        model.find("pose").text = include.findtext("pose")
        ET.SubElement(model, "static").text = "true"
        for link in model.findall("link"):
            for sensor in link.findall("sensor"):
                link.remove(sensor)
        for plugin in model.findall("plugin"):
            model.remove(plugin)
        for uri in model.findall(".//uri"):
            if uri.text.startswith("meshes/"):
                uri.text = str(ROOT / "assets/models" / name / uri.text)
        world.remove(include)
        world.append(model)
    world.append(
        ET.fromstring(
            """<model name="review_camera"><static>true</static>
      <pose>-3.2 -4.5 2.5 0 0.34 0.98</pose><link name="camera_link">
      <sensor name="camera" type="camera"><topic>/review/image</topic>
      <always_on>true</always_on><update_rate>5</update_rate><camera>
      <horizontal_fov>0.95</horizontal_fov><image><width>1400</width><height>1000</height><format>R8G8B8</format></image>
      <clip><near>0.05</near><far>100</far></clip></camera></sensor></link></model>"""
        )
    )
    ready = threading.Event()
    frames = []

    def receive(msg):
        frames.append(msg)
        if len(frames) >= 5:
            ready.set()

    node = Node()
    node.subscribe(Image, "/review/image", receive)
    with tempfile.TemporaryDirectory(prefix="mining-render-") as temp:
        path = Path(temp) / "world.sdf"
        ET.ElementTree(root).write(path)
        with (result / "gazebo_render.log").open("w") as log:
            proc = subprocess.Popen(
                ["gz", "sim", "-s", "-r", "--headless-rendering", str(path)],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 55
                while not ready.wait(0.5):
                    if proc.poll() is not None:
                        raise RuntimeError(
                            "Gazebo exited before producing an image; see gazebo_render.log"
                        )
                    if time.monotonic() > deadline:
                        raise TimeoutError("No Gazebo review image within 55 seconds")
                png(result / "gazebo_preview.png", frames[-1])
                print("Captured Ogre2 preview", frames[-1].width, frames[-1].height)
            finally:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


if __name__ == "__main__":
    main()
