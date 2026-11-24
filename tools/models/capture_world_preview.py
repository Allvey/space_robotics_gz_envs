"""Capture an isolated Ogre2 camera view of a generated mission world."""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from gz.msgs11.image_pb2 import Image
from gz.transport14 import Node

from capture_gazebo_preview import png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pose", required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["GZ_PARTITION"] = "mosaic-review-" + uuid.uuid4().hex
    root = ET.parse(args.world).getroot()
    world = root.find("world")
    # Only this review copy is frozen; mission physics is tested separately.
    for include in world.findall("include"):
        node = include.find("static")
        if node is None:
            node = ET.SubElement(include, "static")
        node.text = "true"
        for plugin in list(include.findall("plugin")):
            include.remove(plugin)
        source = Path(include.findtext("uri"))
        model = ET.parse(source).getroot().find("model")
        for light in model.findall(".//light"):
            marker = light.find("visualize")
            if marker is None:
                marker = ET.SubElement(light, "visualize")
            marker.text = "false"
        if model.findall(".//sensor"):
            model.set("name", include.findtext("name"))
            for link in model.findall("link"):
                for sensor in list(link.findall("sensor")):
                    link.remove(sensor)
            for uri in model.findall(".//uri"):
                value = Path(uri.text.removeprefix("file://"))
                if not value.is_absolute():
                    uri.text = str(source.parent / value)
            for tag, value in [("pose", include.findtext("pose")), ("static", "true")]:
                node = model.find(tag)
                if node is None:
                    node = ET.SubElement(model, tag)
                node.text = value
            for plugin in list(model.findall("plugin")):
                model.remove(plugin)
            world.remove(include)
            world.append(model)
    if world.find("plugin[@name='gz::sim::systems::Sensors']") is None:
        plugin = ET.SubElement(
            world,
            "plugin",
            filename="gz-sim-sensors-system",
            name="gz::sim::systems::Sensors",
        )
        ET.SubElement(plugin, "render_engine").text = "ogre2"
    camera = ET.fromstring(
        """<model name="review_camera"><static>true</static>
      <link name="camera_link"><sensor name="camera" type="camera">
      <topic>/review/image</topic><always_on>true</always_on><update_rate>2</update_rate>
      <camera><horizontal_fov>1.05</horizontal_fov><image><width>1600</width>
      <height>1100</height><format>R8G8B8</format></image><clip><near>0.05</near>
      <far>250</far></clip></camera></sensor></link></model>"""
    )
    ET.SubElement(camera, "pose").text = args.pose
    world.append(camera)
    review = args.output.with_suffix(".sdf")
    ET.ElementTree(root).write(review)
    ready, frames = threading.Event(), []

    def receive(msg):
        frames[:] = [msg]
        ready.set()

    node = Node()
    node.subscribe(Image, "/review/image", receive)
    with args.output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            ["gz", "sim", "-s", "-r", "--headless-rendering", str(review)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + 120
            while not ready.wait(0.5):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(
                        f"No Ogre2 frame; see {args.output.with_suffix('.log')}"
                    )
            # Let textures and shaders settle before capturing the final frame.
            time.sleep(3)
            png(args.output, frames[-1])
            print(args.output)
        finally:
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    main()
