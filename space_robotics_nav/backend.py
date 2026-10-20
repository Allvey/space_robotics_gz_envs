"""Synchronous IPC to Gazebo; no surrogate dynamics or alternate simulator."""

import json
import os
import selectors
import subprocess
import tempfile
import time
from pathlib import Path

from .world import describe, runtime_world


class GazeboBackend:
    def __init__(self, cfg, *, gui=False, software=False):
        self.cfg = cfg
        self.spec = describe(cfg.world, cfg.assets)
        self.decimation = round(cfg.control_dt / self.spec["physics_dt"])
        if (
            self.decimation < 1
            or abs(self.decimation * self.spec["physics_dt"] - cfg.control_dt) > 1e-9
        ):
            raise ValueError(
                "Control period must be a multiple of the original physics step"
            )
        self.path = runtime_world(cfg, gui=gui)
        self.name = f"space-nav-{os.getpid()}-{self.path.stem[:8]}"
        self.closed = False
        self.process = self.selector = self.gui = self.gui_log = self.auth = None
        self.gui_config = None
        self.log = self.path.with_suffix(".log").open("w")
        self.buffer = b""
        robot = self.spec["robot"]
        command = [
            "docker",
            "run",
            "--rm",
            "-i",
            "--name",
            self.name,
            "--network",
            "none",
            "--env",
            f"GZ_PARTITION={self.name}",
            "--env",
            "GZ_IP=127.0.0.1",
            "--env",
            f"NAV_ROBOT={robot['name']}",
            "--env",
            f"NAV_BODY={robot['body']}",
            "--env",
            f"GZ_SIM_RESOURCE_PATH={Path(cfg.assets).resolve()}",
            "--volume",
            f"{Path(cfg.assets).resolve()}:{Path(cfg.assets).resolve()}:ro",
            "--volume",
            f"{self.path.parent}:{self.path.parent}:ro",
        ]
        try:
            if gui:
                command += self._display_options(software)
            self.process = subprocess.Popen(
                command + [cfg.image, str(self.path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
            )
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            if not self._receive(120).get("ready"):
                raise RuntimeError("Gazebo worker did not initialize")
        except BaseException:
            self.close()
            raise

    def _display_options(self, software):
        display = os.environ.get("DISPLAY")
        if not display:
            raise RuntimeError("No DISPLAY: run playback from the desktop terminal")
        cookies = subprocess.check_output(["xauth", "nlist", display], text=True)
        if not cookies.strip():
            raise RuntimeError(f"No X11 authorization found for {display}")
        fd, name = tempfile.mkstemp(prefix="space-nav-", suffix=".xauth")
        os.close(fd)
        self.auth = Path(name)
        subprocess.run(
            ["xauth", "-f", name, "nmerge", "-"],
            check=True,
            text=True,
            input="\n".join("ffff" + line[4:] for line in cookies.splitlines()) + "\n",
            capture_output=True,
        )
        result = [
            "--env",
            f"DISPLAY={display}",
            "--env",
            "XAUTHORITY=/tmp/navigation.xauth",
            "--env",
            "QT_QPA_PLATFORM=xcb",
            "--env",
            "QT_X11_NO_MITSHM=1",
            "--volume",
            "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "--volume",
            f"{name}:/tmp/navigation.xauth:ro",
        ]
        if software:
            result += ["--env", "LIBGL_ALWAYS_SOFTWARE=1"]
        else:
            result += [
                "--gpus",
                "all",
                "--env",
                "NVIDIA_DRIVER_CAPABILITIES=all",
                "--env",
                "__NV_PRIME_RENDER_OFFLOAD=1",
                "--env",
                "__GLX_VENDOR_LIBRARY_NAME=nvidia",
            ]
        if Path("/dev/dri").exists():
            result += ["--device", "/dev/dri:/dev/dri"]
        return result

    def _receive(self, timeout=60):
        deadline = time.monotonic() + timeout
        while True:
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if line.startswith(b"NAV_JSON "):
                    message = json.loads(line[9:])
                    if "error" in message:
                        raise RuntimeError(message["error"])
                    return message
                self.log.write(line.decode(errors="replace") + "\n")
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self.selector.select(remaining):
                raise TimeoutError(f"Gazebo response timed out; see {self.log.name}")
            data = os.read(self.process.stdout.fileno(), 65536)
            if not data:
                self.log.flush()
                raise RuntimeError(f"Gazebo exited; see {self.log.name}")
            self.buffer += data

    def request(self, message):
        self.process.stdin.write(json.dumps(message, allow_nan=False).encode() + b"\n")
        return self._receive()

    def reset(self, pose):
        return self.request(
            {
                "op": "reset",
                "pose": list(map(float, pose)),
                "steps": self.cfg.settle_steps,
            }
        )

    def step(self, action):
        # Same physical units as RLRoverLab: v in m/s, yaw rate in rad/s.
        # Original DiffDrive enforces its own velocity and acceleration limits.
        return self.request(
            {
                "op": "step",
                "velocity": list(map(float, action)),
                "steps": self.decimation,
            }
        )

    def goal(self, position, heading):
        return self.request(
            {"op": "goal", "pose": [*map(float, position), float(heading)]}
        )

    def start_gui(self, config):
        self.gui_config = config
        self.gui_log = self.path.with_suffix(".gui.log").open("w")
        self.gui = subprocess.Popen(
            [
                "docker",
                "exec",
                self.name,
                "gz",
                "sim",
                "-g",
                "--gui-config",
                str(config),
                "-v",
                "3",
            ],
            stdout=self.gui_log,
            stderr=self.gui_log,
        )

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.gui is not None and self.gui.poll() is None:
            self.gui.terminate()
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(b'{"op":"close"}\n')
                    self.process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    self.process.kill()
            subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)
            self.process.wait()
            self.process.stdin.close()
            self.process.stdout.close()
        if self.gui is not None:
            try:
                self.gui.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.gui.kill()
                self.gui.wait()
        if self.selector is not None:
            self.selector.close()
        self.log.close()
        if self.gui_log is not None:
            self.gui_log.close()
        for path in (self.auth, self.gui_config, self.path):
            if path is not None:
                Path(path).unlink(missing_ok=True)
