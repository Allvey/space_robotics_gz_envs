"""Desktop mission control: shared terrain map, interactive POIs and robot status.

Uses Tk from the Python distribution. No web service or simulator-specific UI
dependency. Reads the view model and appends commands through the file adapter.
"""

import argparse
import json
import math
from pathlib import Path
import time
import tkinter as tk
from tkinter import font, messagebox, simpledialog, ttk

import cv2

from .visualization import STATUS_COLORS


class ControlPanel:
    def __init__(self, root, session):
        self.root, self.session = root, Path(session)
        self.state, self.image, self.image_key = None, None, None
        self.selected_poi, self.robot_name = None, tk.StringVar()
        self.adding, self.show_local, self.show_trails = (
            tk.BooleanVar(),
            tk.BooleanVar(),
            tk.BooleanVar(value=True),
        )
        self.kind = tk.StringVar(value="move")
        self.zoom, self.pan = 1, [0, 0]
        self.drag_start, self.hold, self.last_hold = None, None, 0
        # Some Conda Tk builds expose only X11 fonts. Select an installed family
        # instead of letting a missing TrueType font fall back to bitmap "fixed".
        families = set(font.families(root))
        self.font = next(
            (
                name
                for name in ("DejaVu Sans", "latin modern sans", "texgyreheros")
                if name in families
            ),
            "sans",
        )
        self.mono = next(
            (
                name
                for name in (
                    "DejaVu Sans Mono",
                    "latin modern typewriter",
                    "courier 10 pitch",
                )
                if name in families
            ),
            "fixed",
        )
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            font.nametofont(name).configure(family=self.font, size=10)
        root.title("MOSAIC · Lunar Mission Control")
        root.geometry("1480x980")
        root.minsize(1220, 980)
        root.configure(bg="#111923")
        style = ttk.Style(root)
        style.theme_use("clam")
        style.configure(
            ".",
            background="#18232e",
            foreground="#dbe6ef",
            fieldbackground="#101a24",
            font=(self.font, 10),
        )
        style.configure(
            "Treeview",
            background="#101a24",
            foreground="#dbe6ef",
            fieldbackground="#101a24",
            rowheight=27,
        )
        style.map("Treeview", background=[("selected", "#28506a")])
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#101a24")],
            foreground=[("readonly", "#dbe6ef")],
            selectbackground=[("readonly", "#28506a")],
        )
        style.configure("TButton", padding=5)
        self.header = tk.Label(
            root,
            text="MOSAIC  /  Waiting for mission",
            bg="#111923",
            fg="#eff7ff",
            font=(self.font, 17, "bold"),
            anchor="w",
        )
        self.header.pack(fill="x", padx=18, pady=(12, 4))
        self.sources = tk.Label(root, text="", bg="#111923", fg="#a5bbc9", anchor="w")
        self.sources.pack(fill="x", padx=18, pady=(0, 9))
        self.exploration_info = tk.Label(
            root, text="", bg="#111923", fg="#a5bbc9", anchor="w"
        )
        self.exploration_info.pack(fill="x", padx=18, pady=(0, 4))
        body = ttk.Panedwindow(root, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        left, right = ttk.Frame(body), ttk.Frame(body, width=530)
        body.add(left, weight=3)
        body.add(right, weight=2)
        root.after_idle(lambda: body.sashpos(0, round(root.winfo_width() * 0.53)))
        tools = ttk.Frame(left)
        tools.pack(fill="x", pady=5)
        ttk.Checkbutton(tools, text="Add POI", variable=self.adding).pack(side="left")
        ttk.Combobox(
            tools,
            textvariable=self.kind,
            values=["move", "explore", "ground_measurement"],
            width=19,
            state="readonly",
        ).pack(side="left", padx=5)
        ttk.Checkbutton(
            tools, text="Local map", variable=self.show_local, command=self.draw
        ).pack(side="left")
        ttk.Checkbutton(
            tools, text="Trails", variable=self.show_trails, command=self.draw
        ).pack(side="left")
        ttk.Button(tools, text="Fit", command=self.fit).pack(side="right")
        self.canvas = tk.Canvas(left, bg="#0d1721", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda event: self.draw())
        self.canvas.bind("<Button-1>", self.map_click)
        self.canvas.bind("<Button-3>", self.pan_start)
        self.canvas.bind("<B3-Motion>", self.pan_move)
        self.canvas.bind(
            "<MouseWheel>",
            lambda e: self.set_zoom(self.zoom + (1 if e.delta > 0 else -1)),
        )
        self.canvas.bind("<Button-4>", lambda e: self.set_zoom(self.zoom + 1))
        self.canvas.bind("<Button-5>", lambda e: self.set_zoom(self.zoom - 1))
        legend = ttk.Frame(left)
        legend.pack(fill="x", pady=6)
        for status, label in [
            ("available", "? New"),
            ("running", "Assigned"),
            ("succeeded", "Done (faded)"),
            ("needs_help", "Needs help"),
        ]:
            tk.Label(
                legend, text="● " + label, fg=STATUS_COLORS[status], bg="#18232e"
            ).pack(side="left", padx=5)
        ttk.Label(
            left,
            text="Click a robot / POI to select · Add POI then click terrain · Right-drag to pan · Wheel to zoom",
        ).pack(anchor="w")

        robots = ttk.LabelFrame(right, text="TEAM / Select robot")
        robots.pack(fill="x", padx=6, pady=5)
        self.robot_picker = ttk.Combobox(
            robots, textvariable=self.robot_name, state="readonly"
        )
        self.robot_picker.pack(fill="x", padx=7, pady=5)
        self.robot_picker.bind("<<ComboboxSelected>>", lambda e: self.refresh_details())
        self.robot_info = ttk.Label(
            robots, text="Waiting for localization", justify="left", wraplength=490
        )
        self.robot_info.pack(fill="x", padx=7, pady=3)
        buttons = ttk.Frame(robots)
        buttons.pack(fill="x", padx=5, pady=5)
        for label, op in [
            ("Pause", "pause"),
            ("Resume AUTO", "resume"),
            ("Disable", "disable"),
        ]:
            ttk.Button(
                buttons, text=label, command=lambda op=op: self.robot_command(op)
            ).pack(side="left", padx=2)

        tasks = ttk.LabelFrame(right, text="SHARED POIs / Mission objectives")
        tasks.pack(fill="both", expand=True, padx=6, pady=5)
        self.table = ttk.Treeview(
            tasks, columns=("kind", "status", "owner"), show="tree headings", height=5
        )
        self.table.heading("#0", text="POI")
        self.table.heading("kind", text="Type")
        self.table.heading("status", text="State")
        self.table.heading("owner", text="Robot")
        self.table.column("#0", width=160, stretch=True)
        for column, width in [("kind", 155), ("status", 150), ("owner", 135)]:
            self.table.column(column, width=width, stretch=True)
        self.table.pack(fill="both", expand=True, padx=5, pady=5)
        self.table.bind("<<TreeviewSelect>>", self.table_select)
        row = ttk.Frame(tasks)
        row.pack(fill="x", padx=5, pady=4)
        ttk.Button(
            row, text="Assign to robot", command=lambda: self.poi_command("assign")
        ).pack(side="left", padx=2)
        ttk.Button(
            row, text="Deactivate", command=lambda: self.poi_command("remove_poi")
        ).pack(side="left", padx=2)

        tabs = ttk.Notebook(right)
        tabs.pack(fill="both", expand=True, padx=6, pady=5)
        self.execution, self.science, self.events, self.plans = [
            tk.Text(
                tabs,
                height=7,
                width=40,
                bg="#101a24",
                fg="#d6e4ed",
                insertbackground="white",
                font=(self.font, 10),
                wrap="word",
                relief="flat",
            )
            for _ in range(4)
        ]
        for widget, text in [
            (self.execution, "Execution state"),
            (self.science, "Science results"),
            (self.events, "Events"),
            (self.plans, "POI plan"),
        ]:
            tabs.add(widget, text=text)
        teleop = ttk.LabelFrame(
            right, text="DRIVER / Hold a button to drive; Resume AUTO when finished"
        )
        teleop.pack(fill="x", padx=6, pady=5)
        for label, v, w in [
            ("Forward", 0.15, 0),
            ("Left", 0, 0.25),
            ("Stop", 0, 0),
            ("Right", 0, -0.25),
            ("Reverse", -0.15, 0),
        ]:
            button = ttk.Button(teleop, text=label)
            button.pack(side="left", padx=2, pady=5, expand=True, fill="x")
            button.bind("<ButtonPress-1>", lambda e, v=v, w=w: self.start_hold(v, w))
            button.bind("<ButtonRelease-1>", lambda e: self.stop_hold())
            button.bind("<Leave>", lambda e: self.stop_hold())
        self.notice = ttk.Label(
            right,
            text="No camera stream connected. Instrument actions are simulated.",
            wraplength=510,
        )
        self.notice.pack(fill="x", padx=8, pady=4)
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    def send(self, message):
        try:
            if self.state and self.state.get("finished"):
                raise ValueError(
                    "This mission has ended. Start a new run to send commands."
                )
            with (self.session / "control_commands.jsonl").open("a") as file:
                file.write(json.dumps(message, allow_nan=False) + "\n")
        except (OSError, ValueError) as error:
            messagebox.showerror("Command unavailable", str(error), parent=self.root)

    def robot_command(self, op):
        if self.robot_name.get():
            self.send(dict(op=op, robot=self.robot_name.get()))

    def poi_command(self, op):
        if self.selected_poi:
            command = dict(op=op, poi=self.selected_poi)
            if op == "assign":
                command["robot"] = self.robot_name.get()
            self.send(command)

    def start_hold(self, v, w):
        if self.robot_name.get():
            self.hold = self.robot_name.get(), v, w
            self.last_hold = 0
            self.pulse()

    def pulse(self):
        if self.hold and time.monotonic() - self.last_hold >= 0.3:
            name, v, w = self.hold
            self.send(dict(op="teleop", robot=name, velocity=[v, w], seconds=0.8))
            self.last_hold = time.monotonic()

    def stop_hold(self):
        if self.hold:
            name = self.hold[0]
            self.hold = None
            self.send(dict(op="teleop", robot=name, velocity=[0, 0], seconds=0.8))

    def poll(self):
        try:
            self.state = json.loads((self.session / "mission_state.json").read_text())
            state = self.state
            names = [r["name"] for r in state["robots"]]
            self.robot_picker["values"] = names
            if self.robot_name.get() not in names and names:
                self.robot_name.set(names[0])
            finished = sum(
                p["status"] in {"succeeded", "rejected"} for p in state["pois"]
            )
            self.header.configure(
                text=f"MOSAIC  /  {state['world'].upper()}    {state['stamp']:.1f} s    {finished}/{len(state['pois'])} POIs done    {state['reason']}"
            )
            self.sources.configure(
                text=f"MAP: {state['map']['source']}   |   Localization: {state['providers']['localization']}   |   Simulated sensing / instruments   |   ROBOT PROXIMITY CONFIRMATION"
            )
            exploration = state.get("exploration", {})
            self.exploration_info.configure(
                text=(
                    f"Exploration: {exploration.get('state','waiting')} | Generated POIs: {exploration.get('generated',0)} | Unassigned layout stations: {exploration.get('remaining_sites',0)}"
                    if exploration.get("enabled")
                    else "Exploration: fixed task list"
                )
            )
            for p in state["pois"]:
                values = (p["kind"], p["status"], p["claimed_by"] or "—")
                if self.table.exists(p["id"]):
                    self.table.item(p["id"], values=values)
                else:
                    self.table.insert(
                        "", "end", iid=p["id"], text=p["id"], values=values
                    )
            self.refresh_details()
            self.draw()
            self.pulse()
        except (OSError, ValueError, KeyError):
            pass  # Atomic file not published yet, or runner is replacing it.
        self.root.after(150, self.poll)

    def selected_robot(self):
        return next(
            (
                r
                for r in (self.state or {}).get("robots", [])
                if r["name"] == self.robot_name.get()
            ),
            None,
        )

    @staticmethod
    def text(widget, value):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("end", value)
        widget.configure(state="disabled")

    def refresh_details(self):
        if not self.state:
            return
        r = self.selected_robot()
        if r:
            p = r["pose"]
            self.robot_info.configure(
                text=f"{r.get('label',r['name'])} / {r.get('role','')}\n{r['mode'].upper()} · {r['phase']} · Localization {'OK' if r['valid'] else 'INVALID — STOP'}\n{('Task: '+r['poi']) if r['poi'] else r.get('waiting_reason','Waiting')}\nPose: ({p['x']:.2f}, {p['y']:.2f}) m   Heading: {math.degrees(p['yaw']):.0f}°\nspeed={r.get('speed',abs(r['velocity'][0])):.2f} m/s   lateral={r.get('lateral_speed',0):.2f}   ω={r['velocity'][1]:.2f}"
            )
            stages = [
                ("navigate", "Navigate to approach pose"),
                ("analyze", "Explore / stationary analysis"),
                ("action", "Confirm candidate / acquire measurement"),
                ("yield", "Idle vehicle clears science station"),
            ]
            self.text(
                self.execution,
                "Per-robot execution state machine\n\n"
                + "\n".join(
                    ("▶ " if r["phase"] == key else "  ") + label
                    for key, label in stages
                )
                + f"\n\nCurrent: {r['phase']}\n"
                + (
                    f"Analysis remaining: {r['analysis_remaining']:.1f} s\n"
                    if r.get("analysis_remaining") is not None
                    else ""
                )
                + f"Navigation: {r.get('navigation_wait_reason') or 'tracking' if r['phase'] in {'navigate','yield'} else 'inactive'}\nLocal map: {r['local_map']['source']}\nLocal replans: {r['local_replans']}\nCapabilities: {', '.join(r['capabilities'])}"
                + "\n\nPOI sequence: "
                + " → ".join(s["poi_id"] for s in r.get("poi_plan", [])),
            )
            planning = self.state.get("poi_planning", {})
            self.text(
                self.plans,
                f"{r['name']} | Depth {planning.get('depth',1)} | Version {planning.get('version',0)}\n"
                + "Current task: exclusive. Future steps: replaceable intentions.\n"
                + "Dashed map links show task order, not navigation paths.\n\n"
                + "\n".join(
                    f"{'▶' if s['current'] else str(s['depth']+1)+'.'} {s['poi_id']}\n    U={s['utility']:.1f}, discounted={s['discounted_utility']:.1f} | route={s['path_length']:.1f} m | ETA={max(0,s['expected_arrival']-self.state['stamp']):.1f} s"
                    for s in r.get("poi_plan", [])
                )
                + f"\n\nUpdated: {planning.get('generated_at')} s | Valid until: {planning.get('valid_until')} s\n"
                + f"Planning: {'computing' if planning.get('computing') else 'ready'} | Error: {planning.get('error') or 'none'}",
            )
        poi = next(
            (p for p in self.state["pois"] if p["id"] == self.selected_poi), None
        )
        if poi:
            station = poi.get("navigation_station", poi["approach"])
            station_text = (
                f"Station: ({station['x']:.2f}, {station['y']:.2f}) m"
                if station is not None
                else "Station: unavailable; science not measured"
            )
            if poi.get("measurement_radius") is not None:
                station_text += (
                    f"\nScience target range: <= {poi['measurement_radius']:.2f} m"
                )
            attempts = poi.get("attempts", [])
            vehicle_name = poi.get("claimed_by") or (
                attempts[-1]["robot"] if attempts else None
            )
            vehicle = next(
                (r for r in self.state["robots"] if r["name"] == vehicle_name), None
            )
            if vehicle and "rock_measurement" in vehicle["capabilities"]:
                distance = math.hypot(
                    vehicle["pose"]["x"] - poi["target"]["x"],
                    vehicle["pose"]["y"] - poi["target"]["y"],
                )
                station_text += (
                    f"\n{vehicle_name} current target distance: {distance:.2f} m"
                )
            if attempts and not attempts[-1]["success"]:
                station_text += f"\nLast failure: {attempts[-1]['reason']} at {attempts[-1]['stamp']:.1f} s"
            prefix = f"POI: {poi['id']}\n{poi.get('wait_reason',poi['status'])}\n{station_text}\n\n"
            self.execution.configure(state="normal")
            self.execution.insert("1.0", prefix)
            self.execution.configure(state="disabled")
        result = (poi.get("result") or poi.get("confirmation_result")) if poi else None
        if result:
            details = (
                f"Analysis duration: {result['analysis_seconds']:.1f} s\nNew candidates: {', '.join(result['candidates']) or 'None'}\nObserved objects: {', '.join(result.get('observed_objects',[])) or 'None'}"
                if result.get("kind") == "explore"
                else f"Target: {result.get('target','—')}\nRecorded position: {result.get('approach','—')}\nSource: {result.get('source','—')}"
                + (
                    f"\nConfirmed for science: {result['confirmed_measurement']}"
                    if "confirmed_measurement" in result
                    else ""
                )
            )
            self.text(
                self.science,
                f"{poi['id']}\n{'SIMULATED result — no physical instrument' if result.get('simulated') else 'Instrument result'}\n"
                + (
                    "Scout confirmation record; science not measured\n"
                    if poi["kind"] == "rock_measurement" and not poi.get("result")
                    else ""
                )
                + f"\nRobot: {result.get('robot','—')}\nRecorded: {result.get('stamp','—')} s\n{details}"
                + (
                    f"\nTarget distance: {result['target_distance']:.2f} m\nStation error: {result['station_error']:.2f} m"
                    if "target_distance" in result
                    else ""
                ),
            )
        else:
            self.text(
                self.science,
                "Select a completed measurement POI to inspect its result.\n\nNo camera or spectrometer is connected in this version.",
            )
        self.text(
            self.events,
            "\n".join(
                f"{e.get('stamp',0):6.1f}  {e['event']}  {e.get('robot','')}  {e.get('poi','')}  {e.get('reason','')}"
                for e in self.state["events"][-25:]
            ),
        )
        self.events.see("end")

    def table_select(self, event=None):
        selected = self.table.selection()
        if selected:
            self.selected_poi = selected[0]
            self.refresh_details()
            self.draw()

    def fit(self):
        self.zoom, self.pan = 1, [0, 0]
        self.image_key = None
        self.draw()

    def set_zoom(self, value):
        self.zoom = min(3, max(1, value))
        self.image_key = None
        self.draw()

    def pan_start(self, event):
        self.drag_start = event.x, event.y, *self.pan

    def pan_move(self, event):
        if self.drag_start:
            x, y, px, py = self.drag_start
            self.pan = [px + event.x - x, py + event.y - y]
            self.draw()

    def transform(self):
        g = self.state["map"]
        scale = (self.image.width() - 1) / ((g["shape"][1] - 1) * g["resolution"])
        ox = (self.canvas.winfo_width() - self.image.width()) / 2 + self.pan[0]
        oy = (self.canvas.winfo_height() - self.image.height()) / 2 + self.pan[1]
        return scale, ox, oy

    def screen(self, x, y):
        scale, ox, oy = self.transform()
        gx, gy = self.state["map"]["origin"]
        return ox + (x - gx) * scale, oy + self.image.height() - 1 - (y - gy) * scale

    def map_click(self, event):
        if not self.state or self.image is None:
            return
        for robot in self.state["robots"]:
            x, y = self.screen(robot["pose"]["x"], robot["pose"]["y"])
            if math.hypot(x - event.x, y - event.y) < 22:
                self.robot_name.set(robot["name"])
                self.refresh_details()
                self.draw()
                return
        for poi in self.state["pois"]:
            x, y = self.screen(poi["target"]["x"], poi["target"]["y"])
            if math.hypot(x - event.x, y - event.y) < 18:
                self.table.selection_set(poi["id"])
                self.table.see(poi["id"])
                return
        if self.adding.get():
            scale, ox, oy = self.transform()
            gx, gy = self.state["map"]["origin"]
            x = gx + (event.x - ox) / scale
            y = gy + (self.image.height() - 1 - event.y + oy) / scale
            yaw = simpledialog.askfloat(
                "Goal heading",
                f"New {self.kind.get()} at ({x:.2f}, {y:.2f}) m\nFinal heading (degrees)",
                initialvalue=0,
                parent=self.root,
            )
            if yaw is not None:
                self.send(
                    dict(
                        op="add_poi",
                        id=f"{self.kind.get()}_{int(time.time()*1000)}",
                        kind=self.kind.get(),
                        pose=[x, y, math.radians(yaw)],
                    )
                )
                self.adding.set(False)

    def draw(self):
        if not self.state:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        if min(width, height) < 2:
            return
        key = self.state["map"]["version"], self.zoom, width, height
        if key != self.image_key:
            try:
                raster = cv2.imread(str(self.session / self.state["map"]["image"]))
                if raster is None:
                    return
                ratio = (
                    min(width / raster.shape[1], height / raster.shape[0]) * self.zoom
                )
                size = tuple(max(2, round(d * ratio)) for d in raster.shape[1::-1])
                resized = cv2.resize(raster, size, interpolation=cv2.INTER_NEAREST)
                ok, encoded = cv2.imencode(".png", resized)
                if not ok:
                    return
                self.image = tk.PhotoImage(data=encoded.tobytes())
                self.image_key = key
            except tk.TclError:
                return
        c = self.canvas
        c.delete("all")
        scale, ox, oy = self.transform()
        c.create_image(ox, oy, image=self.image, anchor="nw")
        for robot in self.state["robots"]:
            if self.show_local.get() and robot["name"] == self.robot_name.get():
                for x, y in robot["local_map"]["blocked"]:
                    sx, sy = self.screen(x, y)
                    half = max(1, robot["local_map"]["resolution"] * scale / 2)
                    c.create_rectangle(
                        sx - half,
                        sy - half,
                        sx + half,
                        sy + half,
                        fill="#ee784d",
                        outline="",
                        stipple="gray50",
                    )
            if self.show_trails.get() and len(robot["trail"]) > 1:
                points = [v for xy in robot["trail"] for v in self.screen(*xy)]
                c.create_line(*points, fill=robot["color"], width=1, dash=(3, 4))
            if len(robot["path"]) > 1:
                points = [
                    v
                    for xy in [[robot["pose"]["x"], robot["pose"]["y"]], *robot["path"]]
                    for v in self.screen(*xy)
                ]
                c.create_line(*points, fill=robot["color"], width=3)
        label_positions = []
        # These are ordered task intentions, not collision-free navigation paths.
        for robot in self.state["robots"]:
            for step in robot.get("poi_plan", []):
                if step["depth"] == 0 or step["current"]:
                    continue
                a, b = step["start"], step["approach"]
                x0, y0 = self.screen(a["x"], a["y"])
                x1, y1 = self.screen(b["x"], b["y"])
                c.create_line(x0, y0, x1, y1, fill=robot["color"], dash=(7, 5), width=2)
                c.create_text(
                    x1 + 8,
                    y1 + 8,
                    text=f"{step['depth']+1}",
                    fill=robot["color"],
                    anchor="nw",
                )
        for index, p in enumerate(self.state["pois"], 1):
            x, y = self.screen(p["target"]["x"], p["target"]["y"])
            station = p.get("navigation_station", p["approach"])
            color = STATUS_COLORS[p["status"]]
            if station is not None:
                ax, ay = self.screen(station["x"], station["y"])
                if math.hypot(x - ax, y - ay) > 3:
                    c.create_line(x, y, ax, ay, fill=color, dash=(4, 3))
                    c.create_oval(
                        ax - 4, ay - 4, ax + 4, ay + 4, outline=color, width=2
                    )
            width = 3 if p["id"] == self.selected_poi else 1
            if p["kind"] in {"rock_candidate", "rock_measurement"}:
                if p["kind"] == "rock_candidate" and p["id"] == self.selected_poi:
                    radius = p.get("confirmation_radius", 3.0) * scale
                    c.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        outline=color,
                        dash=(6, 5),
                    )
                if (
                    p["kind"] == "rock_measurement"
                    and p["id"] == self.selected_poi
                    and p.get("measurement_radius") is not None
                ):
                    radius = p["measurement_radius"] * scale
                    c.create_oval(
                        x - radius,
                        y - radius,
                        x + radius,
                        y + radius,
                        outline=color,
                        dash=(6, 5),
                    )
                c.create_polygon(
                    x,
                    y - 10,
                    x + 10,
                    y,
                    x,
                    y + 10,
                    x - 10,
                    y,
                    fill="#121c25",
                    outline=color,
                    width=width,
                )
            elif p["kind"] == "ground_measurement":
                c.create_rectangle(
                    x - 9,
                    y - 9,
                    x + 9,
                    y + 9,
                    fill="#121c25",
                    outline=color,
                    width=width,
                )
            else:
                c.create_oval(
                    x - 10,
                    y - 10,
                    x + 10,
                    y + 10,
                    fill="#121c25",
                    outline=color,
                    width=width,
                )
            symbol = {
                "available": "?",
                "succeeded": "✓",
                "needs_help": "!",
            }.get(p["status"], "•")
            c.create_text(x, y, text=symbol, fill=color, font=(self.font, 10, "bold"))
            kinds = dict(
                move="Move",
                explore="Explore",
                ground_measurement="Ground",
                rock_candidate="Candidate",
                rock_measurement="Rock measure",
            )
            label = f"P{index} {kinds[p['kind']]}\n" + (
                p["claimed_by"].replace("lunar_", "")
                if p["claimed_by"]
                else p["status"]
            )
            lx, ly = x + 15, y - 15
            while any(
                abs(lx - px) < 150 and abs(ly - py) < 50 for px, py in label_positions
            ):
                ly -= 52
            label_positions.append((lx, ly))
            c.create_text(
                lx,
                ly,
                text=label,
                anchor="sw",
                fill=color,
                font=(self.font, 10),
            )
        for robot in self.state["robots"]:
            p = robot["pose"]
            x, y = self.screen(p["x"], p["y"])
            if robot["name"] == self.robot_name.get():
                radius = robot["radius"] * scale
                c.create_oval(
                    x - radius,
                    y - radius,
                    x + radius,
                    y + radius,
                    outline=robot["color"],
                    dash=(3, 4),
                )
            vertices = []
            for angle, length in [
                (p["yaw"], 18),
                (p["yaw"] + 2.5, 12),
                (p["yaw"] - 2.5, 12),
            ]:
                vertices.extend(
                    [x + length * math.cos(angle), y - length * math.sin(angle)]
                )
            c.create_polygon(
                *vertices,
                fill=robot["color"] if robot["valid"] else "#f16f6f",
                outline="#061019",
                width=2,
            )
            c.create_text(
                x,
                y + 23,
                text=f"{robot.get('label',robot['name'].replace('lunar_', ''))}\n{robot['phase'].upper()}",
                fill=robot["color"],
                anchor="n",
                font=(self.font, 10, "bold"),
                width=130,
            )

    def close(self):
        if not (self.state or {}).get("finished"):
            self.hold = None
            self.send(dict(op="stop"))
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    args = parser.parse_args(argv)
    root = tk.Tk()
    ControlPanel(root, args.session)
    root.mainloop()


if __name__ == "__main__":
    main()
