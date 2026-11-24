"""Shared POI state, atomic claims and fenced completion records."""

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from threading import RLock
from uuid import uuid4

from .contracts import Kind, Pose, Robot


class Status(StrEnum):
    AVAILABLE = "available"
    CLAIMED = "claimed"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    REJECTED = "rejected"
    NEEDS_HELP = "needs_help"


@dataclass
class POI:
    id: str
    kind: Kind
    target: Pose
    approach: Pose
    status: Status = Status.AVAILABLE
    assigned_to: str | None = None
    object_id: str | None = None
    claimed_by: str | None = None
    token: str | None = None
    lease_until: float = 0.0
    revision: int = 0
    attempts: list[dict] = field(default_factory=list)
    result: dict | None = None
    measurement_approach: Pose | None = None
    confirmation_radius: float = 3.0
    confirmation_result: dict | None = None
    discovered_by: str | None = None
    source_exploration: str | None = None
    task_source: str = "operator_or_config"
    measurement_radius: float | None = None
    blocked_reason: str = ""

    @property
    def navigation_station(self):
        if self.kind == Kind.ROCK_MEASUREMENT and self.measurement_radius is not None:
            return self.measurement_approach
        return self.approach


class POIStore:
    def __init__(self, *, lease_seconds=5.0, retry_seconds=10.0, max_failures=3):
        self.items = {}
        self.lease_seconds, self.retry_seconds = lease_seconds, retry_seconds
        self.max_failures = max_failures
        self.lock = RLock()

    def add(self, poi):
        with self.lock:
            if poi.id in self.items:
                raise ValueError(f"Duplicate POI: {poi.id}")
            self.items[poi.id] = poi
        return poi

    def observe(self, detection):
        with self.lock:
            existing = next(
                (p for p in self.items.values() if p.object_id == detection.object_id),
                None,
            )
            if existing:
                approach = (
                    detection.measurement_approach
                    or existing.measurement_approach
                    or existing.approach
                    if existing.kind == Kind.ROCK_MEASUREMENT
                    else detection.approach
                )
                if existing.target != detection.target or existing.approach != approach:
                    existing.target, existing.approach = (
                        detection.target,
                        approach,
                    )
                    existing.revision += 1
                if detection.measurement_approach is not None:
                    existing.measurement_approach = detection.measurement_approach
                existing.confirmation_radius = detection.confirmation_radius
                existing.measurement_radius = detection.measurement_radius
                return existing, False
            return (
                self.add(
                    POI(
                        id=f"rock:{detection.object_id}",
                        kind=Kind.ROCK_CANDIDATE,
                        target=detection.target,
                        approach=detection.approach,
                        object_id=detection.object_id,
                        measurement_approach=detection.measurement_approach,
                        confirmation_radius=detection.confirmation_radius,
                        measurement_radius=detection.measurement_radius,
                    )
                ),
                True,
            )

    def eligible(self, poi, robot: Robot, now):
        if poi.status != Status.AVAILABLE or poi.kind not in robot.capabilities:
            return False
        if poi.assigned_to not in (None, robot.name):
            return False
        return not any(
            a["robot"] == robot.name
            and not a["success"]
            and not a["cancelled"]
            and now - a["stamp"] < self.retry_seconds
            for a in poi.attempts
        )

    def claim(self, poi_id, robot, now):
        with self.lock:
            self.expire(now)
            p = self.items[poi_id]
            if not self.eligible(p, robot, now):
                return None
            if any(q.claimed_by == robot.name for q in self.items.values()):
                return None
            p.status, p.claimed_by, p.token = Status.CLAIMED, robot.name, uuid4().hex
            p.lease_until = now + self.lease_seconds
            return p.token

    def renew(self, poi_id, token, now):
        with self.lock:
            self.expire(now)
            p = self.items[poi_id]
            if p.token != token or p.token is None:
                return False
            p.lease_until, p.status = now + self.lease_seconds, Status.RUNNING
            return True

    def finish(
        self, poi_id, token, now, *, success, reason="", result=None, cancelled=False
    ):
        with self.lock:
            p = self.items[poi_id]
            # A late worker must not complete a task reclaimed by another robot.
            if not token or p.token != token or now >= p.lease_until:
                return False
            self._release(p, now, success, reason, result, cancelled)
            return True

    def _release(self, p, now, success, reason, result=None, cancelled=False):
        p.attempts.append(
            dict(
                robot=p.claimed_by,
                token=p.token,
                kind=p.kind,
                stamp=now,
                success=success,
                reason=reason,
                cancelled=cancelled,
            )
        )
        p.claimed_by = p.token = None
        p.lease_until = 0.0
        if success:
            p.result = result
            if p.kind == Kind.ROCK_CANDIDATE:
                p.confirmation_result = result
                if result and result.get("confirmed_measurement") is True:
                    p.kind, p.status = Kind.ROCK_MEASUREMENT, Status.AVAILABLE
                    p.approach = p.measurement_approach or p.approach
                    p.result, p.assigned_to = None, None
                    p.revision += 1
                else:
                    p.status = Status.REJECTED
            else:
                p.status = Status.SUCCEEDED
        elif (
            sum(not a["success"] and not a["cancelled"] for a in p.attempts)
            >= self.max_failures
        ):
            p.status = Status.NEEDS_HELP
        else:
            p.status = Status.AVAILABLE

    def expire(self, now):
        with self.lock:
            for p in self.items.values():
                if p.token and now >= p.lease_until:
                    self._release(p, now, False, "lease_expired")

    def approve(self, poi_id, approved=True):
        with self.lock:
            p = self.items[poi_id]
            if p.status != Status.WAITING_APPROVAL:
                raise ValueError("Only inspected candidates can be approved/rejected")
            if approved:
                p.kind, p.status = Kind.ROCK_MEASUREMENT, Status.AVAILABLE
                if p.measurement_approach is not None:
                    p.approach = p.measurement_approach
                p.result = None
                p.assigned_to = None
                p.revision += 1
            else:
                p.status = Status.REJECTED

    def records(self):
        with self.lock:
            return [asdict(p) for p in self.items.values()]
