from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_COMMIT = "578906ad5c3383087aeec038be82b87e05d4ee2c"
CONTRACT = "rlroverlab-gazebo-navigation-v1"


@dataclass
class NavigationConfig:
    world: str = "moon"
    assets: str = str(ROOT / "assets/cache")
    image: str = "space-robotics-rl:ionic"
    device: str = "cpu"
    seed: int = 42
    control_dt: float = 0.2
    episode_seconds: float = 150.0
    target_distance: float = 9.0
    success_distance: float = 0.18
    contact_threshold: float = 1.0
    settle_steps: int = 100
    spawn_margin: float = 10.0
    target_margin: float = 7.0
    slope_limit: float = 0.35
    obstacle_margin: float = 2.0

    def __post_init__(self):
        if self.world not in {"moon", "mars"}:
            raise ValueError("world must be moon or mars")
        if (
            self.control_dt <= 0
            or self.episode_seconds <= 0
            or self.target_distance <= 0
        ):
            raise ValueError("Time and distance settings must be positive")

    def to_dict(self):
        return asdict(self)
