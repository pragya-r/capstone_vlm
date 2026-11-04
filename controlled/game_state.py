"""GameState: structured representation of a single parsed frame."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class WeaponType(str, Enum):
    GAUNTLET = "gauntlet"
    MACHINEGUN = "machinegun"
    SHOTGUN = "shotgun"
    GRENADE_LAUNCHER = "grenade_launcher"
    ROCKET_LAUNCHER = "rocket_launcher"
    LIGHTNING = "lightning"
    RAILGUN = "railgun"
    PLASMAGUN = "plasmagun"
    BFG = "bfg"
    UNKNOWN = "unknown"


@dataclass
class EnemyDetection:
    """One detected enemy in the frame, in image-space coordinates."""
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2) pixel coords
    confidence: float
    center_frac: tuple[float, float]
    track_id: Optional[int] = None  # populated by TemporalMemory, not detection.py


@dataclass
class GameState:
    """Parsed state for a single frame. Populated by detection.py."""

    frame_id: int
    timestamp: float

    health: Optional[int] = None
    armor: Optional[int] = None
    ammo: Optional[int] = None
    weapon: WeaponType = WeaponType.UNKNOWN

    enemies: list[EnemyDetection] = field(default_factory=list)

    raw_hud_text: str = ""

    def to_prompt_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "t": round(self.timestamp, 2),
            "health": self.health,
            "armor": self.armor,
            "ammo": self.ammo,
            "weapon": self.weapon.value,
            "enemies_visible": len(self.enemies),
            "enemies": [
                {
                    "track_id": e.track_id,
                    "bearing_x": round(e.center_frac[0], 3),
                    "bearing_y": round(e.center_frac[1], 3),
                    "confidence": round(e.confidence, 2),
                }
                for e in self.enemies
            ],
        }