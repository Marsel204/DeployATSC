from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class Phase(str, Enum):
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"
    UNKNOWN = "UNKNOWN"

    def __str__(self):return self.value


@dataclass(frozen=True)
class RunPolicy:
    operation: str = "observation"
    movement_id: str = "approach-1"
    offset_sec: float = 0.0
    stopped_speed_mps: float = 0.3
    moving_speed_mps: float = 0.5
    stop_persistence_sec: float = 0.3
    association_gap_sec: float = 1.0
    max_gap_sec: float = 0.5
    freshness_sec: float = 1.0
    debounce_frames: int = 3
    phase_ratio: float = 0.08
    los_method: str | None = None

    def __post_init__(self):
        numeric=("offset_sec","stopped_speed_mps","moving_speed_mps","stop_persistence_sec",
                 "association_gap_sec","max_gap_sec","freshness_sec","phase_ratio")
        for key in numeric:
            if isinstance(getattr(self,key),bool) or not isinstance(getattr(self,key),(int,float)):
                raise ValueError(f"{key} must be a number")
        if isinstance(self.debounce_frames,bool) or not isinstance(self.debounce_frames,int):
            raise ValueError("Signal debounce requires a whole number of frames")
        for key, value in vars(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.operation != "observation":
            raise ValueError("This runtime supports observation only")
        if not self.movement_id.strip():
            raise ValueError("A target movement is required")
        if self.stopped_speed_mps < 0 or self.moving_speed_mps <= self.stopped_speed_mps:
            raise ValueError("Motion thresholds must provide positive hysteresis")
        if min(self.association_gap_sec, self.max_gap_sec, self.freshness_sec) <= 0:
            raise ValueError("Gap and freshness limits must be positive")
        if self.stop_persistence_sec < 0 or self.debounce_frames < 1:
            raise ValueError("Persistence settings are invalid")
        if not 0 < self.phase_ratio <= 1:
            raise ValueError("Phase pixel ratio must be in (0, 1]")


@dataclass(frozen=True)
class VehicleObservation:
    local_id: int | None
    class_name: str
    speed_mps: float | None
    point: tuple[float, float]
    stable_id: bool = True
    velocity_valid: bool = False
    in_region: bool = True
    epoch: int = 0


def vehicle_key(run_id: str, epoch: int, local_id: int) -> str:
    return f"{run_id}:acquisition:{epoch}:{local_id}"
