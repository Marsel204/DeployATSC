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
    # Native IDs only, opt-in: continue a just-lost vehicle row with a newborn
    # native ID when the pair is mutually unique and no other observed box
    # competes. The handoff interval is always unknown coverage.
    fragment_continuation: bool = False
    # Native IDs only, opt-in (0 = off): require consecutive outside-ROI
    # sightings spanning at least this many seconds before completing a row as
    # EXITED, preventing brief border jitter from splitting one vehicle into
    # multiple segments. Outside frames never accrue wait.
    exit_persistence_sec: float = 0.0

    def __post_init__(self):
        numeric=("offset_sec","stopped_speed_mps","moving_speed_mps","stop_persistence_sec",
                 "association_gap_sec","max_gap_sec","freshness_sec","phase_ratio",
                 "exit_persistence_sec")
        for key in numeric:
            if isinstance(getattr(self,key),bool) or not isinstance(getattr(self,key),(int,float)):
                raise ValueError(f"{key} must be a number")
        if isinstance(self.debounce_frames,bool) or not isinstance(self.debounce_frames,int):
            raise ValueError("Signal debounce requires a whole number of frames")
        if not isinstance(self.fragment_continuation,bool):
            raise ValueError("fragment_continuation must be true or false")
        for key, value in vars(self).items():
            if isinstance(value, (int, float)):
                try:
                    finite = math.isfinite(value)
                except OverflowError:
                    finite = False
                if not finite:
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
        if not 0 <= self.exit_persistence_sec <= 5:
            raise ValueError("Exit persistence must be in [0, 5]")
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
    # Bounding-box height in pixels: the image scale for motion gates.
    height: float | None = None
    # Raw image box (x1, y1, x2, y2): competing-box checks for continuation.
    bbox: tuple[float, float, float, float] | None = None
    # The detector's own post-NMS box behind ``bbox`` when ``bbox`` is a tracker
    # (Kalman-smoothed) state. Continuation lets both veto another vehicle's link.
    source_bbox: tuple[float, float, float, float] | None = None


def vehicle_key(run_id: str, epoch: int, local_id: int) -> str:
    return f"{run_id}:acquisition:{epoch}:{local_id}"
