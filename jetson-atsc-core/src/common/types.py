"""
Strongly-Typed Domain Dataclasses for ATSC Architecture.

Replaces loose dictionary passing across Perception, Decision, and Actuation layers
to guarantee type safety, eliminate KeyError runtime crashes, and enable instant serialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from copy import deepcopy
import json
import time
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TrafficSnapshot:
    """
    Normalized physical traffic metrics at a given timestamp.
    Calculated in accordance with Indonesian PKJI 2014 guidelines.
    """
    timestamp: float = field(default_factory=time.time)
    v_w_pcu: float = 0.0          # Weighted vehicle count (MC:0.4, LV:1.0, HV:1.6)
    queue_meters: Optional[float] = 0.0  # None means unavailable, never measured zero
    occupancy_pct: float = 0.0    # Fixed-area perspective-invariant lane occupancy [0.0, 100.0]
    n_mc: int = 0                 # Motorcycle count inside approach corridor
    n_lv: int = 0                 # Light vehicle count inside approach corridor
    n_hv: int = 0                 # Heavy vehicle (bus/truck) count inside approach corridor
    total_vehicles: int = 0       # Total active tracked vehicles in corridor
    stopped_vehicles: Optional[int] = 0  # Vehicles with speed <= 1.0 m/s; None = unknown (single still image)
    occupied_area_m2: float = 0.0 # Physical road surface area occupied (m^2)
    camera_source: str = "KameraSistem"
    quality: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], timestamp: Optional[float] = None) -> TrafficSnapshot:
        """Constructs TrafficSnapshot from metrics extractor dictionary for backward compatibility."""
        return cls(
            timestamp=timestamp if timestamp is not None else time.time(),
            v_w_pcu=float(d.get("V_w", 0.0)),
            queue_meters=float(d.get("Q", 0.0)) if d.get("Q", 0.0) is not None else None,
            occupancy_pct=float(d.get("L", 0.0)),
            n_mc=int(d.get("n_mc", 0)),
            n_lv=int(d.get("n_lv", 0)),
            n_hv=int(d.get("n_hv", 0)),
            total_vehicles=int(d.get("total_corridor_vehicles", 0)),
            # None: stopped state unknown (single still image), never a fabricated zero.
            stopped_vehicles=int(d.get("stopped_vehicles_count", 0)) if d.get("stopped_vehicles_count", 0) is not None else None,
            occupied_area_m2=float(d.get("occupied_area_m2", 0.0)),
            camera_source=str(d.get("camera_source", "KameraSistem")),
            quality=deepcopy(d.get("quality") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes snapshot to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class ActuationDecision:
    """
    Decision produced by the ANFIS green-time allocation engine.
    """
    cycle_id: int
    green_seconds: float
    min_bound: float = 10.0
    max_bound: float = 120.0
    is_clamped_min: bool = False
    is_clamped_max: bool = False
    inference_latency_ms: float = 0.0
    failsafe_active: bool = False
    timestamp_iso: str = ""
    inference_diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes decision to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class HardwareTelemetry:
    """
    Snapshot of edge hardware health (Jetson Orin Nano).
    """
    soc_temp_c: float = 0.0
    fps: float = 0.0
    mode: str = "DUAL_CAM"
    is_thermal_throttled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes hardware status to dictionary."""
        return asdict(self)


@dataclass
class CycleEvent:
    """
    Unified, self-contained record of a single traffic light actuation cycle.
    Passed to all registered CycleSink observers (CSV ledger, Video Recorder, LLM Auditor).
    """
    cycle_id: int
    timestamp_iso: str
    traffic: TrafficSnapshot
    decision: ActuationDecision
    telemetry: HardwareTelemetry
    n_served: int = 0
    theoretical_capacity_pcu: float = 0.0
    watchdog_status: str = "OK"
    llm_anomaly_flag: str = "PENDING"
    llm_severity: str = "PENDING"
    llm_critique: str = "Awaiting asynchronous audit..."

    def to_ledger_row(self) -> Dict[str, Any]:
        """
        Converts the cycle event to exact dictionary schema expected by field_experiment_ledger.csv.
        """
        return {
            "cycle_id": self.cycle_id,
            "timestamp": self.timestamp_iso,
            "mode": self.telemetry.mode,
            "v_w_pcu": self.traffic.v_w_pcu,
            "q_meters": self.traffic.queue_meters,
            "l_occupancy_pct": self.traffic.occupancy_pct,
            "t_anfis_sec": self.decision.green_seconds,
            "n_served_actual": self.n_served,
            "theoretical_capacity_pcu": self.theoretical_capacity_pcu,
            "fps": self.telemetry.fps,
            "soc_temp_c": self.telemetry.soc_temp_c,
            "watchdog_status": self.watchdog_status,
            "llm_anomaly_flag": self.llm_anomaly_flag,
            "llm_severity": self.llm_severity,
            "llm_critique": self.llm_critique,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "timestamp_iso": self.timestamp_iso,
            "traffic": self.traffic.to_dict(),
            "decision": self.decision.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "n_served": self.n_served,
            "theoretical_capacity_pcu": self.theoretical_capacity_pcu,
            "watchdog_status": self.watchdog_status,
            "llm_anomaly_flag": self.llm_anomaly_flag,
            "llm_severity": self.llm_severity,
            "llm_critique": self.llm_critique,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        return json.dumps(self.to_dict(), indent=indent)
