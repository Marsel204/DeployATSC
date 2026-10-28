"""
Pure ATSC Decision Engine.

Decouples ANFIS traffic control logic from hardware drivers, video capture,
and background execution threads. Allows pure input -> decision unit testing.
"""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Optional

from src.common.types import TrafficSnapshot, ActuationDecision
from src.control.anfis_inference import ANFISInferenceEngine


class ATSCDecisionEngine:
    """
    Pure decision engine mapping physical traffic state to green signal duration.
    """

    def __init__(
        self,
        anfis_engine: ANFISInferenceEngine,
        min_green_sec: float = 10.0,
        max_green_sec: float = 120.0,
        fallback_green_sec: float = 30.0,
    ) -> None:
        self.anfis = anfis_engine
        self.min_green = min_green_sec
        self.max_green = max_green_sec
        self.fallback_green = fallback_green_sec

    def evaluate(self, traffic: TrafficSnapshot, cycle_id: int) -> ActuationDecision:
        """
        Maps a TrafficSnapshot to an ActuationDecision.

        Args:
            traffic: Physical traffic parameters (V_w, Q, L).
            cycle_id: Monotonically increasing cycle counter.

        Returns:
            ActuationDecision with allocated green seconds and diagnostic bounds metadata.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        t0 = time.perf_counter()

        diagnostics = {}
        try:
            structured = getattr(self.anfis, "evaluate_green_time_result", None)
            if structured is not None:
                result = structured(V_w=traffic.v_w_pcu, Q=traffic.queue_meters, L=traffic.occupancy_pct)
                diagnostics = result.to_dict()
                green_time = result.rounded_green_sec
                failsafe = result.backend_fallback
            else:
                green_time = self.anfis.evaluate_green_time(
                    V_w=traffic.v_w_pcu, Q=traffic.queue_meters, L=traffic.occupancy_pct)
                failsafe = bool(getattr(self.anfis, "last_error", None))
                diagnostics = {"raw_green_sec":None, "bounded_green_sec":None,
                    "rounded_green_sec":round(green_time,2), "backend_fallback":failsafe,
                    "diagnostics_state":"legacy_scalar", "model_domain":"unknown"}
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as error:
            green_time = self.fallback_green
            latency_ms = (time.perf_counter() - t0) * 1000.0
            failsafe = True
            diagnostics = {"raw_green_sec":None, "bounded_green_sec":None,
                "rounded_green_sec":round(green_time,2), "backend_fallback":True,
                "backend_error":"ANFIS backend failure ("+type(error).__name__+")", "model_domain":"unknown"}

        # Clamp check metadata
        is_clamped_min = abs(green_time - self.min_green) < 1e-3
        is_clamped_max = abs(green_time - self.max_green) < 1e-3

        return ActuationDecision(
            cycle_id=cycle_id,
            green_seconds=round(green_time, 2),
            min_bound=self.min_green,
            max_bound=self.max_green,
            is_clamped_min=is_clamped_min,
            is_clamped_max=is_clamped_max,
            inference_latency_ms=round(latency_ms, 3),
            failsafe_active=failsafe,
            timestamp_iso=now_iso,
            inference_diagnostics=diagnostics,
        )
