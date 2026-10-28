"""
ONNXRuntime Inference Engine for 126-Parameter Sugeno ANFIS Green-Time Allocation.

Loads models/anfis_sugeno.onnx and executes hardware-accelerated forward inference
on input traffic metrics [V_w, Q, L], enforcing strict green duration bounds [10s, 120s].
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class ANFISResult:
    raw_green_sec: float | None
    bounded_green_sec: float
    rounded_green_sec: float
    clipping_applied: bool | None
    bound_touch_min: bool
    bound_touch_max: bool
    rounding_applied: bool
    rounded_to_bound: bool
    backend_fallback: bool
    backend_error: str | None
    provider: str
    inference_latency_ms: float
    model_domain: str = "unknown"

    def to_dict(self):
        return asdict(self)


class ANFISInferenceEngine:
    """
    Evaluates optimal green phase duration using trained Sugeno ANFIS model in ONNX format.
    """

    def __init__(
        self,
        model_path: str,
        prefer_cuda: bool = True,
        min_green_sec: float = 10.0,
        max_green_sec: float = 120.0,
    ) -> None:
        """
        Args:
            model_path: Path to models/anfis_sugeno.onnx.
            prefer_cuda: If True, uses CUDAExecutionProvider; falls back to CPUExecutionProvider.
            min_green_sec: Minimum allowable green phase duration (default 10.0s).
            max_green_sec: Maximum allowable green phase duration (default 120.0s).
        """
        self.model_path = model_path
        self.min_green = min_green_sec
        self.max_green = max_green_sec

        self.session = None
        self.input_name: str = ""
        self.output_name: str = ""
        self.active_provider: str = ""
        self.last_inference_ms: float = 0.0
        self.last_error: Optional[str] = None

        self._init_session(prefer_cuda)

    def _init_session(self, prefer_cuda: bool) -> None:
        """Initializes ONNXRuntime session with appropriate execution providers."""
        import onnxruntime as ort

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"ANFIS ONNX model not found at: {self.model_path}")

        available_providers = ort.get_available_providers()
        print(f"[ANFIS Inference] Available ONNX providers: {available_providers}")

        providers = []
        if prefer_cuda and "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2

        self.session = ort.InferenceSession(self.model_path, sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        print(
            f"[ANFIS Inference] Loaded '{self.model_path}' successfully with provider: {self.active_provider}"
        )

    def evaluate_green_time(self, V_w: float, Q: float, L: float) -> float:
        """Compatible scalar API. Structured consumers use the result of their call."""
        return self.evaluate_green_time_result(V_w, Q, L).rounded_green_sec

    def evaluate_green_time_result(self, V_w: float, Q: float, L: float) -> ANFISResult:
        t0 = time.perf_counter()
        raw = None
        error = None
        try:
            inputs = np.array([[float(V_w), float(Q), float(L)]], dtype=np.float32)
            output = self.session.run([self.output_name], {self.input_name:inputs})[0]
            value = float(output[0, 0])
            if not np.isfinite(value):
                raise ValueError("ANFIS returned a non-finite green duration")
            raw = value
        except Exception as failure:
            # Native errors can echo input URLs/settings. Persist only a safe category.
            error = "ANFIS backend failure (" + type(failure).__name__ + ")"
            value = 30.0
        bounded = max(self.min_green, min(self.max_green, value))
        rounded = round(float(bounded), 2)
        elapsed = (time.perf_counter()-t0)*1000
        result = ANFISResult(raw, bounded, rounded,
            raw != bounded if raw is not None else None,
            bounded == self.min_green, bounded == self.max_green,
            rounded != bounded, rounded != bounded and rounded in (self.min_green,self.max_green),
            error is not None, error, self.active_provider, elapsed)
        # Retained for legacy scalar callers only; structured consumers never read these.
        self.last_error = error
        self.last_inference_ms = elapsed
        return result


if __name__ == "__main__":
    # Self-test and latency evaluation
    test_onnx = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../models/anfis_sugeno.onnx")
    )
    engine = ANFISInferenceEngine(test_onnx)

    # Test cases across various traffic conditions
    test_cases = [
        (5.0, 10.0, 5.0),    # Low traffic
        (25.0, 35.0, 20.0),  # Moderate traffic
        (75.0, 45.0, 45.0),  # Severe congestion
    ]

    print("\n--- ANFIS Inference Engine Test Results ---")
    for vw, q, l in test_cases:
        t_g = engine.evaluate_green_time(vw, q, l)
        print(f"Input: (V_w={vw:4.1f} PCU, Q={q:4.1f}m, L={l:4.1f}%) -> t_ANFIS = {t_g:5.1f}s | Latency: {engine.last_inference_ms:.3f}ms")
