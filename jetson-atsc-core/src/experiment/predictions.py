from __future__ import annotations

from src.common.json_data import clone

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import math
import threading
import time

from src.common.source_identity import public_data, safe_text


class PredictionConflict(ValueError):
    pass


LEGACY_QUALIFICATION_POLICY = "motion-phase-v1"
QUALIFICATION_POLICY = "motion-phase-scale-v2"


def qualification(record, complete_green=None):
    """Keep each record under its original policy; v2 also requires scale binding.

    Called only for newly captured records. Historical rows without a policy
    are left unknown by readers, never retroactively promoted.
    """
    reasons = []
    # A single-frame capture has no snapshot until its one still image is analysed.
    quality = (record["snapshot"] or {}).get("quality") or {}
    missing = False
    for metric in ("V_w", "Q", "L"):
        evidence = quality.get(metric)
        if not evidence:
            missing = True
            reasons.append(f"{metric}_quality_missing")
        elif not (evidence.get("available") is True and evidence.get("complete") is True):
            reasons.append(f"{metric}_measurement_{evidence.get('state','unknown')}")
    if record["measurement_phase_at_capture"] != "RED":
        reasons.append("measurement_phase_not_red")
    if not record["eligible"]:
        reasons.append("timing_or_backend_ineligible")
    if complete_green is False:
        reasons.append("observed_green_incomplete")
    policy = (record.get("qualification") or {}).get("policy", QUALIFICATION_POLICY)
    scale = "unverified"
    if policy == QUALIFICATION_POLICY:
        evidence = record.get("scale_evidence") or {}
        for role in ("system", "acquisition"):
            if evidence.get(role, {}).get("state") != "verified":
                reasons.append(role+"_scale_review_required")
        if all(evidence.get(role, {}).get("state") == "verified" for role in ("system", "acquisition")):
            scale = "verified"
    return {"policy":policy,
            "state":"unknown" if missing else "diagnostic" if reasons else "qualified",
            "reasons":reasons,"scale":scale,"model_domain":"unknown"}


class PredictionBook:
    def __init__(self, run_id, policy, evaluate, clock, *, scale_evidence=None):
        self.run_id = run_id; self.policy = policy
        self.evaluate = evaluate; self.clock = clock
        self.scale_evidence = deepcopy(scale_evidence) or {}
        self.records = {}; self.pending = None
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ANFIS-observation")
        self.future = None; self.closed = False
        self.revision = 0; self.in_flight = False
        self.generation = 0
        self.last_analysis = None  # last published Camera 1 still-image analysis (not live)

    def capture(self, snapshot, timestamp, phase, request_id, *, display_phase=None, phase_evidence=None, provenance=None):
        with self.lock:
            if request_id in self.records: return deepcopy(self.records[request_id])
            if self.closed: raise PredictionConflict("This run is closed")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("A stable request ID is required")
            if self.pending or self.in_flight:
                raise PredictionConflict("A prediction is already pending; cancel it before another capture")
            if phase not in ("RED", "UNKNOWN"):
                raise PredictionConflict("Predict the upcoming green during red, not during green/yellow")
            if snapshot is None or not 0 <= timestamp-snapshot.timestamp <= self.policy.freshness_sec:
                raise ValueError("System-camera snapshot is stale or unavailable")
            snapshot = deepcopy(snapshot)
            if not all(math.isfinite(x) and x >= 0 for x in (snapshot.v_w_pcu, snapshot.occupancy_pct)) or (
                    snapshot.queue_meters is not None and
                    (not math.isfinite(snapshot.queue_meters) or snapshot.queue_meters < 0)):
                raise ValueError("Traffic metrics are invalid")
            record = {"request_id": request_id, "run_id": self.run_id,
                "movement_id": self.policy.movement_id, "snapshot": asdict(snapshot),
                "clicked_at": timestamp, "phase_at_capture": str(phase),
                "display_phase_at_capture": str(display_phase if display_phase is not None else phase),
                "measurement_phase_at_capture": str(phase),
                "phase_evidence":deepcopy(phase_evidence), "provenance":public_data(deepcopy(provenance)),
                "clicked_utc":datetime.now(timezone.utc).isoformat(),
                "completed_at": None, "inference_latency_ms": None,
                "recommended_green_sec": None, "status": "predicting", "cycle_id": None,
                "green_start": None, "click_lead_sec": None, "completed_lead_sec": None,
                "inference_diagnostics":None, "scale_evidence":deepcopy(self.scale_evidence),
                "eligible": False, "qualified":False, "error": None}
            record["qualification"] = qualification(record)
            self.records[request_id] = record; self.pending = request_id
            self.revision += 1; self.in_flight = True
            self.future = self.executor.submit(self._run, request_id, snapshot, self.generation)
            return deepcopy(record)

    def capture_frame(self, frame, timestamp, phase, request_id, analyze, *, display_phase=None, phase_evidence=None, provenance=None):
        """Admit one Camera 1 still image; ``analyze`` runs exactly once on this book's worker.

        Click time, measurement phase and frame identity are frozen here; inference and
        ANFIS run later outside every measurement/control lock.
        """
        with self.lock:
            if request_id in self.records: return deepcopy(self.records[request_id])
            if self.closed: raise PredictionConflict("This run is closed")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("A stable request ID is required")
            if self.pending or self.in_flight:
                raise PredictionConflict("A prediction is already pending; cancel it before another capture")
            if phase not in ("RED", "UNKNOWN"):
                raise PredictionConflict("Predict the upcoming green during red, not during green/yellow")
            if frame is None or not 0 <= timestamp-frame["timestamp"] <= self.policy.freshness_sec:
                raise ValueError("Camera 1 frame is stale or unavailable")
            frame = public_data(deepcopy(frame))
            record = {"request_id": request_id, "run_id": self.run_id,
                "movement_id": self.policy.movement_id, "snapshot": None,
                "metric_method": "camera1_single_frame_spatial_proxy", "camera1_frame": frame,
                "camera_analysis": {"state":"pending", "method":"single_frame_still_image", "inference_calls":0,
                                    "completed_at":None, "latency_ms":None, "lead_sec":None, "detections":None,
                                    "inference_api":None, "neural_device":None, "error":None},
                "clicked_at": timestamp, "phase_at_capture": str(phase),
                "display_phase_at_capture": str(display_phase if display_phase is not None else phase),
                "measurement_phase_at_capture": str(phase),
                "phase_evidence":deepcopy(phase_evidence), "provenance":public_data(deepcopy(provenance)),
                "clicked_utc":datetime.now(timezone.utc).isoformat(),
                "completed_at": None, "inference_latency_ms": None,
                "recommended_green_sec": None, "status": "predicting", "cycle_id": None,
                "green_start": None, "click_lead_sec": None, "completed_lead_sec": None,
                "inference_diagnostics":None, "scale_evidence":deepcopy(self.scale_evidence),
                "eligible": False, "qualified":False, "error": None}
            record["qualification"] = qualification(record)
            self.records[request_id] = record; self.pending = request_id
            self.revision += 1; self.in_flight = True
            self.future = self.executor.submit(self._run, request_id, None, self.generation, analyze)
            return deepcopy(record)

    def _run(self, request_id, snapshot, generation, analyze=None):
        start = time.perf_counter()
        started_utc=datetime.now(timezone.utc).isoformat()
        diagnostics = None; analysis = None; analysis_error = None
        try:
            if analyze is not None:
                # Exactly one still-image call per accepted capture; never retried here.
                try:snapshot, analysis = analyze()
                except Exception as failure:
                    analysis_error = safe_text(failure); raise
                analysis["completed_at"] = self.clock()
                if not all(math.isfinite(x) and x >= 0 for x in (snapshot.v_w_pcu, snapshot.occupancy_pct, snapshot.queue_meters)):
                    raise ValueError("Traffic metrics are invalid")
            if snapshot.queue_meters is None:
                raise ValueError("Queue measurement is unavailable; inference requires measured inputs")
            decision = self.evaluate(snapshot, 0)
            diagnostics = public_data(deepcopy(getattr(decision,"inference_diagnostics",None))) or {
                "diagnostics_state":"legacy_decision", "model_domain":"unknown",
                "backend_fallback":bool(decision.failsafe_active), "raw_green_sec":None}
            value = float(decision.green_seconds)
            if not math.isfinite(value) or not 10 <= value <= 120:
                raise ValueError("ANFIS output is outside the valid green bounds")
            if decision.failsafe_active or diagnostics.get("backend_fallback") is True:
                raise ValueError("ANFIS inference used a fallback; recommendation is invalid")
            error = None
        except Exception as failure:
            value = None; error = safe_text(failure)
        completed = self.clock()
        with self.lock:
            # Sealing (close/storage failure) rejects late publication of any result.
            if self.closed or generation != self.generation:return
            self.in_flight = False; self.revision += 1
            record = self.records[request_id]
            # Completion makes terminal rows eligible for bounded archival even
            # when their result was cancelled/invalidated during the native call.
            record.update(completed_at=completed, inference_started_utc=started_utc,
                          completed_utc=datetime.now(timezone.utc).isoformat(),
                          inference_latency_ms=round((time.perf_counter()-start)*1000, 3))
            if analyze is not None:
                # inference_calls counts the one attempted native call, even when it failed.
                result = {**record["camera_analysis"], "inference_calls":1, **(analysis or {}),
                          "error":analysis_error, "state":"failed" if analysis is None else "complete"}
                if record["status"] in ("cancelled", "unmatched"):
                    # Terminal before the native call returned: keep no stale metrics/recommendation.
                    result["state"] = "discarded_after_"+record["status"]
                    record.update(camera_analysis=result, late_result_discarded=True)
                    return
                record["camera_analysis"] = result
                if analysis is not None:
                    record["snapshot"] = asdict(snapshot)
                    self.last_analysis = {"request_id":request_id, "frame":deepcopy(record["camera1_frame"]),
                        "snapshot":deepcopy(record["snapshot"]), "completed_at":analysis["completed_at"]}
            record.update(recommended_green_sec=value, inference_diagnostics=diagnostics, error=error)
            if record["status"] in ("cancelled", "unmatched"): return
            if error:
                record["status"] = "failed"; self.pending = None
            elif record["green_start"] is not None: self._classify(record)
            else: record["status"] = "awaiting_green"
            record["qualification"] = qualification(record)
            record["qualified"] = record["qualification"]["state"] == "qualified"

    def _classify(self, record):
        if record["completed_at"] is None: return
        record["click_lead_sec"] = record["green_start"]-record["clicked_at"]
        # Total lead: green onset minus completion of the whole capture (camera + ANFIS).
        record["completed_lead_sec"] = record["green_start"]-record["completed_at"]
        analysis = record.get("camera_analysis")
        if analysis and analysis.get("completed_at") is not None:
            analysis["lead_sec"] = record["green_start"]-analysis["completed_at"]
        record["eligible"] = (record["phase_at_capture"] == "RED" and
                              record["completed_lead_sec"] > 0 and record["error"] is None)
        record["status"] = "matched" if record["eligible"] else "late" if record["completed_lead_sec"] <= 0 else "unaligned"
        record["qualification"] = qualification(record)
        record["qualified"] = record["qualification"]["state"] == "qualified"

    def on_green(self, timestamp, cycle_id):
        with self.lock:
            if self.closed:return
            if self.pending:
                record = self.records[self.pending]
                record.update(green_start=timestamp, cycle_id=cycle_id)
                self._classify(record)
                self.pending = None
                self.revision += 1

    def invalidate(self, reason, timestamp):
        with self.lock:
            if self.closed:return
            if self.pending:
                self.records[self.pending].update(status="unmatched", eligible=False, qualified=False,
                                                  invalidated_at=timestamp, invalidation_reason=reason)
                self.records[self.pending]["qualification"] = qualification(self.records[self.pending])
                self.pending = None
                self.revision += 1

    def cancel(self, request_id):
        with self.lock:
            if self.closed:raise PredictionConflict("This run is closed")
            if request_id not in self.records: raise KeyError(request_id)
            record = self.records[request_id]
            if record["cycle_id"] is not None: raise PredictionConflict("An observed-cycle prediction cannot be cancelled")
            # Terminal rows are immutable (and may already be selected for archival).
            if record["status"] not in ("predicting", "awaiting_green"):
                raise PredictionConflict("Only a pending capture can be cancelled")
            record.update(status="cancelled", eligible=False, qualified=False, cancelled_at=self.clock())
            record["qualification"] = qualification(record)
            if self.pending == request_id: self.pending = None
            self.revision += 1
            return deepcopy(record)

    def rows(self):
        with self.lock: return clone(list(self.records.values()))

    def view(self):
        """One atomic input for all representations of a telemetry snapshot."""
        with self.lock:
            return {"rows":clone(list(self.records.values())),
                    "busy":bool(self.pending or self.in_flight),"revision":self.revision,
                    "last_analysis":clone(self.last_analysis)}

    @property
    def resources_released(self):
        # Admission closure cannot cancel a running native inference call.
        with self.lock:return self.future is None or self.future.done()

    def await_idle(self, timeout=5):
        if self.future: self.future.result(timeout=timeout)

    def close(self, reason="recording_end"):
        with self.lock:
            if self.closed:return
            self.closed = True
            self.generation += 1
            # This is the book's admission slot, not resource-release evidence.
            # Late inference cannot update this closed book.
            self.in_flight = False
            self.revision += 1
            for record in self.records.values():
                if record["status"] in ("predicting", "awaiting_green"):
                    record.update(status="unmatched", eligible=False, qualified=False,
                                  invalidated_at=self.clock(), invalidation_reason=reason)
                    record["qualification"] = qualification(record)
                if (record.get("camera_analysis") or {}).get("state") == "pending":
                    # The native call may still be running; its result can never be published.
                    record["camera_analysis"]["state"] = "sealed_before_completion"
            self.pending = None
        self.executor.shutdown(wait=False, cancel_futures=True)
