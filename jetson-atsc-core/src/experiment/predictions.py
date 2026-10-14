from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import math
import threading
import time


class PredictionConflict(ValueError):
    pass


class PredictionBook:
    def __init__(self, run_id, policy, evaluate, clock):
        self.run_id = run_id; self.policy = policy
        self.evaluate = evaluate; self.clock = clock
        self.records = {}; self.pending = None
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ANFIS-observation")
        self.future = None; self.closed = False

    def capture(self, snapshot, timestamp, phase, request_id):
        with self.lock:
            if request_id in self.records: return deepcopy(self.records[request_id])
            if self.closed: raise PredictionConflict("This run is closed")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("A stable request ID is required")
            if self.pending or (self.future and not self.future.done()):
                raise PredictionConflict("A prediction is already pending; cancel it before another capture")
            if phase not in ("RED", "UNKNOWN"):
                raise PredictionConflict("Predict the upcoming green during red, not during green/yellow")
            if snapshot is None or not 0 <= timestamp-snapshot.timestamp <= self.policy.freshness_sec:
                raise ValueError("System-camera snapshot is stale or unavailable")
            if not all(math.isfinite(x) and x >= 0 for x in (snapshot.v_w_pcu, snapshot.queue_meters, snapshot.occupancy_pct)):
                raise ValueError("Traffic metrics are invalid")
            record = {"request_id": request_id, "run_id": self.run_id,
                "movement_id": self.policy.movement_id, "snapshot": asdict(snapshot),
                "clicked_at": timestamp, "phase_at_capture": str(phase),
                "clicked_utc":datetime.now(timezone.utc).isoformat(),
                "completed_at": None, "inference_latency_ms": None,
                "recommended_green_sec": None, "status": "predicting", "cycle_id": None,
                "green_start": None, "click_lead_sec": None, "completed_lead_sec": None,
                "eligible": False, "error": None}
            self.records[request_id] = record; self.pending = request_id
            self.future = self.executor.submit(self._run, request_id, snapshot)
            return deepcopy(record)

    def _run(self, request_id, snapshot):
        start = time.perf_counter()
        started_utc=datetime.now(timezone.utc).isoformat()
        try:
            decision = self.evaluate(snapshot, 0)
            value = float(decision.green_seconds)
            if not math.isfinite(value) or not 10 <= value <= 120:
                raise ValueError("ANFIS output is outside the valid green bounds")
            if decision.failsafe_active:
                raise ValueError("ANFIS inference used a fallback; recommendation is invalid")
            error = None
        except Exception as failure:
            value = None; error = str(failure)
        completed = self.clock()
        with self.lock:
            if self.closed:return
            record = self.records[request_id]
            record.update(completed_at=completed, recommended_green_sec=value,
                          inference_started_utc=started_utc,completed_utc=datetime.now(timezone.utc).isoformat(),
                          inference_latency_ms=round((time.perf_counter()-start)*1000, 3), error=error)
            if record["status"] in ("cancelled", "unmatched"): return
            if error:
                record["status"] = "failed"; self.pending = None
            elif record["green_start"] is not None: self._classify(record)
            else: record["status"] = "awaiting_green"

    def _classify(self, record):
        if record["completed_at"] is None: return
        record["click_lead_sec"] = record["green_start"]-record["clicked_at"]
        record["completed_lead_sec"] = record["green_start"]-record["completed_at"]
        record["eligible"] = (record["phase_at_capture"] == "RED" and
                              record["completed_lead_sec"] > 0 and record["error"] is None)
        record["status"] = "matched" if record["eligible"] else "late" if record["completed_lead_sec"] <= 0 else "unaligned"

    def on_green(self, timestamp, cycle_id):
        with self.lock:
            if self.pending:
                record = self.records[self.pending]
                record.update(green_start=timestamp, cycle_id=cycle_id)
                self._classify(record)
                self.pending = None

    def cancel(self, request_id):
        with self.lock:
            if request_id not in self.records: raise KeyError(request_id)
            record = self.records[request_id]
            if record["cycle_id"] is not None: raise PredictionConflict("An observed-cycle prediction cannot be cancelled")
            record.update(status="cancelled", eligible=False, cancelled_at=self.clock())
            if self.pending == request_id: self.pending = None
            return deepcopy(record)

    def rows(self):
        with self.lock: return deepcopy(list(self.records.values()))

    def await_idle(self, timeout=5):
        if self.future: self.future.result(timeout=timeout)

    def close(self):
        with self.lock:
            self.closed = True
            if self.pending:
                self.records[self.pending].update(status="unmatched", eligible=False)
                self.pending = None
        self.executor.shutdown(wait=False, cancel_futures=True)
