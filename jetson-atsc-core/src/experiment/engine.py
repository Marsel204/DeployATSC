from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import threading

from src.perception.signal_phase_observer import SignalPhaseObserver
from src.experiment.predictions import PredictionBook
from src.experiment.vehicle_waits import VehicleWaits
from src.experiment.summary import summarize


class ObservationEngine:
    def __init__(self, run_id, policy, roi, side, evaluate, lamp_rois=None):
        self.run_id = run_id; self.policy = policy
        self.lock = threading.RLock(); self.time = 0.; self.closed = False
        self.phase = SignalPhaseObserver(roi, policy, lamp_rois)
        self.waits = VehicleWaits(run_id, policy, side)
        self.predictions = PredictionBook(run_id, policy, evaluate, lambda:self.time)
        self.system_snapshot = None
        self.processed = {"system":0, "acquisition":0}
        self.gaps = []

    def update_system(self, snapshot):
        with self.lock:
            self.system_snapshot = snapshot
            self.time = max(self.time, snapshot.timestamp)
            self.processed["system"] += 1

    def observe_acquisition(self, frame, timestamp, sequence, observations):
        with self.lock:
            self.time = max(self.time, timestamp)
            if (self.phase.last_time is not None and sequence != self.phase.last_sequence and
                    timestamp-self.phase.last_time > self.policy.max_gap_sec):
                self.predictions.invalidate("observation_gap", self.phase.last_time)
            result = self.phase.observe(frame, timestamp, sequence)
            if result["duplicate"]: return
            self.processed["acquisition"] += 1
            event = result["transition"]
            if event and event["phase"] == "UNKNOWN":
                self.predictions.invalidate(event["reason"], event["timestamp"])
            if event and event["phase"] == "GREEN":
                self.predictions.on_green(event["timestamp"], self.phase.cycle_id)
            cycle = self.phase.cycle_id + (1 if self.phase.phase in ("RED", "UNKNOWN") else 0)
            self.waits.observe(observations, result["measurement_phase"], timestamp, sequence, max(1,cycle))

    def predict(self, request_id, timestamp=None):
        with self.lock:
            if self.closed: raise ValueError("Run has ended")
            return self.predictions.capture(self.system_snapshot, self.time if timestamp is None else timestamp, self.phase.phase, request_id)

    def invalidate(self, reason, timestamp=None):
        with self.lock:
            t = self.time if timestamp is None else timestamp
            self.gaps.append({"timestamp":t,"reason":reason})
            self.phase.invalidate(t, reason)
            self.waits.invalidate(t, reason)
            self.predictions.invalidate(reason, t)

    def comparisons(self):
        records = {row["cycle_id"]:row for row in self.predictions.rows() if row["cycle_id"] is not None}
        cycles = deepcopy(self.phase.cycles)
        if self.phase.green:
            cycle = dict(self.phase.green)
            cycle["running_green_sec"] = max(0, self.time-cycle["green_start"])
            cycles.append(cycle)
        results = []
        for cycle in cycles:
            prediction = records.get(cycle["cycle_id"])
            actual = cycle.get("actual_green_sec")
            recommended = prediction["recommended_green_sec"] if prediction else None
            values = []
            served = 0
            partial = 0
            for row in self.waits.records.values():
                if cycle["cycle_id"] not in row["cycles"]: continue
                values.append(row["wait_by_cycle"].get(str(cycle["cycle_id"]), {}).get("red_wait_sec", 0.))
                end=cycle.get("green_end",self.time)
                served += bool(row["crossed_stop_line"] and row["crossing_cycle"]==cycle["cycle_id"] and
                               cycle["green_start"]<=row["completed_at"]<end)
                partial += bool(row["left_censored"] or row["right_censored"] or row["quality_flags"])
            cycle.update(run_id=self.run_id, movement_id=self.policy.movement_id,
                recommended_green_sec=recommended, prediction=prediction,
                difference_sec=round(recommended-actual,6) if recommended is not None and actual is not None else None,
                eligible=bool(cycle["complete"] and prediction and prediction["eligible"]),
                observed_mean_red_wait_sec=round(sum(values)/len(values),6) if values else None,
                observed_vehicle_count=len(values), partial_vehicle_count=partial, n_served=served)
            results.append(cycle)
        return results

    def snapshot(self, offset=0, limit=100, evidence=False):
        with self.lock:
            comparisons = self.comparisons()
            summary = summarize(self.waits.records.values(), comparisons, self.policy.los_method)
            result = {"schema_version":1, "run_id":self.run_id,
                "movement_id":self.policy.movement_id, "operation":"observation",
                "time":self.time, "processed_frames":dict(self.processed),
                "metrics":asdict(self.system_snapshot) if self.system_snapshot else None,
                "phase":self.phase.status(), "predictions":self.predictions.rows(),
                "comparisons":comparisons, "vehicles":self.waits.rows(offset,limit,evidence),
                "vehicle_total":len(self.waits.records), "closed":self.closed,
                "prediction_busy":bool(self.predictions.pending or (self.predictions.future and not self.predictions.future.done())),
                "summary":summary,"los":summary["los"]}
            if evidence:
                result.update(policy=asdict(self.policy), phase_events=deepcopy(self.phase.events), observation_gaps=deepcopy(self.gaps))
            return result

    def close(self, reason="recording_end"):
        with self.lock:
            if self.closed:return
            self.phase.invalidate(self.time, reason)
            self.waits.close(self.time, reason)
            self.closed = True
            self.predictions.close()
