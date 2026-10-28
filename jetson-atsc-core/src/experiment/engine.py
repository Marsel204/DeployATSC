from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import threading

from src.perception.signal_phase_observer import SignalPhaseObserver
from src.experiment.predictions import PredictionBook, QUALIFICATION_POLICY, LEGACY_QUALIFICATION_POLICY, qualification
from src.experiment.vehicle_waits import VehicleWaits
from src.experiment.summary import summarize


class ObservationEngine:
    def __init__(self, run_id, policy, roi, side, evaluate, lamp_rois=None, *, scale_evidence=None):
        self.run_id = run_id; self.policy = policy
        self.lock = threading.RLock(); self.time = 0.; self.closed = False
        self.phase = SignalPhaseObserver(roi, policy, lamp_rois)
        self.waits = VehicleWaits(run_id, policy, side)
        self.predictions = PredictionBook(run_id, policy, evaluate, lambda:self.time, scale_evidence=scale_evidence)
        self.system_snapshot = None
        self.processed = {"system":0, "acquisition":0}
        self.gaps = []
        self.revision = 0
        self.generation = 0
        self.retention = None

    def update_system(self, snapshot, *, generation=None):
        with self.lock:
            if self.closed or (generation is not None and generation != self.generation):return
            self.system_snapshot = snapshot
            self.time = max(self.time, snapshot.timestamp)
            self.processed["system"] += 1
            self.revision += 1

    def observe_system_frame(self, timestamp, *, generation=None):
        """A decoded Camera 1 preview frame: advances time only; no metrics, no inference."""
        with self.lock:
            if self.closed or (generation is not None and generation != self.generation):return False
            self.time = max(self.time, timestamp)
            self.processed["system"] += 1
            self.revision += 1
            return True

    def observe_acquisition(self, frame, timestamp, sequence, observations, *, generation=None):
        with self.lock:
            if self.closed or (generation is not None and generation != self.generation):return
            # Reject sequence retries before committing the clock or continuity gaps.
            if self.phase.last_sequence is not None and sequence <= self.phase.last_sequence:return
            if (self.phase.last_time is not None and sequence != self.phase.last_sequence and
                    timestamp-self.phase.last_time > self.policy.max_gap_sec):
                self.predictions.invalidate("observation_gap", self.phase.last_time)
            result = self.phase.observe(frame, timestamp, sequence)
            if result["duplicate"]: return
            self.time = max(self.time, timestamp)
            self.processed["acquisition"] += 1
            self.revision += 1
            event = result["transition"]
            if event and event["phase"] == "UNKNOWN":
                self.predictions.invalidate(event["reason"], event["timestamp"])
            if event and event["phase"] == "GREEN":
                self.predictions.on_green(event["timestamp"], self.phase.cycle_id)
            measured = result["measurement_phase"]
            cycle = None if measured == "UNKNOWN" else max(1,self.phase.cycle_id + (1 if measured == "RED" else 0))
            self.waits.observe(observations, measured, timestamp, sequence, cycle)

    def predict(self, request_id, timestamp=None, *, provenance=None, frame=None, analyze=None):
        with self.lock:
            if self.closed: raise ValueError("Run has ended")
            evidence = self.phase.status()
            evidence.update(candidate_onset=self.phase.candidate_time,last_observed_at=self.phase.last_time,
                            last_transition=deepcopy(self.phase.events[-1]) if self.phase.events else None)
            if analyze is not None:
                # Camera 1 on demand: frame identity, click time and measured phase freeze here.
                return self.predictions.capture_frame(frame, self.time if timestamp is None else timestamp,
                    self.phase.measurement_phase, request_id, analyze, display_phase=self.phase.phase,
                    phase_evidence=evidence, provenance=provenance)
            return self.predictions.capture(self.system_snapshot, self.time if timestamp is None else timestamp,
                self.phase.measurement_phase, request_id, display_phase=self.phase.phase, phase_evidence=evidence, provenance=provenance)

    def invalidate(self, reason, timestamp=None):
        with self.lock:
            if self.closed:return
            self.generation += 1
            t = self.time if timestamp is None else timestamp
            self.gaps.append({"timestamp":t,"reason":reason})
            self.phase.invalidate(t, reason)
            self.waits.invalidate(t, reason)
            self.predictions.invalidate(reason, t)
            self.revision += 1

    def comparisons(self, prediction_view=None):
        with self.lock:
            return self._comparisons(prediction_view if prediction_view is not None else self.predictions.view())

    def _comparisons(self, prediction_view):
        records = {row["cycle_id"]:row for row in prediction_view["rows"] if row["cycle_id"] is not None}
        cycles = deepcopy(self.phase.cycles)
        if self.phase.green:
            cycle = dict(self.phase.green)
            cycle["running_green_sec"] = max(0, self.time-cycle["green_start"])
            cycles.append(cycle)
        results = []
        self.comparison_visits = 0
        for cycle in cycles:
            prediction = records.get(cycle["cycle_id"])
            actual = cycle.get("actual_green_sec")
            recommended = prediction["recommended_green_sec"] if prediction else None
            values = []
            served = 0
            partial = 0
            for row in self.waits.records.values():
                self.comparison_visits += 1
                if cycle["cycle_id"] not in row["cycles"]: continue
                values.append(row["wait_by_cycle"].get(str(cycle["cycle_id"]), {}).get("red_wait_sec", 0.))
                end=cycle.get("green_end",self.time)
                served += bool(row["crossed_stop_line"] and row["crossing_cycle"]==cycle["cycle_id"] and
                               cycle["green_start"]<=row["completed_at"]<end)
                partial += bool(row["left_censored"] or row["right_censored"] or row["quality_flags"])
            if self.retention:
                population=self.retention.cycle_population.get(cycle["cycle_id"],{})
                count=len(values)+population.get("count",0)
                mean=(sum(values)+population.get("wait",0.))/count if count else None
                partial+=population.get("partial",0);served+=population.get("served",0)
                served+=sum(cycle["green_start"]<=t<cycle.get("green_end",self.time) for t in population.get("crossings",[]))
            else:
                count=len(values);mean=sum(values)/len(values) if values else None
            cycle.update(run_id=self.run_id, movement_id=self.policy.movement_id,
                recommended_green_sec=recommended, prediction=prediction,
                difference_sec=round(recommended-actual,6) if recommended is not None and actual is not None else None,
                eligible=bool(cycle["complete"] and prediction and prediction["eligible"]),
                observed_mean_red_wait_sec=round(mean,6) if mean is not None else None,
                observed_vehicle_count=count, partial_vehicle_count=partial, n_served=served)
            if prediction and (prediction.get("qualification") or {}).get("policy") in (QUALIFICATION_POLICY, LEGACY_QUALIFICATION_POLICY):
                cycle["qualification"] = qualification(prediction,complete_green=bool(cycle["complete"]))
            else:
                cycle["qualification"] = {"policy":QUALIFICATION_POLICY if prediction is None else None,"state":"unknown",
                                          "reasons":["prediction_qualification_missing"],"scale":"unverified"}
            cycle["qualified"] = bool(cycle["eligible"] and cycle["qualification"]["state"] == "qualified")
            results.append(cycle)
        return results

    def snapshot(self, offset=0, limit=100, evidence=False):
        with self.lock:
            view = self.predictions.view()
            comparisons = self.comparisons(view)
            summary = (self.retention.summary if self.retention else summarize)(self.waits.records.values(), comparisons, self.policy.los_method, view["rows"])
            if self.retention:self.retention.visited_records+=self.comparison_visits
            vehicles=self.waits.rows(0,len(self.waits.records),evidence)
            if self.retention:
                if not evidence:
                    for row in vehicles:row["history"]=deepcopy(self.retention.progress.get(row["vehicle_id"],{}))
                vehicles=list(deepcopy(self.retention.recent))+vehicles
            vehicles.sort(key=lambda r:(r["first_seen"],r["vehicle_id"]))
            result = {"schema_version":2, "run_id":self.run_id,
                "movement_id":self.policy.movement_id, "operation":"observation",
                "time":self.time, "processed_frames":dict(self.processed),
                "snapshot_revision":self.revision+view["revision"], "prediction_revision":view["revision"],
                "qualification_policy":QUALIFICATION_POLICY,
                "scale_trust":"verified" if all(self.predictions.scale_evidence.get(role,{}).get("state")=="verified" for role in ("system","acquisition")) else "unverified",
                # Continuous legacy snapshot if one exists, else the last analysed Camera 1 still image.
                "metrics":asdict(self.system_snapshot) if self.system_snapshot else deepcopy((view["last_analysis"] or {}).get("snapshot")),
                "camera1_last_analysis":deepcopy(view["last_analysis"]),
                "phase":self.phase.status(), "predictions":view["rows"],
                "comparisons":comparisons, "vehicles":vehicles[offset:offset+limit],
                "vehicle_total":summary["observed_vehicle_count"], "closed":self.closed, "generation":self.generation,
                "prediction_busy":view["busy"],
                "summary":summary,"los":summary["los"]}
            if evidence:
                result.update(policy=asdict(self.policy), phase_events=deepcopy(self.phase.events), observation_gaps=deepcopy(self.gaps))
            if self.retention:
                result["history"]={key:{"total":self.retention.archived_counts[key]+len(result.get(key,[])),
                    "retained":len(result.get(key,[])),"historical_endpoint":"/api/observation/history/"+key}
                    for key in ("vehicles","comparisons","predictions","phase_events","observation_gaps")}
                result["history"]["vehicles"]["total"]=result["vehicle_total"]
                result["summary"]["quantiles"]["snapshot_revision"]=result["snapshot_revision"]
            return result

    def enable_retention(self,budget=64,directory=None):
        from src.experiment.retention import Retention
        with self.lock:
            self.retention=Retention(budget,directory)
            self.waits.archived=self.retention.identity

    def retention_due(self):
        with self.lock:
            if self.retention is None or self.closed:return False
            work=sum(len(r["intervals"])+int(r["completed"]) for r in self.waits.records.values())
            work+=max(0,len(self.phase.events)-self.retention.budget)+max(0,len(self.gaps)-self.retention.budget)
            return work>=self.retention.budget

    def prepare_commit(self):
        with self.lock:
            state=self.snapshot(0,len(self.waits.records)+(self.retention.budget if self.retention else 0),True)
            return self.retention.prepare(self,state) if self.retention else (state,None)

    def acknowledge(self,token):
        # A commit in flight never suspends admission: acknowledgment evicts only
        # the prepared final rows and immutable prefixes; later evidence stays live.
        if token is not None:
            with self.lock:self.retention.acknowledge(self,token)

    def close(self, reason="recording_end"):
        with self.lock:
            if self.closed:return
            self.phase.invalidate(self.time, reason)
            self.waits.close(self.time, reason)
            self.closed = True
            self.generation += 1
            self.revision += 1
            self.predictions.close(reason)
