from __future__ import annotations

from collections import deque
import math

from src.common.json_data import clone as deepcopy

from src.experiment.contracts import RunPolicy, vehicle_key


# Image-motion evidence (pixels, scaled by the vehicle's own box height).
HISTORY_SAMPLES = 8          # recent observations, gaps <= max_gap_sec only
MIN_FIT_SAMPLES = 3
MIN_FIT_SPAN_SEC = .1
MIN_APPROACH_HEIGHTS_PER_SEC = .5   # speed toward the line, in box heights/s
MIN_MOVING_HEIGHTS_PER_SEC = .5     # image speed that counts as motion evidence
OCCUPIED_HEIGHTS = .35       # another box this close to the projection hides ours
CONTINUATION_BASE_HEIGHTS = .35
CONTINUATION_HEIGHTS_PER_SEC = .3
MAX_HEIGHT_RATIO = 1.5
# Without moving-motion evidence (no fit, or a stationary fit) a newborn nearby
# is as likely a neighbour as the same vehicle, so only a flicker-length gap may
# be bridged (frame review: a stationary motorcycle hidden by a pedestrian for
# 0.4 s was otherwise linked to the neighbouring motorcycle).
STATIC_CONTINUATION_MAX_GAP_SEC = MIN_FIT_SPAN_SEC


def _fit(history):
    """Least-squares image velocity and approach rate over timed samples."""
    if len(history) < MIN_FIT_SAMPLES or history[-1][0]-history[0][0] < MIN_FIT_SPAN_SEC-1e-9:
        return None
    times = [h[0] for h in history]
    mean = sum(times)/len(times)
    denominator = sum((t-mean)**2 for t in times)
    if denominator <= 0:
        return None
    def slope(index):
        values = [h[index] for h in history]
        centre = sum(values)/len(values)
        return sum((t-mean)*(v-centre) for t, v in zip(times, values))/denominator
    return {'vx': slope(2), 'vy': slope(3), 'rate': slope(4)}


class VehicleWaits:
    """Integrate observed intervals. Missing observations are never imputed.

    Physical-vehicle persistence (both bounded and evidenced on the row):
    * native fragment continuation: a newborn native ID continues a vehicle
      lost <= max_gap_sec earlier only when each is the other's sole match;
    * departure inference: a moving vehicle vanishing just before the stop
      line is closed as a crossing marked ``dead_reckoned``; the hidden span
      is unknown coverage, never wait.
    """
    fragment_continuation = True

    def __init__(self, run_id, policy: RunPolicy, side):
        self.run_id = run_id
        self.policy = policy
        self.side = side
        self.records = {}; self.work = 0
        self.active = {}
        self.last_sequence = None
        self.first_sequence = None
        self.last_time = None
        # Derived lookup of identities whose rows were archived and evicted.
        self.archived = None
        # Native identity -> the continued row it handed its vehicle to.
        self.continued_from = {}

    def _flag(self, row, flag):
        if flag not in row["quality_flags"]: row["quality_flags"].append(flag)

    def _account(self, row, interval, multiplier=1):
        duration = interval["end"]-interval["start"]
        if interval["phase"] == "UNKNOWN":
            row["phase_unknown_sec"] += multiplier*duration
        if not interval["valid"]:
            row["unknown_sec"] += multiplier*duration
        elif interval["stopped"]:
            row["total_stopped_sec"] += multiplier*duration
            if interval["phase"] == "RED": row["red_wait_sec"] += multiplier*duration
            if interval["cycle_id"] is None:
                row["unattributed_stopped_sec"] += multiplier*duration
                return
            cycle = row["wait_by_cycle"].setdefault(str(interval["cycle_id"]), {"red_wait_sec":0., "total_stopped_sec":0.})
            cycle["total_stopped_sec"] += multiplier*duration
            if interval["phase"] == "RED":cycle["red_wait_sec"] += multiplier*duration

    def _revise_since(self, row, timestamp, stopped):
        for interval in reversed(row["intervals"]):
            if interval["start"] < timestamp-1e-8: break
            if interval["valid"] and interval["stopped"] != stopped:
                self._account(row, interval, -1)
                interval["stopped"] = stopped
                self._account(row, interval)

    def _new(self, observation, timestamp, sequence, segment=0):
        identity = (observation.epoch, observation.local_id)
        base = vehicle_key(self.run_id, *identity)
        # The identity's own row count, as in the full-history oracle.
        key = base if segment == 0 else f"{base}:segment-{segment}"
        reused = False
        while key in self.records:
            # This native ID's row now belongs to a continued physical vehicle.
            segment += 1; reused = True
            key = f"{base}:segment-{segment}"
        row = {"vehicle_id": key, "local_id": observation.local_id,
            "camera": "acquisition", "tracker_epoch": observation.epoch,
            "class_name": observation.class_name, "first_seen": timestamp,
            "last_seen": timestamp, "red_wait_sec": 0., "total_stopped_sec": 0.,
            "unknown_sec": 0., "phase_unknown_sec":0., "unattributed_stopped_sec":0.,
            "cycles": [], "quality_flags": [], "intervals": [],
            "wait_by_cycle": {},
            "left_censored": segment > 0 or sequence == self.first_sequence or (observation.velocity_valid and observation.speed_mps <= self.policy.stopped_speed_mps),
            "right_censored": True, "completed": False, "crossed_stop_line": False,
            "crossing_cycle": None, "crossing_at":None,"crossing_phase":None,
            "crossing_direction":None,"crossing_evidence":None,
            "completed_at": None, "tracking_state": "UNKNOWN"}
        if reused:
            row["quality_flags"].append("native_id_reused")
            continued = self.continued_from.get(identity)
            if continued is not None and continued.get("vehicle_id") in self.records:
                # The handed-over ID is back as a separate vehicle: the link is contradicted.
                self._flag(continued, "identity_link_contested")
                for link in continued.get("identity_links", []):
                    if (link["from"]["epoch"], link["from"]["local_id"]) == identity:link.setdefault("contested_at", timestamp)
        self.records[key] = row
        state = {"row": row, "previous": None, "stopped": False,
                 "stop_candidate": None, "move_candidate": None, "segments": segment+1,
                 "history": deque(maxlen=HISTORY_SAMPLES), "projection": None}
        self.active[identity] = state
        return state

    def _restore(self, identity, timestamp):
        """Memory forgot an archived identity; its derived history still counts."""
        base = vehicle_key(self.run_id, *identity)
        found = self.archived(base)
        if found is None:return None
        segments, crossed = found
        row = {"vehicle_id": base if segments == 1 else f"{base}:segment-{segments-1}",
               "completed": True, "crossed_stop_line": crossed, "last_seen": timestamp}
        state = {"row": row, "previous": None, "stopped": False, "stop_candidate": None,
                 "move_candidate": None, "segments": segments, "_last_seen": timestamp}
        self.active[identity] = state
        return state

    def _cross(self, row, previous, observation, phase, timestamp, cycle_id):
        crossing_phase = str(phase) if previous["phase"] != "UNKNOWN" else "UNKNOWN"
        attributed = cycle_id if crossing_phase != "UNKNOWN" else None
        if not row["completed"]:self.work+=1
        row.update(crossed_stop_line=True,crossing_cycle=attributed,
                   crossing_at=timestamp,crossing_phase=crossing_phase,crossing_direction=1,
                   crossing_evidence={"method":"observed","from_point":previous["point"],"to_point":observation.point,
                                      "from_time":previous["time"],"to_time":timestamp,
                                      "from_side":self.side(previous["point"]),"to_side":self.side(observation.point)},
                   completed=True,completed_at=timestamp,right_censored=False,tracking_state="SERVED")
        if attributed is None:self._flag(row,"crossing_phase_unknown")

    def observe(self, observations, phase, timestamp, sequence, cycle_id):
        if sequence == self.last_sequence or (self.last_time is not None and timestamp <= self.last_time): return
        if self.first_sequence is None: self.first_sequence = sequence
        frame_gap = self.last_time is not None and timestamp-self.last_time > self.policy.max_gap_sec+1e-9
        self.last_sequence = sequence; self.last_time = timestamp
        if phase == "UNKNOWN":cycle_id = None
        if frame_gap:
            # Absence during a camera/packet gap is not evidence of departure.
            for state in self.active.values():
                state["projection"] = None; state["unobserved_gap"] = True
        if self.fragment_continuation and self.policy.fragment_continuation:
            self._continue_fragments(observations, timestamp)
        present = set()
        for observation in observations:
            if observation.local_id is None or not observation.stable_id: continue
            identity = (observation.epoch, observation.local_id)
            present.add(identity)
            state = self.active.get(identity)
            if state is None and observation.in_region and self.archived is not None:
                state = self._restore(identity, timestamp)
            segment = 0
            if state and state["row"]["completed"]:
                if state["row"]["crossed_stop_line"]:
                    state["_last_seen"]=timestamp
                    row = state["row"]; evidence = row.get("crossing_evidence") or {}
                    if (evidence.get("method") == "dead_reckoned" and observation.in_region and
                            self.side(observation.point) < 0 and "contradicted_at" not in evidence):
                        # Seen again before the line under the same ID: the inferred
                        # event is contradicted. Kept (it may be archived) but flagged.
                        evidence["contradicted_at"] = timestamp
                        self._flag(row, "departure_inference_contradicted")
                    continue
                segment = state["segments"]
                state = None
            if not observation.in_region:
                if state and not state["row"]["completed"]:
                    row=state["row"];previous=state["previous"]
                    crossing=bool(previous and previous["sequence"]==sequence-1 and
                        0<timestamp-previous["time"]<=self.policy.max_gap_sec and
                        self.side(previous["point"])<0<=self.side(observation.point))
                    if crossing:self._cross(row,previous,observation,phase,timestamp,cycle_id)
                    else:
                        row.update(completed=True,completed_at=timestamp,tracking_state="EXITED",right_censored=False);self.work+=1
                continue
            if state is None: state = self._new(observation, timestamp, sequence, segment)
            row = state["row"]; previous = state["previous"]
            valid_motion = observation.velocity_valid and observation.speed_mps is not None
            continuous = previous is not None and previous["sequence"] == sequence-1 and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec
            if previous is not None:
                interval = {"start": previous["time"], "end": timestamp,
                            "phase": previous["phase"] if continuous else "UNKNOWN",
                            "cycle_id": previous["cycle"] if continuous else None,
                            "valid": bool(continuous and previous["valid_motion"] and valid_motion),
                            "stopped": state["stopped"]}
                row["intervals"].append(interval); self.work+=1; self._account(row, interval)
                if not continuous:
                    self._flag(row, "observation_gap")
                    state["stop_candidate"] = state["move_candidate"] = None
                if continuous and self.side(previous["point"]) < 0 <= self.side(observation.point):
                    self._cross(row,previous,observation,phase,timestamp,cycle_id)
            if valid_motion:
                speed = observation.speed_mps
                if not state["stopped"]:
                    state["move_candidate"] = None
                    if speed <= self.policy.stopped_speed_mps:
                        if state["stop_candidate"] is None: state["stop_candidate"] = timestamp
                        if timestamp-state["stop_candidate"] >= self.policy.stop_persistence_sec-1e-8:
                            self._revise_since(row, state["stop_candidate"], True)
                            state["stopped"] = True; state["stop_candidate"] = None
                    else: state["stop_candidate"] = None
                else:
                    state["stop_candidate"] = None
                    if speed >= self.policy.moving_speed_mps:
                        if state["move_candidate"] is None: state["move_candidate"] = timestamp
                        if timestamp-state["move_candidate"] >= self.policy.stop_persistence_sec-1e-8:
                            self._revise_since(row, state["move_candidate"], False)
                            state["stopped"] = False; state["move_candidate"] = None
                    else: state["move_candidate"] = None
                if not row["completed"]: row["tracking_state"] = "STOPPED" if state["stopped"] else "MOVING"
            else:
                self._flag(row, "velocity_unknown")
                state["stop_candidate"] = state["move_candidate"] = None
                if not row["completed"]: row["tracking_state"] = "UNKNOWN"
            if phase == "UNKNOWN": self._flag(row, "phase_unknown")
            if cycle_id is not None and cycle_id not in row["cycles"]: row["cycles"].append(cycle_id)
            row["last_seen"] = timestamp
            history = state.setdefault("history", deque(maxlen=HISTORY_SAMPLES))
            # Samples stay comparable across short native flicker (<= max_gap_sec);
            # a longer or unknown gap starts a new motion history.
            recent = previous is not None and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec+1e-9
            if not recent or not observation.height or observation.height <= 0: history.clear()
            if observation.height and observation.height > 0:
                history.append((timestamp, sequence, float(observation.point[0]), float(observation.point[1]),
                                float(self.side(observation.point)), float(observation.height)))
            state["projection"] = None; state["unobserved_gap"] = False
            state["previous"] = {"time": timestamp, "sequence": sequence,
                "phase": str(phase), "cycle": cycle_id, "valid_motion": valid_motion,
                "point": observation.point}
        for identity, state in list(self.active.items()):
            row = state["row"]
            if row["vehicle_id"] not in self.records and identity not in present and timestamp-state.get("_last_seen",row["last_seen"])>self.policy.association_gap_sec:
                del self.active[identity]
                continue
            if identity not in present and not row["completed"]:
                row["tracking_state"] = "LOST"
                if self._infer_departure(identity, state, observations, timestamp):continue
                if timestamp-row["last_seen"] > self.policy.association_gap_sec:
                    row.update(completed=True, completed_at=timestamp);self.work+=1
                    self._flag(row, "tracking_lost")

    def physical(self, identity):
        """The physical vehicle row a raw (epoch, native ID) currently drives."""
        state = self.active.get(identity)
        if state is None or state["row"].get("vehicle_id") not in self.records:return None
        row = state["row"]
        return {"vehicle_id": row["vehicle_id"], "local_id": row["local_id"],
                "tracker_epoch": row["tracker_epoch"], "native": {"epoch": identity[0], "local_id": identity[1]},
                "continued": bool(row.get("identity_links"))}

    def _continue_fragments(self, observations, timestamp):
        """Link a newborn native ID to a just-lost vehicle when the pair is mutually unique."""
        stable = [o for o in observations if o.local_id is not None and o.stable_id]
        present = {(o.epoch, o.local_id) for o in stable}
        newborns = []
        for o in stable:
            identity = (o.epoch, o.local_id)
            if identity in self.active or not o.in_region or not o.height or o.height <= 0:continue
            if self.archived is not None and self.archived(vehicle_key(self.run_id, *identity)) is not None:continue
            newborns.append(o)
        if not newborns:return
        candidates = []
        for identity, state in self.active.items():
            row, previous, history = state["row"], state["previous"], state.get("history")
            if identity in present or row["completed"] or previous is None or not history:continue
            if "red_wait_sec" not in row or row["vehicle_id"] not in self.records:continue
            dt = timestamp-previous["time"]
            if not 0 < dt <= self.policy.max_gap_sec+1e-9:continue
            candidates.append((identity, state, dt))
        if not candidates:return
        matches = {}
        for o in newborns:
            for identity, state, dt in candidates:
                row = state["row"]; last = state["history"][-1]
                if identity[0] != o.epoch or row["class_name"] != o.class_name:continue
                height = last[5]
                if max(height, o.height)/min(height, o.height) > MAX_HEIGHT_RATIO:continue
                motion = _fit(state["history"])
                fitted = motion is not None and math.hypot(motion["vx"], motion["vy"]) >= MIN_MOVING_HEIGHTS_PER_SEC*height
                if not fitted:
                    if dt > STATIC_CONTINUATION_MAX_GAP_SEC+1e-9:continue
                    motion = {"vx": 0., "vy": 0.}
                    gate = CONTINUATION_BASE_HEIGHTS
                else:
                    gate = CONTINUATION_BASE_HEIGHTS+CONTINUATION_HEIGHTS_PER_SEC*dt
                predicted = (last[2]+motion["vx"]*dt, last[3]+motion["vy"]*dt)
                error = math.dist(predicted, o.point)/height
                if error <= gate:
                    matches[(o.epoch, o.local_id), identity] = (o, state, dt, error, gate, predicted, fitted)
        for (new, old), (o, state, dt, error, gate, predicted, fitted) in matches.items():
            if sum(1 for n, _ in matches if n == new) != 1 or sum(1 for _, c in matches if c == old) != 1:continue
            row = state["row"]
            del self.active[old]
            self.active[new] = state
            state["projection"] = None
            native = {"epoch": new[0], "local_id": new[1]}
            # Added on the first link only, so rows of unchanged runs keep their schema.
            row.setdefault("native_ids", [{"epoch": row["tracker_epoch"], "local_id": row["local_id"]}])
            if native not in row["native_ids"]:row["native_ids"].append(native)
            row.setdefault("identity_links", []).append({"from": {"epoch": old[0], "local_id": old[1]}, "to": native,
                "lost_at": state["previous"]["time"], "linked_at": timestamp, "gap_sec": dt,
                "predicted_point": list(predicted), "observed_point": list(o.point),
                "error_heights": error, "gate_heights": gate, "moving_fit": fitted,
                "rule": "mutually_unique_motion_gate"})
            self._flag(row, "identity_continued")
            self.continued_from[old] = row
            if len(self.continued_from) > 4*max(64, len(self.records)):
                self.continued_from = {k: r for k, r in self.continued_from.items() if r.get("vehicle_id") in self.records}

    def _infer_departure(self, identity, state, observations, timestamp):
        """Close a vanished, approaching vehicle as a dead-reckoned stop-line crossing."""
        row = state["row"]
        if (not self.policy.departure_inference or "red_wait_sec" not in row or state.get("previous") is None
                or state.get("unobserved_gap")):
            return False
        projection = state.get("projection")
        if projection is None:
            history = state.get("history")
            motion = _fit(history) if history and history[-1][0] == state["previous"]["time"] else None
            if motion is None or state["stopped"]:return False
            time, _, x, y, side, height = history[-1]
            if side >= 0 or motion["rate"] < MIN_APPROACH_HEIGHTS_PER_SEC*height:return False
            delay = -side/motion["rate"]
            if delay > self.policy.max_gap_sec+1e-9:return False
            projection = state["projection"] = {"last_time": time, "last_point": [x, y], "last_side": side,
                "height": height, "vx": motion["vx"], "vy": motion["vy"], "rate": motion["rate"],
                "samples": len(history), "crossing_at": time+delay, "blocked": False}
        if projection["blocked"]:return False
        elapsed = timestamp-projection["last_time"]
        predicted = (projection["last_point"][0]+projection["vx"]*elapsed,
                     projection["last_point"][1]+projection["vy"]*elapsed)
        for o in observations:
            if o.local_id is not None and (o.epoch, o.local_id) == identity:continue
            if math.dist(predicted, o.point) <= OCCUPIED_HEIGHTS*projection["height"]:
                projection["blocked"] = True
                self._flag(row, "departure_inference_blocked")
                return False
        due = timestamp >= projection["crossing_at"]+self.policy.max_gap_sec-1e-9
        lost = timestamp-row["last_seen"] > self.policy.association_gap_sec and timestamp >= projection["crossing_at"]
        if not (due or lost):return False
        crossing = projection["crossing_at"]
        if crossing > row["last_seen"]:
            interval = {"start": row["last_seen"], "end": crossing, "phase": "UNKNOWN", "cycle_id": None,
                        "valid": False, "stopped": False}
            row["intervals"].append(interval); self.work += 1; self._account(row, interval)
        previous = state["previous"]
        row.update(crossed_stop_line=True, crossing_cycle=None, crossing_at=crossing, crossing_phase="UNKNOWN",
                   crossing_direction=1, completed=True, completed_at=crossing, right_censored=False,
                   tracking_state="SERVED", crossing_evidence={"method": "dead_reckoned",
                   "last_point": projection["last_point"], "last_time": projection["last_time"],
                   "last_side": projection["last_side"], "approach_px_per_sec": projection["rate"],
                   "image_velocity_px_per_sec": [projection["vx"], projection["vy"]],
                   "box_height_px": projection["height"], "samples": projection["samples"],
                   "predicted_crossing_at": crossing, "decided_at": timestamp,
                   "last_observed_phase": previous["phase"], "last_observed_cycle": previous["cycle"]})
        self.work += 1
        self._flag(row, "departure_inferred"); self._flag(row, "crossing_phase_unknown")
        return True

    def invalidate(self, timestamp, reason):
        for state in self.active.values():
            row = state["row"]
            if not row["completed"]:
                self._flag(row, reason)
                state["previous"] = None
                state["stop_candidate"] = state["move_candidate"] = None
                if "history" in state: state["history"].clear()
                state["projection"] = None
                row["tracking_state"] = "UNKNOWN"

    def close(self, timestamp, reason="recording_end"):
        self.invalidate(timestamp, reason)
        for row in self.records.values():
            if not row["completed"]: row.update(completed=True, completed_at=timestamp);self.work+=1

    def rows(self, offset=0, limit=100, include_intervals=False):
        rows = list(self.records.values())
        return [{k: (round(v, 6) if isinstance(v, float) else deepcopy(v))
                 for k, v in row.items() if include_intervals or k != "intervals"}
                for row in rows[offset:offset+limit]]
