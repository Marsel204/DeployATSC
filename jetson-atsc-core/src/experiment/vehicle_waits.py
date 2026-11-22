from __future__ import annotations

from collections import deque
import math

from src.common.json_data import clone as deepcopy

from src.experiment.contracts import RunPolicy, vehicle_key


# Opt-in native fragment continuation (RunPolicy.fragment_continuation).
# Image-motion evidence is in pixels scaled by the vehicle's own box height.
HISTORY_SAMPLES = 8          # recent observations of one raw ID, gaps <= max_gap_sec
MIN_FIT_SAMPLES = 3
MIN_FIT_SPAN_SEC = .1
MIN_MOVING_HEIGHTS_PER_SEC = .5     # image speed that counts as motion evidence
CONTINUATION_BASE_HEIGHTS = .35
CONTINUATION_HEIGHTS_PER_SEC = .3
MAX_HEIGHT_RATIO = 1.5
# Without moving-motion evidence a newborn nearby is as likely a neighbour as
# the same vehicle, so only a flicker-length gap may be bridged.
STATIC_CONTINUATION_MAX_GAP_SEC = MIN_FIT_SPAN_SEC
# Another observed box (any class, size or ID state) covering this much of the
# smaller of it and the newborn, with footpoints this close (in smaller-box
# heights), may be the same body or its neighbour: the link is ambiguous.
COMPETITOR_OVERLAP = .5
COMPETITOR_FOOTPOINT_HEIGHTS = .5
# Bounded work and memory: one row is continued at most this many times, and a
# frame with more newborn x lost pairs than this skips continuation entirely.
MAX_LINKS_PER_ROW = 8
MAX_PAIR_CHECKS = 4096


def _height(observation):
    height = observation.height
    if height is None and observation.bbox is not None:
        height = observation.bbox[3]-observation.bbox[1]
    try:height = float(height)
    except (TypeError, ValueError):return None
    return height if math.isfinite(height) and height > 0 else None


def _fit(history):
    """Least-squares image velocity over timed (t, x, y, height) samples."""
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
    return slope(1), slope(2)


def _raw_box(box):
    if isinstance(box, (str, bytes)):return None
    try:x1, y1, x2, y2 = (float(v) for v in box[:4])
    except (TypeError, ValueError, IndexError):return None
    return (x1, y1, x2, y2) if all(map(math.isfinite, (x1, y1, x2, y2))) and x2 > x1 and y2 > y1 else None


def _competitors(observations, unassigned):
    """(observation or None, footpoint, height, bbox) for every box of the frame,
    or None when any supplied box is unreadable (fail closed).

    An observation competes with its own box and, when it carries one, with
    the raw detection the tracker emitted it from (``source_bbox``): smoothing
    can move a tracked box out of a gate its detection is in. Both entries
    belong to the observation, so a newborn's own source never vetoes it.
    ``unassigned`` are the frame's actual detections the tracker did not emit:
    they can only veto a link, never become rows."""
    found = []
    for o in observations:
        box = None
        if o.bbox is not None:
            box = _raw_box(o.bbox)
            if box is None:return None
        found.append((o, o.point, _height(o), box))
        raw_source = getattr(o, "source_bbox", None)
        if raw_source is not None:
            source = _raw_box(raw_source)
            if source is None:return None
            if source != box:
                found.append((o, ((source[0]+source[2])/2, source[3]), source[3]-source[1], source))
    if isinstance(unassigned, (str, bytes)):return None
    try:
        for raw in unassigned:
            box = _raw_box(raw)
            if box is None:return None
            found.append((None, ((box[0]+box[2])/2, box[3]), box[3]-box[1], box))
    except TypeError:
        return None
    return found


def _overlap_of_smaller(a, b):
    width = min(a[2], b[2])-max(a[0], b[0]); height = min(a[3], b[3])-max(a[1], b[1])
    if width <= 0 or height <= 0:return 0.
    smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
    return width*height/smaller if smaller > 0 else 0.


class VehicleWaits:
    """Integrate observed intervals. Missing observations are never imputed.

    Opt-in native fragment continuation (``RunPolicy.fragment_continuation``):
    a never-seen native ID continues a vehicle row lost <= max_gap_sec earlier
    only when the pair is mutually unique on position alone, the class and box
    size agree, and no other observed box competes. The interval across the
    handoff is unknown coverage (no wait, no crossing, no motion history), and
    every raw ID keeps its own segment and crossing history.

    ``unassigned`` boxes (``observe``) are the frame's other actual detections
    that the tracker withheld; they only veto links. ``None`` declares that
    evidence unavailable, and then no link is made in that frame.
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
        # Raw IDs that handed their row to a continuation: {identity: {"row_id", "segments"}}.
        # Pruned once the row is evicted; the archive index answers after that.
        self.handed_off = {}
        # Stable raw IDs recently seen outside the region (never "newborn").
        self.outside_seen = {}
        self.continuation_skipped_frames = 0
        # Frames with a continuation candidate but no complete same-frame detections.
        self.continuation_unavailable_frames = 0
        # Persistent identities record per-interval tracking continuity for
        # acquisition collection; legacy exports keep their historical schema.
        self.record_tracking = False

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
        self.records[key] = row
        state = {"row": row, "previous": None, "stopped": False,
                 "stop_candidate": None, "move_candidate": None, "segments": segment+1}
        self.active[identity] = state
        return state

    def _restore(self, identity, timestamp):
        """Memory forgot an archived identity; its derived history still counts."""
        base = vehicle_key(self.run_id, *identity)
        found = self.archived(base)
        if found is None:return None
        segments, crossed = found
        # segments == 0: the raw ID only ever continued another row (an alias).
        row = {"vehicle_id": base if segments == 1 else f"{base}:segment-{segments-1}" if segments else f"{base}:alias",
               "tracker_epoch": identity[0], "local_id": identity[1],
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
                   crossing_evidence={"from_point":previous["point"],"to_point":observation.point,
                                      "from_time":previous["time"],"to_time":timestamp,
                                      "from_side":self.side(previous["point"]),"to_side":self.side(observation.point)},
                   completed=True,completed_at=timestamp,right_censored=False,tracking_state="SERVED")
        if attributed is None:self._flag(row,"crossing_phase_unknown")

    def observe(self, observations, phase, timestamp, sequence, cycle_id, unassigned=None):
        if sequence == self.last_sequence or (self.last_time is not None and timestamp <= self.last_time): return
        if self.first_sequence is None: self.first_sequence = sequence
        self.last_sequence = sequence; self.last_time = timestamp
        if phase == "UNKNOWN":cycle_id = None
        continuing = self.fragment_continuation and self.policy.fragment_continuation
        if continuing:
            # Read more than once (prepasses, accounting): a one-shot iterator is materialized once.
            observations = observations if isinstance(observations, (list, tuple)) else list(observations)
            # A sighting older than the association timeout no longer makes an ID "not newborn".
            if self.outside_seen:
                self.outside_seen = {k: t for k, t in self.outside_seen.items()
                                     if timestamp-t <= self.policy.association_gap_sec}
            # A handed-off ID seen again contests the live link before any row of this
            # frame completes (e.g. the continuing ID crossing the line): order-independent.
            if self.handed_off:
                for observation in observations:
                    if observation.local_id is None or not observation.stable_id:continue
                    identity = (observation.epoch, observation.local_id)
                    if identity in self.handed_off:
                        self._contest(self.handed_off[identity]["row_id"], identity, timestamp)
            self._continue_fragments(observations, timestamp, unassigned)
        present = set()
        for observation in observations:
            if observation.local_id is None or not observation.stable_id: continue
            identity = (observation.epoch, observation.local_id)
            present.add(identity)
            state = self.active.get(identity)
            segment = 0; seen_before = False
            if state is None and identity in self.handed_off:
                handed = self.handed_off[identity]
                self._contest(handed["row_id"], identity, timestamp)
                if not observation.in_region:continue
                # The raw ID is back as its own vehicle: its own next segment.
                del self.handed_off[identity]
                segment = handed["segments"]; seen_before = True
            elif state is None and observation.in_region and self.archived is not None:
                state = self._restore(identity, timestamp)
            if state and state["row"]["completed"]:
                if state["row"]["crossed_stop_line"]:
                    state["_last_seen"]=timestamp
                    continue
                segment = self._own_segments(identity, state); seen_before = True
                state = None
            if not observation.in_region:
                if state is None and continuing:self.outside_seen[identity] = timestamp
                if state and not state["row"]["completed"]:
                    row=state["row"];previous=state["previous"]
                    crossing=bool(previous and not previous.get("handoff") and previous["sequence"]==sequence-1 and
                        0<timestamp-previous["time"]<=self.policy.max_gap_sec and
                        self.side(previous["point"])<0<=self.side(observation.point))
                    if crossing:self._cross(row,previous,observation,phase,timestamp,cycle_id)
                    else:
                        row.update(completed=True,completed_at=timestamp,tracking_state="EXITED",right_censored=False);self.work+=1
                continue
            if state is None:
                state = self._new(observation, timestamp, sequence, segment)
                if seen_before:state["row"]["left_censored"] = True
            row = state["row"]; previous = state["previous"]
            valid_motion = observation.velocity_valid and observation.speed_mps is not None
            # A heuristic ID handoff is never continuous, even on adjacent frames.
            continuous = previous is not None and not previous.get("handoff") and previous["sequence"] == sequence-1 and 0 < timestamp-previous["time"] <= self.policy.max_gap_sec
            if previous is not None:
                interval = {"start": previous["time"], "end": timestamp,
                            "phase": previous["phase"] if continuous else "UNKNOWN",
                            "cycle_id": previous["cycle"] if continuous else None,
                            "valid": bool(continuous and previous["valid_motion"] and valid_motion),
                            "stopped": state["stopped"]}
                # Same identity seen in consecutive frames: no tracking break.
                if self.record_tracking: interval["tracked"] = bool(continuous)
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
            if continuing:
                history = state.setdefault("history", deque(maxlen=HISTORY_SAMPLES))
                recent = (previous is not None and not previous.get("handoff") and
                          0 < timestamp-previous["time"] <= self.policy.max_gap_sec+1e-9)
                height = _height(observation)
                if not recent or height is None:history.clear()
                if height is not None:
                    history.append((timestamp, float(observation.point[0]), float(observation.point[1]), height))
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
                if timestamp-row["last_seen"] > self.policy.association_gap_sec:
                    row.update(completed=True, completed_at=timestamp);self.work+=1
                    self._flag(row, "tracking_lost")

    def _own_segments(self, identity, state):
        """Rows this raw ID itself started; an alias of another row started none."""
        row = state["row"]
        return state["segments"] if identity == (row.get("tracker_epoch"), row.get("local_id")) else 0

    def _contest(self, row_id, identity, timestamp):
        """A handed-off raw ID is seen again: the live link is contradicted.

        Only an unfinished row is flagged; a finished row may already be archived."""
        row = self.records.get(row_id)
        if row is None or row["completed"]:return
        self._flag(row, "identity_link_contested")
        for link in row.get("identity_links", []):
            if (link["from"]["epoch"], link["from"]["local_id"]) == identity:link.setdefault("contested_at", timestamp)

    def release(self):
        """Drop handoff references to evicted rows (the archive index answers for them)."""
        if self.handed_off:
            self.handed_off = {k: v for k, v in self.handed_off.items() if v["row_id"] in self.records}

    def physical(self, identity):
        """The vehicle row a raw (epoch, native ID) currently drives, if any."""
        state = self.active.get(identity)
        if state is None or state["row"].get("vehicle_id") not in self.records:return None
        row = state["row"]
        return {"vehicle_id": row["vehicle_id"], "local_id": row["local_id"],
                "tracker_epoch": row["tracker_epoch"], "native": {"epoch": identity[0], "local_id": identity[1]},
                "continued": bool(row.get("identity_links"))}

    def _fresh(self, identity):
        if identity in self.active or identity in self.handed_off or identity in self.outside_seen:return False
        return self.archived is None or self.archived(vehicle_key(self.run_id, *identity)) is None

    def _continue_fragments(self, observations, timestamp, unassigned=None):
        """Link a never-seen native ID to a just-lost row when nothing else competes.

        Work is O(newborns x lost) gate arithmetic (motion fitted once per lost
        row) plus O(observations) per surviving pair, and capped per frame."""
        newborns = []
        for o in observations:
            if o.local_id is None or not o.stable_id or not o.in_region:continue
            identity = (o.epoch, o.local_id)
            height = _height(o)
            if height is not None and self._fresh(identity):newborns.append((identity, o, height))
        if not newborns:return
        present = {(o.epoch, o.local_id) for o in observations if o.local_id is not None and o.stable_id}
        lost = []
        for identity, state in self.active.items():
            row, previous, history = state["row"], state["previous"], state.get("history")
            if identity in present or row["completed"] or previous is None or not history:continue
            if row["vehicle_id"] not in self.records or len(row.get("identity_links", ())) >= MAX_LINKS_PER_ROW:continue
            dt = timestamp-previous["time"]
            if not 0 < dt <= self.policy.max_gap_sec+1e-9:continue
            last_time, x, y, height = history[-1]
            # A fragment seen in fewer than MIN_FIT_SAMPLES frames has no position or
            # size consistency to continue (junction f475: a one-frame edge sliver).
            if last_time != previous["time"] or len(history) < MIN_FIT_SAMPLES:continue
            motion = _fit(history)
            fitted = motion is not None and math.hypot(*motion) >= MIN_MOVING_HEIGHTS_PER_SEC*height
            if fitted:
                gate = CONTINUATION_BASE_HEIGHTS+CONTINUATION_HEIGHTS_PER_SEC*dt
            elif dt <= STATIC_CONTINUATION_MAX_GAP_SEC+1e-9:
                motion = (0., 0.); gate = CONTINUATION_BASE_HEIGHTS
            else:continue
            lost.append((identity, state, dt, (x+motion[0]*dt, y+motion[1]*dt), height, gate, fitted))
        if not lost:return
        if unassigned is None:
            # Without the frame's withheld detections a competitor may be invisible: do not guess.
            self.continuation_unavailable_frames += 1
            return
        if len(newborns)*len(lost) > MAX_PAIR_CHECKS:
            self.continuation_skipped_frames += 1
            return
        # Validate every supplied observation/source/unassigned box before matching:
        # any unreadable competitor box makes the frame's evidence unavailable.
        competitors = _competitors(observations, unassigned)
        if competitors is None:
            self.continuation_unavailable_frames += 1
            return
        # The raw detection a newborn was emitted from (``source_bbox``) is the same
        # vehicle seen before smoothing: it widens ambiguity, never acceptance.
        source_points = []
        for _, o, _ in newborns:
            raw = getattr(o, "source_bbox", None)
            source = None if raw is None else _raw_box(raw)
            source_points.append(None if source is None else ((source[0]+source[2])/2, source[3]))
        # Degrees on position alone: class and size do not make a nearby box less of a competitor.
        new_degree = [0]*len(newborns); lost_degree = [0]*len(lost); edge = {}
        for i, (new, o, _) in enumerate(newborns):
            px, py = o.point
            source_point = source_points[i]
            for j, (old, _, _, predicted, height, gate, _) in enumerate(lost):
                if old[0] != new[0]:continue
                error = math.hypot(predicted[0]-px, predicted[1]-py)/height
                in_emitted = error <= gate
                in_source = source_point is not None and math.hypot(
                    predicted[0]-source_point[0], predicted[1]-source_point[1])/height <= gate
                if in_emitted or in_source:
                    new_degree[i] += 1; lost_degree[j] += 1
                    if in_emitted:edge[i] = (j, error)
        for i, (j, error) in edge.items():
            if new_degree[i] != 1 or lost_degree[j] != 1:continue
            new, o, new_height = newborns[i]
            old, state, dt, predicted, height, gate, fitted = lost[j]
            if state["row"]["class_name"] != o.class_name:continue
            if max(height, new_height)/min(height, new_height) > MAX_HEIGHT_RATIO:continue
            if self._competed(o, new_height, predicted, height, gate, competitors):continue
            self._link(old, new, state, o, timestamp, dt, error, gate, predicted, fitted)

    @staticmethod
    def _competed(newborn, new_height, predicted, height, gate, competitors):
        # The newborn's emitted box and its raw source box each compete with every
        # other box (and its raw source): a smoothed box can drift off a neighbour
        # its detection overlaps.
        newborn_boxes = []
        box = None if newborn.bbox is None else _raw_box(newborn.bbox)
        if box is not None:newborn_boxes.append((newborn.point, new_height, box))
        source = getattr(newborn, "source_bbox", None)
        source = None if source is None else _raw_box(source)
        if source is not None and source != box:
            newborn_boxes.append((((source[0]+source[2])/2, source[3]), source[3]-source[1], source))
        for other, point, other_height, bbox in competitors:
            if other is newborn:continue
            if math.dist(predicted, point)/height <= gate:return True
            if not newborn_boxes or bbox is None or other_height is None:continue
            for own_point, own_height, own_box in newborn_boxes:
                if (_overlap_of_smaller(own_box, bbox) >= COMPETITOR_OVERLAP and
                        math.dist(own_point, point)/min(own_height, other_height) <= COMPETITOR_FOOTPOINT_HEIGHTS):
                    return True
        return False

    def _link(self, old, new, state, observation, timestamp, dt, error, gate, predicted, fitted):
        row = state["row"]
        own = (row["tracker_epoch"], row["local_id"])
        del self.active[old]
        self.handed_off[old] = {"row_id": row["vehicle_id"], "segments": state["segments"] if old == own else 0}
        self.active[new] = state
        # The handoff is a heuristic: nothing (wait, crossing, motion) is carried across it.
        state["previous"] = dict(state["previous"], handoff=True)
        state["history"].clear()
        state["stopped"] = False; state["stop_candidate"] = state["move_candidate"] = None
        native = {"epoch": new[0], "local_id": new[1]}
        # Added on the first link only, so rows of unchanged runs keep their schema.
        row.setdefault("native_ids", [{"epoch": own[0], "local_id": own[1]}]).append(native)
        row.setdefault("identity_links", []).append({"from": {"epoch": old[0], "local_id": old[1]}, "to": native,
            "lost_at": state["previous"]["time"], "linked_at": timestamp, "gap_sec": dt,
            "predicted_point": list(predicted), "observed_point": list(observation.point),
            "error_heights": error, "gate_heights": gate, "moving_fit": fitted,
            "rule": "mutually_unique_uncontested_motion_gate"})
        self._flag(row, "identity_continued")

    def invalidate(self, timestamp, reason):
        for state in self.active.values():
            row = state["row"]
            if not row["completed"]:
                self._flag(row, reason)
                state["previous"] = None
                state["stop_candidate"] = state["move_candidate"] = None
                if "history" in state:state["history"].clear()
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
