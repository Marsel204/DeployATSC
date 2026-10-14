"""Evidence-based multi-color observation; never a cabinet control driver."""
from __future__ import annotations

import cv2
import numpy as np
from src.experiment.contracts import Phase, RunPolicy


class SignalPhaseObserver:
    def __init__(self, roi, policy: RunPolicy, lamp_rois=None):
        self.roi = roi
        self.policy = policy
        self.lamp_rois = lamp_rois or {}
        self.phase = Phase.UNKNOWN
        self.events = []
        self.cycles = []
        self.last_sequence = None
        self.last_time = None
        self.candidate = Phase.UNKNOWN
        self.candidate_time = None
        self.count = 0
        self.scores = {}
        self.green = None
        self.cycle_id = 0
        self.measurement_phase = Phase.UNKNOWN

    def _scores(self, frame):
        scores = {}
        for phase in (Phase.RED, Phase.YELLOW, Phase.GREEN):
            roi = self.lamp_rois.get(phase, self.roi)
            crop = frame[roi["ymin"]:roi["ymax"], roi["xmin"]:roi["xmax"]]
            if not crop.size: scores[phase] = 0.; continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            bright = (s >= 120) & (v >= 150)
            color = ((h <= 10) | (h >= 170)) if phase == Phase.RED else (
                    (h >= 18) & (h <= 35) if phase == Phase.YELLOW else (h >= 36) & (h <= 95))
            scores[phase] = float(np.mean(bright & color))
        return scores

    def _transition(self, phase, onset, confirmed, reason="observed"):
        previous = self.phase
        if previous == phase: return None
        event = {"from": str(previous), "phase": str(phase), "timestamp": onset,
                 "confirmed_at": confirmed, "reason": reason, "scores": dict(self.scores)}
        self.events.append(event)
        if self.green is not None and phase != Phase.GREEN:
            complete = phase == Phase.YELLOW and self.green["start_complete"]
            self.green.update({"green_end": onset, "end_phase": str(phase), "end_confirmed_at": confirmed,
                "yellow_confirmed_at": confirmed if phase == Phase.YELLOW else None,
                "actual_green_sec": round(onset-self.green["green_start"], 6) if complete else None,
                "complete": complete, "quality": "complete" if complete else reason if reason != "observed" else "incomplete_boundary"})
            self.cycles.append(self.green)
            self.green = None
        if phase == Phase.GREEN:
            self.cycle_id += 1
            self.green = {"cycle_id": self.cycle_id, "green_start": onset,
                          "green_confirmed_at": confirmed, "start_complete": previous == Phase.RED,
                          "complete": False, "actual_green_sec": None}
        self.phase = phase
        return event

    def invalidate(self, timestamp, reason):
        self.candidate = Phase.UNKNOWN; self.count = 0
        self.measurement_phase = Phase.UNKNOWN
        return self._transition(Phase.UNKNOWN, timestamp, timestamp, reason)

    def observe(self, frame, timestamp, sequence):
        if sequence == self.last_sequence or (self.last_time is not None and timestamp <= self.last_time):
            return {"phase": str(self.phase), "measurement_phase": str(self.measurement_phase), "transition": None, "duplicate": True}
        if self.last_time is not None and timestamp-self.last_time > self.policy.max_gap_sec:
            self.invalidate(self.last_time, "observation_gap")
        self.last_time = timestamp; self.last_sequence = sequence
        self.scores = self._scores(frame)
        active = [key for key, value in self.scores.items() if value >= self.policy.phase_ratio]
        raw = active[0] if len(active) == 1 else Phase.UNKNOWN
        if raw != self.candidate:
            self.candidate = raw; self.candidate_time = timestamp; self.count = 1
        else: self.count += 1
        transition = None
        if raw == Phase.UNKNOWN:
            transition = self.invalidate(timestamp, "ambiguous_or_dark")
        elif self.count >= self.policy.debounce_frames:
            transition = self._transition(raw, self.candidate_time, timestamp)
        self.measurement_phase = raw if raw == self.phase else Phase.UNKNOWN
        return {"phase": str(self.phase), "measurement_phase": str(self.measurement_phase),
                "transition": transition, "scores": dict(self.scores), "duplicate": False}

    def status(self):
        return {"phase": str(self.phase), "measurement_phase": str(self.measurement_phase),
                "scores": dict(self.scores), "candidate": str(self.candidate),
                "confirmation_frames": self.count, "cycle_id": self.cycle_id,
                "green": dict(self.green) if self.green else None}
