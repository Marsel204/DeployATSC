"""Bounded, display-only acquisition performance statistics."""
from collections import deque


class AcquisitionRates:
    def __init__(self):
        self.samples = deque(maxlen=512)
        self.inference = deque(maxlen=60)
        self.cold_start_ms = None

    def reset_playback(self):
        self.samples.clear()

    def record(self, now, source_time, inference_ms):
        self.samples.append((now, source_time))
        if self.cold_start_ms is None:
            self.cold_start_ms = inference_ms
        else:
            self.inference.append(inference_ms)

    def snapshot(self, now, playing):
        recent = [sample for sample in self.samples if now - sample[0] <= 5.]
        elapsed = now - recent[0][0] if len(recent) > 1 else 0.
        valid = playing and elapsed > 0
        return {
            "processed_fps": round((len(recent)-1)/elapsed, 2) if valid else 0.,
            "realtime_factor": round((recent[-1][1]-recent[0][1])/elapsed, 2) if valid else 0.,
            "steady_inference_ms": round(sum(self.inference)/len(self.inference), 2) if self.inference else None,
            "cold_start_ms": self.cold_start_ms,
            "fps_scope": "recent_playback_5s",
        }
