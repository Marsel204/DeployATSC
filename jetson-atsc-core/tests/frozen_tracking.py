"""Frozen post-NMS detections through the installed Ultralytics tracker callbacks.

``FrozenTrackModel`` stands in for ``ultralytics.YOLO`` only where the neural
network would run: it registers the installed ``register_tracker`` callbacks
and runs predictor events in the order ``BasePredictor.stream_inference``
does (on_predict_start, on_predict_batch_start, postprocess,
on_predict_postprocess_end), with a fresh installed BYTETracker. Production
``DetectorTracker.track`` then parses the results unchanged.
"""
from types import SimpleNamespace

import numpy as np
import torch
from ultralytics.engine.results import Results
from ultralytics.trackers.track import register_tracker
from ultralytics.utils import callbacks as ultralytics_callbacks

from src.perception.detector_tracker import DetectorTracker

NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}


class FrozenTrackModel:
    names = NAMES

    def __init__(self, frames, *, fail_on=()):
        # frames: callable(call_index, image) -> (N, 6) [x1, y1, x2, y2, conf, cls], or a list of those.
        self.frames = frames
        self.callbacks = ultralytics_callbacks.get_default_callbacks()
        self.predictor = None
        self.calls = 0
        self.fail_on = set(fail_on)
        self.predict_calls = 0
        self.last_boxes = np.zeros((0, 6), dtype=np.float32)

    def add_callback(self, event, func):
        self.callbacks[event].append(func)

    def _boxes(self, image):
        source = self.frames(self.calls, image) if callable(self.frames) else self.frames[self.calls]
        return np.asarray(source, dtype=np.float32).reshape(-1, 6)

    def track(self, source, conf, persist=False, tracker="bytetrack.yaml", **kwargs):
        assert len(source) == 1
        register_tracker(self, persist)
        if self.predictor is None:
            self.predictor = SimpleNamespace(
                args=SimpleNamespace(mode="track", task="detect", tracker=tracker),
                dataset=SimpleNamespace(bs=1, mode="image"), device=torch.device("cpu"),
                callbacks=self.callbacks, model=None)
        predictor = self.predictor
        predictor.args.mode = "track"
        index = self.calls
        boxes = self.last_boxes = self._boxes(source[0])
        self.calls += 1
        for event in ("on_predict_start", "on_predict_batch_start"):
            for callback in self.callbacks[event]:callback(predictor)
        predictor.results = [Results(source[0], path="frozen", names=self.names, boxes=torch.from_numpy(boxes))]
        if index in self.fail_on:
            for tracker_instance in predictor.trackers:tracker_instance._fail_next = True
        for callback in self.callbacks["on_predict_postprocess_end"]:callback(predictor)
        return predictor.results

    def predict(self, source, **kwargs):
        """Ultralytics' untracked path (DetectorTracker's fallback): no tracker callbacks run."""
        self.predict_calls += 1
        frames = source if isinstance(source, list) else [source]
        if self.predictor is not None:
            self.predictor.args.mode = "predict"
            for callback in self.callbacks["on_predict_batch_start"]:callback(self.predictor)
        # Same frame as the failed track() call: its detections, without IDs.
        return [Results(image, path="frozen", names=self.names,
                        boxes=torch.from_numpy(self.last_boxes)) for image in frames]


def failing_trackers(model):
    """Make the installed tracker raise once when its frame is listed in ``fail_on``."""
    def wrap(predictor):
        for tracker in getattr(predictor, "trackers", ()):
            if getattr(tracker, "_failing_wrapped", False):continue
            update = type(tracker).update

            def raising(self_, *args, _update=update, **kwargs):
                if getattr(self_, "_fail_next", False):
                    self_._fail_next = False
                    raise RuntimeError("tracker failure")
                return _update(self_, *args, **kwargs)
            tracker.__class__ = type("Failing" + type(tracker).__name__, (type(tracker),), {"update": raising})
            tracker._failing_wrapped = True
    # Runs before DetectorTracker's provenance hook (registered later), so the hook wraps the failing update.
    model.callbacks["on_predict_batch_start"].insert(0, wrap)


def frozen_detector(model, profile="baseline", conf=.25):
    detector = DetectorTracker.__new__(DetectorTracker)
    detector.model_path = "frozen-post-nms"; detector.conf_threshold = conf; detector.iou_threshold = .5
    detector.imgsz = 640; detector.device = "cpu"; detector.max_gap_sec = 5.; detector.is_tensorrt = True
    detector.track_trajectories = {}; detector.track_obs_counts = {}; detector.max_history_len = 10
    detector.min_stopped_observations = 5; detector.last_seen_timestamp = 0.; detector.epoch = 0
    detector.set_tracking_profile(profile)
    detector.model = model
    detector._attach_provenance()
    return detector
