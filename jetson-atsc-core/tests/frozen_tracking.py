"""Frozen post-NMS detections through the installed Ultralytics tracker callbacks.

``FrozenTrackModel`` stands in for ``ultralytics.YOLO`` only where the neural
network would run. ``track`` is the installed ``Model.track`` itself, so tracker
registration follows the installed version (8.3.x registers only when the
predictor has no trackers and appends; 8.4.x replaces in place). ``predict``
then runs predictor events in the order ``BasePredictor.stream_inference``
does (on_predict_start, on_predict_batch_start, postprocess,
on_predict_postprocess_end) on images prepared by the installed loader, with
the installed tracker. Production ``DetectorTracker.track`` parses the results
unchanged.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import ultralytics
from ultralytics.data.loaders import LoadPilAndNumpy
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils import callbacks as ultralytics_callbacks

from src.perception.detector_tracker import DetectorTracker

NAMES = {0: "motorcycle", 1: "car", 2: "bus", 3: "truck"}
ULTRALYTICS_VERSION = tuple(int(part) for part in ultralytics.__version__.split(".")[:3])
# 8.4.x hides a frame whose only tracks await confirmation; 8.3.x passes its raw detections through untracked.
HIDES_UNCONFIRMED = ULTRALYTICS_VERSION >= (8, 4, 0)


class FrozenTrackModel:
    names = NAMES
    track = Model.track

    def __init__(self, frames, *, fail_on=(), batched=False):
        # frames: callable(call_index, image) -> (N, 6) [x1, y1, x2, y2, conf, cls], or a list of those.
        # batched: each list entry is one call's list of per-image detections.
        self.frames = frames
        self.batched = batched
        self.callbacks = ultralytics_callbacks.get_default_callbacks()
        self.predictor = None
        self.calls = 0
        self.fail_on = set(fail_on)
        self.predict_calls = 0
        self.last_boxes = []

    def add_callback(self, event, func):
        self.callbacks[event].append(func)

    def _boxes(self, images):
        if callable(self.frames):
            sources = [self.frames(self.calls, image) for image in images]
        elif self.batched:
            sources = self.frames[self.calls]
        else:
            sources = [self.frames[self.calls]]
        assert len(sources) == len(images)
        return [np.asarray(source, dtype=np.float32).reshape(-1, 6) for source in sources]

    def predict(self, source=None, stream=False, **kwargs):
        frames = source if isinstance(source, list) else [source]
        if kwargs.get("mode") != "track":
            # Ultralytics' untracked path (DetectorTracker's fallback): no tracker callbacks run.
            self.predict_calls += 1
            if self.predictor is not None:
                self.predictor.args.mode = "predict"
                for callback in self.callbacks["on_predict_batch_start"]:callback(self.predictor)
            # Same frame as the failed track() call: its detections, without IDs.
            return [Results(image, path="frozen", names=self.names, boxes=torch.from_numpy(boxes))
                    for image, boxes in zip(frames, self.last_boxes)]
        images = [LoadPilAndNumpy._single_check(image) for image in frames]
        paths = [f"image{i}.jpg" for i in range(len(images))]
        if self.predictor is None:
            self.predictor = SimpleNamespace(
                args=SimpleNamespace(mode="track", task="detect", tracker=kwargs.get("tracker", "bytetrack.yaml")),
                dataset=SimpleNamespace(bs=len(images), mode="image"), device=torch.device("cpu"),
                callbacks=self.callbacks, model=None, save_dir=Path("frozen"))
        predictor = self.predictor
        predictor.args.mode = "track"
        index = self.calls
        boxes = self.last_boxes = self._boxes(images)
        self.calls += 1
        for callback in self.callbacks["on_predict_start"]:callback(predictor)
        predictor.batch = (paths, images, [""]*len(images))
        for callback in self.callbacks["on_predict_batch_start"]:callback(predictor)
        predictor.results = [Results(image, path=path, names=self.names, boxes=torch.from_numpy(found))
                             for image, path, found in zip(images, paths, boxes)]
        if index in self.fail_on:
            for tracker_instance in predictor.trackers:tracker_instance._fail_next = True
        for callback in self.callbacks["on_predict_postprocess_end"]:callback(predictor)
        return predictor.results


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


def frozen_detector(model, profile="baseline", conf=.25, *, evidence=True):
    detector = DetectorTracker.__new__(DetectorTracker)
    detector.model_path = "frozen-post-nms"; detector.conf_threshold = conf; detector.iou_threshold = .5
    detector.imgsz = 640; detector.device = "cpu"; detector.max_gap_sec = 5.; detector.is_tensorrt = True
    detector.track_trajectories = {}; detector.track_obs_counts = {}; detector.max_history_len = 10
    detector.min_stopped_observations = 5; detector.last_seen_timestamp = 0.; detector.epoch = 0
    detector.set_tracking_profile(profile)
    detector.model = model
    if evidence:detector.enable_withheld_evidence()
    return detector
