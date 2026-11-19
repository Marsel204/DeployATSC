"""
Shared-Engine YOLO11 Batch Inference & Multi-Object Tracking Engine.

Supports:
- Batched TensorRT FP16 engine (yolo11s_fp16.engine) on Jetson Orin Nano
- Native PyTorch CUDA/CPU fallback (Final.pt)
- Concurrent dual-camera batch tracking (frame_sys + frame_acq) in a single forward pass
- ByteTrack integration with track trajectory and ground velocity estimation
"""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.perception.tracking_profiles import get_tracking_profile


# Standard PKJI vehicle categories
CLASS_NAMES = {
    0: "motorcycle",
    1: "car",
    2: "bus",
    3: "truck",
}

# Canonical normalization map for standard vehicle categories (COCO and custom)
CANONICAL_VEHICLE_MAP = {
    "motorcycle": "motorcycle",
    "motorbike": "motorcycle",
    "bike": "motorcycle",
    "car": "car",
    "automobile": "car",
    "bus": "bus",
    "truck": "truck",
}


@lru_cache(maxsize=1)
def _still_predictor_type():
    """Camera 1's dedicated predictor must forward the real image only once.

    Installed Ultralytics warms CUDA by forwarding the first supplied image,
    then forwards it again for inference. Warmup is an optional optimization:
    skip it for still captures, leaving the continuous Camera 2 predictor alone.
    Cache the class so YOLO reuses its backend on subsequent clicks.
    """
    from ultralytics.models.yolo.detect.predict import DetectionPredictor

    class StillImagePredictor(DetectionPredictor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.done_warmup = True

    return StillImagePredictor


class TrackedVehicle:
    """Represents a single tracked vehicle in rectified pixel and metric space."""

    def __init__(
        self,
        track_id: int,
        class_id: int,
        class_name: str,
        bbox: List[float],
        confidence: float,
        timestamp: float,
        obs_count: int = 5,
        stable_id: bool = True,
        epoch: int = 0,
    ) -> None:
        self.track_id = track_id
        self.class_id = class_id
        self.class_name = class_name
        self.bbox = [float(b) for b in bbox]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.timestamp = timestamp
        self.obs_count = obs_count
        self.stable_id = stable_id
        self.epoch = epoch
        self.velocity_valid = False

        # Contact point: center of bottom edge
        self.bottom_center = (
            float((self.bbox[0] + self.bbox[2]) / 2.0),
            float(self.bbox[3]),
        )

        # Ground plane coordinates [X_m, Y_m] (set by homography engine)
        self.ground_pos: Optional[Tuple[float, float]] = None
        self.velocity_mps: float = 0.0

    @property
    def is_stopped(self) -> bool:
        """Determines if vehicle is queued / stationary (speed <= 1.0 m/s and persistent >= 5 frames)."""
        return getattr(self, "obs_count", 5) >= 5 and self.velocity_mps <= 1.0


class TrackedFrame(list):
    """One frame's tracked vehicles (a plain list to every caller), plus the
    frame's actual post-NMS detections the native tracker did not emit.

    ``withheld`` holds (bbox, class_name, confidence) for vehicle-class boxes
    ByteTrack discarded or kept unconfirmed in this very frame, identified by
    the tracker's own detection index (never by box matching, which tracker
    smoothing would make a guess). ``None`` means that evidence is unavailable
    (predict fallback, injected model, incomplete capture): callers must then
    not infer anything that needs it. Evidence only; never a vehicle.
    """
    __slots__ = ("withheld", "epoch")

    def __init__(self, vehicles=(), withheld=None, epoch=None):
        super().__init__(vehicles)
        self.withheld = withheld
        self.epoch = epoch


class _TrackerProvenance:
    """Record each native tracker update's image, input detections and output indices.

    Installed as an ``on_predict_batch_start`` callback, which runs after the
    installed Ultralytics ``on_predict_start`` created the trackers and before
    its ``on_predict_postprocess_end`` calls ``tracker.update``; so the order
    relative to the tracker callbacks is fixed by the predictor, not by
    registration order. Records only while armed by DetectorTracker.track.

    Each call is ``(image, evidence)``: ``image`` is the array the tracker was
    given (the result's ``orig_img``), so a call is attributed to its own image
    even when the installed Ultralytics skips the update for an image with no
    detections (8.3.x); ``evidence`` is None when it could not be read.
    """

    def __init__(self) -> None:
        self.armed = False
        self.calls: List[Tuple[Any, Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]] = []

    def __call__(self, predictor) -> None:
        if getattr(getattr(predictor, "args", None), "mode", None) != "track":
            return
        for tracker in getattr(predictor, "trackers", ()) or ():
            update = getattr(tracker, "update", None)
            if update is None or getattr(update, "_atsc_provenance", None) is self:
                continue
            tracker.update = self._wrap(update)

    def _wrap(self, update):
        def recorded(results, *args, **kwargs):
            tracks = update(results, *args, **kwargs)
            if self.armed:
                image = args[0] if args else kwargs.get("img")
                try:
                    xyxy = np.array(results.xyxy, dtype=np.float64).reshape(-1, 4)
                    cls = np.array(results.cls, dtype=np.float64).reshape(-1)
                    conf = np.array(results.conf, dtype=np.float64).reshape(-1)
                    out = np.asarray(tracks)
                    idx = out[:, -1].astype(int).copy() if out.ndim == 2 and len(out) else np.zeros(0, dtype=int)
                    if not (len(xyxy) == len(cls) == len(conf)):raise ValueError("ragged detections")
                    self.calls.append((image, (xyxy, cls, conf, idx)))
                except Exception:
                    # Unreadable evidence: poison this frame rather than guess.
                    self.calls.append((image, None))
            return tracks
        recorded._atsc_provenance = self
        return recorded


def _drop_tracker_callbacks(model) -> None:
    """Remove the installed Ultralytics tracker callbacks before the predictor is dropped.

    Ultralytics 8.3.x ``Model.track`` registers them whenever the predictor has
    no trackers and appends rather than replaces, so after a reset each frame
    would run ``tracker.update`` once per stacked registration. The next
    ``track`` call registers exactly one fresh pair (on every version)."""
    callbacks = getattr(model, "callbacks", None)
    if not isinstance(callbacks, dict):
        return
    try:
        from ultralytics.trackers import track as installed
    except Exception:
        return
    native = {getattr(installed, "on_predict_start", None), getattr(installed, "on_predict_postprocess_end", None)} - {None}
    for event in ("on_predict_start", "on_predict_postprocess_end"):
        registered = callbacks.get(event)
        if isinstance(registered, list):
            registered[:] = [cb for cb in registered if getattr(cb, "func", cb) not in native]


class DetectorTracker:
    """
    Batched vehicle detection and ByteTrack tracking system.
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.30,
        iou_threshold: float = 0.50,
        imgsz: int = 640,
        device: str = "cuda:0",
        max_gap_sec: float = 5.0,
        tracking_profile: str = "baseline",
    ) -> None:
        """
        Args:
            model_path: Path to TensorRT engine or PyTorch .pt weights.
            conf_threshold: Minimum detection confidence threshold.
            iou_threshold: NMS IoU threshold.
            imgsz: Input resolution size.
            device: 'cuda:0' or 'cpu'.
            tracking_profile: Named tracking floor; does not change still/fallback detection.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.device = device
        self.max_gap_sec = max_gap_sec
        self.set_tracking_profile(tracking_profile)

        self.is_tensorrt: bool = str(model_path).endswith(".engine")
        self.model = None

        # Track history for velocity calculation: {track_id: list of (timestamp, ground_x, ground_y)}
        self.track_trajectories: Dict[int, List[Tuple[float, float, float]]] = {}
        self.track_obs_counts: Dict[int, int] = {}
        self.max_history_len = 10
        self.min_stopped_observations = 5
        self.last_seen_timestamp: float = 0.0
        self.epoch = 0

        self._load_model()

    def enable_withheld_evidence(self) -> bool:
        """Opt in to capturing each frame's tracker-withheld detections.

        Off by default: only a run that uses the evidence (opt-in native
        fragment continuation) pays the per-update capture. Without it every
        TrackedFrame reports ``withheld=None`` (unavailable). Returns whether
        capture is installed."""
        self._attach_provenance()
        return getattr(self, "_provenance", None) is not None

    def _attach_provenance(self) -> None:
        """Register the frame-local detection provenance hook once per model."""
        if getattr(self, "_provenance", None) is not None:
            return
        add = getattr(self.model, "add_callback", None)
        if add is None:
            return
        self._provenance = _TrackerProvenance()
        add("on_predict_batch_start", self._provenance)

    def _frame_evidence(self, results, calls) -> List[Optional[tuple]]:
        """Per-image withheld detections from the recorded tracker updates.

        Calls are attributed by image identity, never by position: Ultralytics
        8.3.x skips ``tracker.update`` for an image without detections, so a
        mixed batch has fewer calls than images. An image with no call has
        evidence only if it has no detections at all; an image with several
        calls (stacked tracker callbacks) or an unattributable call has none."""
        images = [getattr(result, "orig_img", None) for result in results]
        per_image: List[List[Any]] = [[] for _ in results]
        for image, evidence in calls:
            matches = [j for j, candidate in enumerate(images) if image is not None and candidate is image]
            if len(matches) != 1:
                return [None]*len(results)
            per_image[matches[0]].append(evidence)
        frames: List[Optional[tuple]] = []
        for result, found in zip(results, per_image):
            if len(found) == 1:
                frames.append(self._withheld(result, found[0]))
            elif not found:
                boxes = getattr(result, "boxes", None)
                frames.append(() if boxes is not None and len(boxes) == 0 else None)
            else:
                frames.append(None)
        return frames

    def _withheld(self, result, call) -> Optional[tuple]:
        """Post-NMS vehicle boxes of this result the tracker did not emit, or None."""
        if call is None:
            return None
        xyxy, cls, conf, idx = call
        boxes = getattr(result, "boxes", None)
        emitted_count = 0 if boxes is None else len(boxes)
        if boxes is not None and boxes.id is not None:
            if emitted_count != len(idx) or (len(idx) and (idx.min() < 0 or idx.max() >= len(xyxy))):
                return None
            emitted = set(idx.tolist())
        elif emitted_count == 0:
            emitted = set()         # all hidden (e.g. new tracks awaiting confirmation)
        elif emitted_count == len(xyxy) and len(idx) == 0:
            return ()               # untracked passthrough: every detection is an observation
        else:
            return None
        withheld = []
        for j in range(len(xyxy)):
            if j in emitted:
                continue
            name = self.map_class(int(cls[j]))
            box = tuple(float(v) for v in xyxy[j])
            if name is None or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            withheld.append((box, name, float(conf[j])))
        return tuple(withheld)

    def set_tracking_profile(self, name: str) -> None:
        """Select before tracking starts; reset_tracking preserves this choice."""
        self.tracking_profile = get_tracking_profile(name)

    def tracking_profile_provenance(self) -> dict:
        return self.tracking_profile.provenance(self.conf_threshold, self.iou_threshold)

    def reset_tracking(self) -> None:
        """Start a new identity epoch without reloading the model weights."""
        self.epoch += 1
        self.track_trajectories.clear()
        self.track_obs_counts.clear()
        self.last_seen_timestamp = 0.0
        if self.model is not None and hasattr(self.model, "predictor"):
            _drop_tracker_callbacks(self.model)
            self.model.predictor = None

    def _load_model(self) -> None:
        """Loads Ultralytics YOLO model or TensorRT engine."""
        from ultralytics import YOLO

        if not os.path.exists(self.model_path):
            # If specified engine does not exist, check for Final.pt fallback
            alt_pt = os.path.join(os.path.dirname(self.model_path), "Final.pt")
            if os.path.exists(alt_pt):
                print(f"[Detector] Engine '{self.model_path}' not found. Falling back to: {alt_pt}")
                self.model_path = alt_pt
                self.is_tensorrt = False
            else:
                raise FileNotFoundError(f"Neither model nor fallback found at: {self.model_path}")

        print(f"[Detector] Initializing YOLO from: {self.model_path} (TensorRT: {self.is_tensorrt})")
        self.model = YOLO(self.model_path, task="detect")
        import torch
        self.inference_device=self.device
        if not self.is_tensorrt and self.device!="cpu" and not torch.cuda.is_available():
            self.inference_device="cpu"

    def map_class(self, class_id: int) -> Optional[str]:
        """
        Maps model output class_id to canonical PKJI vehicle category:
        'motorcycle', 'car', 'bus', 'truck'.
        Returns None if class is not a recognized road vehicle (e.g. person, traffic light).
        """
        if self.model is not None and hasattr(self.model, "names") and self.model.names:
            raw_name = str(self.model.names.get(class_id, "")).lower().strip()
            if raw_name in CANONICAL_VEHICLE_MAP:
                return CANONICAL_VEHICLE_MAP[raw_name]
            # If model only has 4 classes without full names, fallback to CLASS_NAMES
            if len(self.model.names) == 4 and class_id in CLASS_NAMES:
                return CLASS_NAMES[class_id]
            return None
        return CLASS_NAMES.get(class_id, None)

    def is_valid_vehicle_class(self, class_id: int) -> bool:
        """Returns True if class_id corresponds to a tracked road vehicle."""
        return self.map_class(class_id) is not None

    def track(
        self,
        frames: List[np.ndarray],
        timestamp: Optional[float] = None,
    ) -> List[TrackedFrame]:
        """
        Performs batched inference and multi-object tracking across 1 or 2 frames.

        Args:
            frames: List of 1 or 2 BGR images (e.g. [frame_sys] or [frame_sys, frame_acq]).
            timestamp: Ingestion timestamp.

        Returns:
            List of tracked vehicle lists, one list per input frame.
        """
        if timestamp is None:
            timestamp = time.time()

        if not frames:
            return []
        if self.is_tensorrt and len(frames)!=1:
            raise ValueError("This TensorRT engine accepts exactly one frame per call")

        # Opt-in (enable_withheld_evidence): without it no capture runs and evidence is unavailable.
        provenance = getattr(self, "_provenance", None)
        tracked = False
        if provenance is not None:
            provenance.calls = []
            provenance.armed = True
        # Batched inference via Ultralytics track
        try:
            results = self.model.track(
                source=frames,
                conf=self.tracking_profile.track_confidence(self.conf_threshold),
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=getattr(self,"inference_device",self.device),
                persist=True,
                tracker=self.tracking_profile.tracker_argument,
                verbose=False,
            )
            tracked = True
        except Exception as e:
            # Fallback if tracker config fails or tracking fails
            print(f"[Detector Tracker] Warning in track(): {e}. Running predict without tracker...")
            results = self.model.predict(
                source=frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=getattr(self,"inference_device",self.device),
                verbose=False,
            )
        finally:
            if provenance is not None:
                provenance.armed = False
        calls = provenance.calls if provenance is not None else []
        if provenance is not None:
            provenance.calls = []
        results = list(results)
        if tracked and provenance is not None:
            evidence = self._frame_evidence(results, calls)
        else:
            evidence = [None]*len(results)

        batch_outputs: List[TrackedFrame] = []

        for index, res in enumerate(results):
            frame_vehicles = TrackedFrame(withheld=evidence[index], epoch=self.epoch)
            boxes = getattr(res, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                xyxy_arr = boxes.xyxy.cpu().numpy()
                conf_arr = boxes.conf.cpu().numpy()
                cls_arr = boxes.cls.cpu().numpy().astype(int)

                # Track IDs if tracker succeeded, otherwise fallback index
                if boxes.id is not None:
                    id_arr = boxes.id.cpu().numpy().astype(int)
                else:
                    id_arr = np.arange(len(boxes), dtype=int)

                for i in range(len(boxes)):
                    c_id = int(cls_arr[i])
                    c_name = self.map_class(c_id)
                    if c_name is None:
                        continue  # Skip non-vehicles (e.g. traffic light #114 class_9, pedestrians)

                    tid = int(id_arr[i])
                    current_obs = self.track_obs_counts.get(tid, 0) if boxes.id is not None else 0
                    v = TrackedVehicle(
                        track_id=tid,
                        class_id=c_id,
                        class_name=c_name,
                        bbox=xyxy_arr[i].tolist(),
                        confidence=float(conf_arr[i]),
                        timestamp=timestamp,
                        obs_count=current_obs,
                        stable_id=boxes.id is not None,
                        epoch=self.epoch,
                    )
                    frame_vehicles.append(v)

            batch_outputs.append(frame_vehicles)

        return batch_outputs

    def detect_once(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[TrackedVehicle]:
        """One native still-image forward on exactly this frame; no tracker, history or velocity.

        Use a dedicated instance: a predictor previously used by track() carries tracker callbacks.
        """
        if timestamp is None:
            timestamp = time.time()
        results = self.model.predict(
            source=frame,
            predictor=_still_predictor_type(),
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=getattr(self,"inference_device",getattr(self,"device","cuda:0")),
            verbose=False,
        )
        vehicles: List[TrackedVehicle] = []
        for res in results[:1]:
            boxes = getattr(res, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue
            xyxy_arr = boxes.xyxy.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(boxes)):
                c_name = self.map_class(int(cls_arr[i]))
                if c_name is None:
                    continue
                vehicles.append(TrackedVehicle(
                    track_id=i, class_id=int(cls_arr[i]), class_name=c_name,
                    bbox=xyxy_arr[i].tolist(), confidence=float(conf_arr[i]),
                    timestamp=timestamp, obs_count=0, stable_id=False, epoch=self.epoch,
                ))
        return vehicles

    def update_velocities(
        self,
        vehicles: List[TrackedVehicle],
        homography_func,
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Projects vehicles to the ground plane and updates their estimated ground velocity.

        Args:
            vehicles: List of TrackedVehicle objects from the approach camera.
            homography_func: Callable (u, v) -> (X_m, Y_m).
            timestamp: Optional reference timestamp (video replay time or epoch).
        """
        if timestamp is not None:
            ref_time = timestamp
        elif vehicles:
            ref_time = max(v.timestamp for v in vehicles)
        elif self.last_seen_timestamp > 0.0:
            ref_time = self.last_seen_timestamp
        else:
            ref_time = time.time()
        self.last_seen_timestamp = max(self.last_seen_timestamp, ref_time)

        active_ids = set()

        for v in vehicles:
            gx, gy = homography_func(v.bottom_center[0], v.bottom_center[1])
            v.ground_pos = (gx, gy)
            v.velocity_valid = False
            v.velocity_mps = 0.0
            v.obs_count = 0
            # Each instance is camera-local; reset clears its epoch's namespace.
            # Detector-only array indices must never read or write stable history.
            if not v.stable_id or v.epoch != getattr(self,"epoch",0) or not np.isfinite([v.timestamp,gx,gy]).all():
                continue
            active_ids.add(v.track_id)
            traj = self.track_trajectories.setdefault(v.track_id, [])
            if traj and v.timestamp <= traj[-1][0]:
                continue
            # Check on return as well as during pruning: no update may have run
            # during the gap. All continuity decisions use source timestamps.
            if traj and v.timestamp-traj[-1][0] > getattr(self,"max_gap_sec",5.0):
                traj.clear()
                self.track_obs_counts.pop(v.track_id, None)
            traj.append((v.timestamp, gx, gy))

            if len(traj) > self.max_history_len:
                traj.pop(0)

            # Accumulate temporal persistence observations
            self.track_obs_counts[v.track_id] = self.track_obs_counts.get(v.track_id, 0) + 1
            v.obs_count = self.track_obs_counts[v.track_id]

            # Calculate speed over the last 3-5 frames (>= 0.1s delta)
            if len(traj) >= 3:
                t_old, x_old, y_old = traj[0]
                dt = v.timestamp - t_old
                if dt > 0.05:
                    dist_m = float(np.hypot(gx - x_old, gy - y_old))
                    speed = dist_m / dt
                    # Clamp unrealistic tracking jump noise
                    v.velocity_mps = min(35.0, max(0.0, speed))
                    v.velocity_valid = bool(v.stable_id and np.isfinite(speed))
                else:
                    v.velocity_mps = 0.0
            else:
                v.velocity_mps = 0.0

        # Prune using the same source-time continuity policy as returning tracks.
        for tid in list(self.track_trajectories.keys()):
            if tid not in active_ids:
                if self.track_trajectories[tid] and (ref_time - self.track_trajectories[tid][-1][0] > getattr(self,"max_gap_sec",5.0)):
                    del self.track_trajectories[tid]
                    self.track_obs_counts.pop(tid, None)
