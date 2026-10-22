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
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


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
        device: str = "auto",
    ) -> None:
        """
        Args:
            model_path: Path to TensorRT engine or PyTorch .pt weights.
            conf_threshold: Minimum detection confidence threshold.
            iou_threshold: NMS IoU threshold.
            imgsz: Input resolution size.
            device: 'auto' selects CUDA when available; 'cuda:0' or 'cpu' selects explicitly.
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        if device == "auto":
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

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

    def reset_tracking(self) -> None:
        """Start a new identity epoch without reloading the model weights."""
        self.epoch += 1
        self.track_trajectories.clear()
        self.track_obs_counts.clear()
        self.last_seen_timestamp = 0.0
        if self.model is not None and hasattr(self.model, "predictor"):
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
    ) -> List[List[TrackedVehicle]]:
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

        # Batched inference via Ultralytics track
        try:
            results = self.model.track(
                source=frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except Exception as e:
            # Fallback if tracker config fails or tracking fails
            print(f"[Detector Tracker] Warning in track(): {e}. Running predict without tracker...")
            results = self.model.predict(
                source=frames,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )

        batch_outputs: List[List[TrackedVehicle]] = []

        for res in results:
            frame_vehicles: List[TrackedVehicle] = []
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
                    current_obs = self.track_obs_counts.get(tid, 0)
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
        self.last_seen_timestamp = ref_time

        active_ids = set()

        for v in vehicles:
            active_ids.add(v.track_id)
            gx, gy = homography_func(v.bottom_center[0], v.bottom_center[1])
            v.ground_pos = (gx, gy)

            if v.track_id not in self.track_trajectories:
                self.track_trajectories[v.track_id] = []

            traj = self.track_trajectories[v.track_id]
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

        # Prune stale trajectories older than 5.0 seconds using replay/wall reference time
        for tid in list(self.track_trajectories.keys()):
            if tid not in active_ids:
                if self.track_trajectories[tid] and (ref_time - self.track_trajectories[tid][-1][0] > 5.0):
                    del self.track_trajectories[tid]
                    self.track_obs_counts.pop(tid, None)
