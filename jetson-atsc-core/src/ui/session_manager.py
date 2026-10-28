"""Session Manager for ATSC Web Dashboard and Replay Engine."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from typing import Any, Dict, Generator, List, Optional
import cv2
import numpy as np

CORE_ROOT = Path(__file__).resolve().parent.parent.parent

from src.ui.runtime_ownership import runtime_ownership, RuntimeConflict
from src.common.exception_cleanup import clear_exception_frames
from src.common.types import TrafficSnapshot, ActuationDecision
from src.perception.lens_rectifier import LensRectifier
from src.perception.phase_monitor import OpticalPhaseMonitor
from src.perception.detector_tracker import DetectorTracker, TrackedVehicle
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine

logger = logging.getLogger("session_manager")


class ReplaySession:
    """Manages an active replay session in a dedicated background worker thread."""

    def __init__(
        self,
        session_id: str,
        video_path: str,
        config_path: str,
        calib_path: Optional[str] = None,
        model_path: Optional[str] = None,
        anfis_path: Optional[str] = None,
        trigger_interval_sec: float = 15.0,
        step_mode: bool = False,
    ) -> None:
        self.session_id = session_id
        # Lifecycle ownership exists before the first allocation; every component starts absent.
        self._released=threading.Event();self._release_lock=threading.Lock();self.release_error=None
        self.cap=self.detector=self.anfis=self.decision_engine=None
        try:
            self.video_path = os.path.abspath(video_path)
            self.config_path = os.path.abspath(config_path)

            with open(self.config_path, "r", encoding="utf-8") as f:
                self.config = json.load(f)

            # Calibration & Models
            sys_cfg = self.config.get("system", {})
            default_calib = os.path.join(CORE_ROOT, "configs/rectilinear_calib.npz")
            self.calib_path = calib_path or sys_cfg.get("calibration_path", default_calib)
            if not os.path.isabs(self.calib_path):
                self.calib_path = os.path.join(CORE_ROOT, self.calib_path)

            default_model = os.path.join(CORE_ROOT, "models/Final.pt")
            self.model_path = model_path or sys_cfg.get("model_path", default_model)
            if not os.path.isabs(self.model_path):
                self.model_path = os.path.join(CORE_ROOT, self.model_path)

            self.anfis_path = anfis_path or os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx")
            self.trigger_interval_sec = trigger_interval_sec

            # Pipeline components
            self.rectifier = LensRectifier(self.calib_path)
            self.phase_monitor = OpticalPhaseMonitor.from_config(self.config)
            self.homography = HomographyEngine.from_config(self.config)
            self.metrics_extractor = TrafficMetricsExtractor.from_config(self.config, self.homography)
            self.detector = DetectorTracker(self.model_path, conf_threshold=0.28, imgsz=640)

            bounds = self.config.get("lane_metrics", {}).get("green_time_bounds_sec", [10.0, 120.0])
            self.anfis = ANFISInferenceEngine(
                self.anfis_path,
                prefer_cuda=False,
                min_green_sec=bounds[0],
                max_green_sec=bounds[1],
            )
            self.decision_engine = ATSCDecisionEngine(self.anfis)

            # Video capture
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.video_path}")

            self.native_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Runtime State
            self.frame_idx = 0
            self.cycle_count = 0
            self.last_trigger_timestamp = 0.0
            self.last_decision: Optional[ActuationDecision] = None
            self.current_snapshot: Optional[TrafficSnapshot] = None
            self.history_decisions: List[Dict[str, Any]] = []

            self.latest_jpeg: Optional[bytes] = None
            self.state: str = "paused" if step_mode else "running"
            self._is_stopped = False

            # Threading controls
            self._lock = threading.Lock()
            self._proc_lock = threading.Lock()
            self._step_event = threading.Event()
            self._play_event = threading.Event()
            if not step_mode:
                self._play_event.set()

            # Generate initial frame synchronously before worker thread starts
            self._process_single_frame()
            self._thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._thread.start()
        except Exception as error:
            # Exited frames (e.g. a failed first track call) would otherwise retain model owners.
            clear_exception_frames(error)
            self._release_resources();error.session_owner=self
            raise

    def _process_single_frame(self) -> bool:
        """Processes one video frame and updates internal state."""
        with self._proc_lock:
            ret, raw_frame = self.cap.read()
            if not ret or raw_frame is None:
                self.state = "completed"
                return False

            self.frame_idx += 1
            timestamp = self.frame_idx / self.native_fps

        # 1. Lens Rectification
        rectified = self.rectifier.rectify(raw_frame)

        # 2. Phase Monitor
        yellow_triggered = self.phase_monitor.process_frame(rectified, timestamp)

        # 3. Detection & Tracking
        tracks = self.detector.track([rectified], timestamp=timestamp)
        vehicles = tracks[0] if tracks else []
        self.detector.update_velocities(vehicles, self.homography.pixel_to_ground, timestamp=timestamp)

        # 4. PKJI Metrics
        raw_metrics = self.metrics_extractor.extract_approach_metrics(vehicles)
        snapshot = TrafficSnapshot.from_dict(raw_metrics, timestamp=timestamp)

        # 5. Periodic or Optical Yellow Trigger
        cycle_triggered = yellow_triggered
        if not cycle_triggered and self.trigger_interval_sec > 0:
            if (timestamp - self.last_trigger_timestamp) >= self.trigger_interval_sec and timestamp >= self.trigger_interval_sec:
                cycle_triggered = True
                self.last_trigger_timestamp = timestamp

        decision = None
        if cycle_triggered:
            self.cycle_count += 1
            decision = self.decision_engine.evaluate(snapshot, self.cycle_count)
            dec_entry = {
                "cycle_id": decision.cycle_id,
                "timestamp": round(timestamp, 2),
                "green_seconds": round(decision.green_seconds, 1),
                "v_w_pcu": round(snapshot.v_w_pcu, 2),
                "queue_meters": round(snapshot.queue_meters, 1),
                "occupancy_pct": round(snapshot.occupancy_pct, 1),
                "latency_ms": round(decision.inference_latency_ms, 2),
                "trigger_type": "OPTICAL_YELLOW" if yellow_triggered else "PERIODIC_INTERVAL",
            }
            self.history_decisions.append(dec_entry)

        # 6. Render HUD
        annotated = self._render_frame(rectified, vehicles, snapshot, decision)
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])

        with self._lock:
            self.current_snapshot = snapshot
            if decision:
                self.last_decision = decision
            self.latest_jpeg = jpeg.tobytes()

        return True

    def _worker_loop(self) -> None:
        """Background playback loop."""
        try:
            while not self._is_stopped:
                if self.state == "paused":
                    # Wait for step event or play event
                    self._step_event.wait(timeout=0.1)
                    if self._step_event.is_set():
                        self._step_event.clear()
                        has_more = self._process_single_frame()
                        if not has_more:
                            break
                    elif self._play_event.is_set():
                        self.state = "running"
                elif self.state == "running":
                    has_more = self._process_single_frame()
                    if not has_more:
                        break
                    time.sleep(max(0.01, 1.0 / self.native_fps))
                elif self.state == "completed":
                    break
        except Exception as error:
            # The same ownership rule applies after startup if an unhandled worker error
            # is retained by an excepthook: exited native-model frames must be cleared.
            clear_exception_frames(error)
            raise
        finally:self._release_resources()

    def _render_frame(
        self,
        frame: np.ndarray,
        vehicles: List[TrackedVehicle],
        snapshot: TrafficSnapshot,
        decision: Optional[ActuationDecision] = None,
    ) -> np.ndarray:
        """Renders comprehensive HUD overlay onto frame."""
        vis = frame.copy()

        # Draw corridor polygon
        self.homography.draw_corridor_overlay(vis)

        # Draw phase monitor ROI
        self.phase_monitor.draw_overlay(vis)

        # Draw tracked vehicles
        for v in vehicles:
            x1, y1, x2, y2 = [int(b) for b in v.bbox]
            # Color by status/class
            color = (0, 0, 255) if v.is_stopped else (0, 255, 0)
            if v.class_name == "motorcycle":
                class_color = (0, 230, 255)
            elif v.class_name in ("truck", "bus"):
                class_color = (0, 140, 255)
            else:
                class_color = (255, 200, 0)

            cv2.rectangle(vis, (x1, y1), (x2, y2), class_color, 2)

            g_str = f"({v.ground_pos[0]:.1f}m,{v.ground_pos[1]:.1f}m)" if v.ground_pos else ""
            label = f"#{v.track_id} {v.class_name} {v.velocity_mps:.1f}m/s {g_str}"
            cv2.putText(vis, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, class_color, 1)

            bx, by = int(v.bottom_center[0]), int(v.bottom_center[1])
            cv2.circle(vis, (bx, by), 4, color, -1)

        # Top Status Bar
        status_bar = f"FRAME: {self.frame_idx}/{self.total_frames} | V_w: {snapshot.v_w_pcu:.2f} PCU | Q: {snapshot.queue_meters:.1f}m | L: {snapshot.occupancy_pct:.1f}%"
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 40), (20, 20, 20), -1)
        cv2.putText(vis, status_bar, (15, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        # Active ANFIS decision banner
        active_dec = decision or self.last_decision
        if active_dec:
            dec_text = f"ANFIS GREEN: {active_dec.green_seconds:.1f}s (Cycle #{active_dec.cycle_id})"
            cv2.putText(vis, dec_text, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 50), 2)

        return vis

    def play(self) -> None:
        self.state = "running"
        self._play_event.set()

    def pause(self) -> None:
        self.state = "paused"
        self._play_event.clear()

    def step(self) -> None:
        self.state = "paused"
        self._play_event.clear()
        self._step_event.set()

    def _release_resources(self):
        with self._release_lock:
            if self._released.is_set():return
            # Callers run after the inference worker exited its calls, so model owners are
            # dropped even when the capture release fails; that failure keeps ownership.
            for name in ("detector","anfis","decision_engine"):setattr(self,name,None)
            try:
                if getattr(self,"cap",None) is not None:self.cap.release()
            except Exception as error:
                self.release_error=str(error);return
            self.release_error=None;self._released.set()

    @property
    def resources_released(self):
        return self._released.is_set()

    def stop(self,timeout=1.) -> None:
        self._is_stopped = True
        self.state = "stopped"
        for name in ("_play_event","_step_event"):
            if hasattr(self,name):getattr(self,name).set()
        if hasattr(self,"_thread") and self._thread.ident is not None:
            if self._thread is not threading.current_thread():self._thread.join(timeout=max(0,timeout))
            # Retry a release that failed after the worker exited.
            if not self._thread.is_alive():self._release_resources()
        else:self._release_resources()
        if not self.resources_released:raise RuntimeError("Legacy replay is stopping; resources remain owned")

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns instantaneous snapshot telemetry for the UI."""
        with self._lock:
            snap = self.current_snapshot.to_dict() if self.current_snapshot else {}
            dec = self.last_decision
            return {
                "session_id": self.session_id,
                "state": self.state,
                "frame_idx": self.frame_idx,
                "total_frames": self.total_frames,
                "fps": round(self.native_fps, 1),
                "metrics": {
                    "V_w": round(snap.get("v_w_pcu", 0.0), 2),
                    "Q": round(snap.get("queue_meters", 0.0), 1),
                    "L": round(snap.get("occupancy_pct", 0.0), 1),
                    "n_mc": int(snap.get("n_mc", 0)),
                    "n_lv": int(snap.get("n_lv", 0)),
                    "n_hv": int(snap.get("n_hv", 0)),
                    "total_vehicles": int(snap.get("total_vehicles", 0)),
                    "stopped_vehicles": int(snap.get("stopped_vehicles", 0)),
                },
                "decision": {
                    "cycle_id": dec.cycle_id,
                    "green_seconds": round(dec.green_seconds, 1),
                    "latency_ms": round(dec.inference_latency_ms, 2),
                    "is_clamped_min": dec.is_clamped_min,
                    "is_clamped_max": dec.is_clamped_max,
                } if dec else None,
                "history_decisions": self.history_decisions[-10:],
            }

    def get_latest_frame(self) -> Optional[bytes]:
        with self._lock:
            return self.latest_jpeg


class SessionManager:
    """Singleton-style manager coordinating active sessions, videos, and configs."""

    def __init__(self) -> None:
        self.active_session: Optional[ReplaySession] = None
        self.closing=False;self.starting=False;self.start_complete=threading.Event();self.start_complete.set()
        self._session_lock = threading.Lock()
        self._preview_manager=None

    def list_configs(self) -> List[Dict[str, Any]]:
        configs_dir = CORE_ROOT / "configs"
        results = []
        for path in sorted(configs_dir.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                desc = data.get("system", {}).get("description") or data.get("homography", {}).get("description") or "Calibration Profile"
                ground_dims = data.get("homography", {}).get("ground_dimensions_m", {})
                results.append({
                    "name": path.name,
                    "path": str(path),
                    "description": desc,
                    "width_m": ground_dims.get("width", 10.5),
                    "length_m": ground_dims.get("length", 45.0),
                })
            except Exception as e:
                logger.warning(f"Failed to parse config {path.name}: {e}")
        return results

    def list_videos(self) -> List[Dict[str, Any]]:
        video_dirs = [
            CORE_ROOT / "data" / "sample_videos",
            CORE_ROOT / "data" / "sample_videos" / "multi_angle_test",
            CORE_ROOT / "data" / "sample_videos" / "stress_test",
            CORE_ROOT / "data" / "uploads",
        ]
        results = []
        seen = set()
        for vdir in video_dirs:
            if not vdir.exists():
                continue
            for path in sorted(vdir.glob("*.mp4")):
                if path.name.endswith("_raw.mp4") or path.name in seen or path.stat().st_size < 1024:
                    continue
                seen.add(path.name)
                size_mb = round(path.stat().st_size / (1024 * 1024), 2)
                results.append({
                    "name": path.name,
                    "path": str(path),
                    "size_mb": size_mb,
                    "rel_dir": str(vdir.relative_to(CORE_ROOT)),
                })
        return results

    def start_session(
        self,
        video_path: str,
        config_path: str,
        trigger_interval_sec: float = 15.0,
        step_mode: bool = False,
    ) -> ReplaySession:
        with runtime_ownership.reserve("legacy"):
            with self._session_lock:
                # Admission and shutdown's closing decision are atomic; nothing waits under this lock.
                if self.closing:raise RuntimeConflict("Legacy manager is shutting down")
                self.starting=True;self.start_complete.clear()
            try:
                _,owner=runtime_ownership.current_owner()
                if owner is not None and owner is not self.active_session and not owner.resources_released:
                    raise RuntimeConflict("Another runtime owner still owns resources")
                try:self.stop_session()
                except RuntimeError as error:raise RuntimeConflict(str(error)) from error
                session_id = f"sess_{int(time.time())}"
                try:session=ReplaySession(session_id=session_id,video_path=video_path,config_path=config_path,
                                          trigger_interval_sec=trigger_interval_sec,step_mode=step_mode)
                except Exception as error:
                    failed=getattr(error,"session_owner",None)
                    if failed is not None and not failed.resources_released:
                        # A failed startup that still owns a resource stays registered until released.
                        self.active_session=failed;runtime_ownership.claim("legacy",failed)
                    raise
                self.active_session=session;runtime_ownership.claim("legacy",session)
                if self.closing:
                    session.stop(timeout=0)
                    raise RuntimeConflict("Shutdown requested during legacy startup")
                return session
            finally:self.starting=False;self.start_complete.set()

    def stop_session(self,timeout=1.) -> None:
        with self._session_lock:session=self.active_session
        if session:
            session.stop(timeout=timeout)
            if not session.resources_released:raise RuntimeConflict("Legacy replay still owns resources")
            with self._session_lock:
                if self.active_session is session:self.active_session=None

    def shutdown(self,timeout=5.):
        deadline=time.monotonic()+max(0,timeout)
        with self._session_lock:
            self.closing=True;preview_manager=self._preview_manager
        if preview_manager is not None:preview_manager.shutdown(max(0,deadline-time.monotonic()))
        self.start_complete.wait(max(0,deadline-time.monotonic()))
        previews_released=runtime_ownership.stop_previews(max(0,deadline-time.monotonic()))
        try:self.stop_session(timeout=max(0,deadline-time.monotonic()))
        except RuntimeError as error:
            result={"resources_released":False,"finalized":False,"error":str(error),"start_pending":self.starting}
            if not previews_released:result["preview_release_pending"]=True
            return result
        if not previews_released:
            return {"resources_released":False,"finalized":False,"start_pending":self.starting,"preview_release_pending":True}
        return {"resources_released":not self.starting,"finalized":not self.starting,"start_pending":self.starting}

    def download_youtube_video(self, url: str, section_duration_sec: int = 30) -> str:
        """Downloads YouTube clip and transcodes to standard H.264 MP4."""
        upload_dir = CORE_ROOT / "data" / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)

        base_name = f"yt_{int(time.time())}"
        raw_target = upload_dir / f"{base_name}_raw.mp4"
        final_target = upload_dir / f"{base_name}.mp4"

        # 1. Download via yt-dlp
        end_sec = section_duration_sec
        cmd_ytdlp = [
            "yt-dlp",
            "--download-sections", f"*00:00:05-00:00:{end_sec + 5:02d}",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
            url,
            "-o", str(raw_target),
        ]
        subprocess.run(cmd_ytdlp, check=True)

        # 2. Transcode via FFmpeg
        cmd_ffmpeg = [
            "ffmpeg", "-y",
            "-i", str(raw_target),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-r", "30",
            str(final_target),
        ]
        subprocess.run(cmd_ffmpeg, check=True)
        raw_target.unlink(missing_ok=True)

        return str(final_target)

    def get_video_snapshot(self, video_path: str) -> bytes:
        """Extracts a clean, unannotated first frame JPEG from the video."""
        if not os.path.isabs(video_path):
            video_path = os.path.join(CORE_ROOT, video_path)
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with self._session_lock:
            if self.closing:raise RuntimeConflict("Legacy manager is shutting down")
            if self._preview_manager is None:
                from src.ui.observation_session import ObservationManager
                self._preview_manager=ObservationManager()
            previews=self._preview_manager
        # Both HTTP snapshot entrypoints use the same clean-frame/ownership protocol.
        return previews.preview("system",video_path)

    def validate_and_save_custom_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validates user-drawn ROI coordinates and saves custom configuration JSON."""
        name = payload.get("name", f"custom_{int(time.time())}")
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name)
        if not clean_name.endswith(".json"):
            clean_name = f"{clean_name}.json"

        # 1. Validate Lamp ROI
        lamp_roi = payload.get("lamp_roi_pixel", {})
        for k in ("ymin", "xmin", "ymax", "xmax"):
            if k not in lamp_roi:
                raise ValueError(f"Missing '{k}' in lamp_roi_pixel")
        if lamp_roi["ymin"] >= lamp_roi["ymax"] or lamp_roi["xmin"] >= lamp_roi["xmax"]:
            raise ValueError(f"Invalid lamp_roi_pixel dimensions: ymin={lamp_roi['ymin']}, ymax={lamp_roi['ymax']}, xmin={lamp_roi['xmin']}, xmax={lamp_roi['xmax']}")

        # 2. Validate Corridor Perspective Anchors (4 Points)
        src_points = payload.get("src_points_pixel", [])
        if len(src_points) != 4:
            raise ValueError(f"src_points_pixel must contain exactly 4 points, got {len(src_points)}")

        src_arr = np.array(src_points, dtype=np.float32)
        int_pts = np.array(src_points, dtype=np.int32)
        if not cv2.isContourConvex(int_pts):
            raise ValueError("Corridor 4-point polygon must form a convex quadrilateral (no self-intersections or concavity)")

        # Validate Optional Multi-point Polygon Boundary (N >= 3)
        polygon_points = payload.get("polygon_points_pixel")
        if polygon_points is not None:
            if not isinstance(polygon_points, list) or len(polygon_points) < 3:
                raise ValueError("polygon_points_pixel must contain at least 3 points for a valid polygon area")
            corridor_polygon_pixel = [[float(p[0]), float(p[1])] for p in polygon_points]
        else:
            corridor_polygon_pixel = [[float(p[0]), float(p[1])] for p in src_points]

        ground_dims = payload.get("ground_dimensions_m", {"width": 8.0, "length": 40.0})
        width_m = float(ground_dims.get("width", 8.0))
        length_m = float(ground_dims.get("length", 40.0))

        dst_points = [
            [0.0, length_m],
            [width_m, length_m],
            [width_m, 0.0],
            [0.0, 0.0],
        ]
        dst_arr = np.array(dst_points, dtype=np.float32)

        # 3. Compute and validate Homography Matrix
        try:
            H, _ = cv2.findHomography(src_arr, dst_arr, method=0)
            if H is None:
                raise ValueError("Perspective anchors do not define a homography")
            cond = np.linalg.cond(H)
            if np.isnan(cond) or np.isinf(cond) or cond > 1e7:
                raise ValueError(f"Homography matrix is degenerate or ill-conditioned (condition number: {cond:.1e})")
        except Exception as e:
            raise ValueError(f"Invalid quadrilateral geometry: {e}")

        # Instantiate temporary HomographyEngine to validate and handle sample ruler calibration
        temp_engine = HomographyEngine(
            src_points_pixel=src_points,
            dst_points_ground=dst_points,
            ground_width_m=width_m,
            ground_length_m=length_m,
            corridor_polygon_pixel=corridor_polygon_pixel,
        )

        sample_ruler = payload.get("sample_ruler")
        scale_factor = 1.0
        if sample_ruler:
            if "p1" not in sample_ruler or "p2" not in sample_ruler or "length_m" not in sample_ruler:
                raise ValueError("sample_ruler must contain 'p1', 'p2', and 'length_m'")
            ruler_len = float(sample_ruler["length_m"])
            scale_factor = temp_engine.calibrate_from_sample_ruler(
                sample_ruler["p1"],
                sample_ruler["p2"],
                ruler_len,
            )
            width_m = temp_engine.ground_width
            length_m = temp_engine.ground_length
            dst_points = [[float(pt[0]), float(pt[1])] for pt in temp_engine.dst_points]

        # Calculate exact metric physical road area in m^2 using Shoelace formula
        ground_poly = [temp_engine.pixel_to_ground(p[0], p[1]) for p in corridor_polygon_pixel]
        n_pts = len(ground_poly)
        area_m2 = 0.5 * abs(sum(
            ground_poly[i][0] * ground_poly[(i + 1) % n_pts][1] - ground_poly[(i + 1) % n_pts][0] * ground_poly[i][1]
            for i in range(n_pts)
        ))
        if area_m2 <= 0.0:
            area_m2 = width_m * length_m

        # 4. Construct complete config schema
        homography_section: Dict[str, Any] = {
            "description": "Sidewalk-curb perspective projection to physical metric ground plane",
            "src_points_pixel": [[float(p[0]), float(p[1])] for p in src_points],
            "ground_dimensions_m": {
                "width": round(width_m, 2),
                "length": round(length_m, 2),
            },
            "dst_points_ground": dst_points,
            "corridor_polygon_pixel": corridor_polygon_pixel,
        }
        if sample_ruler:
            homography_section["sample_ruler"] = {
                "p1": [float(sample_ruler["p1"][0]), float(sample_ruler["p1"][1])],
                "p2": [float(sample_ruler["p2"][0]), float(sample_ruler["p2"][1])],
                "length_m": float(sample_ruler["length_m"]),
                "scale_factor": round(float(scale_factor), 4),
            }

        config_data = {
            "system": {
                "description": payload.get("description", "User-Drawn Custom Calibration"),
                "mode": "SINGLE_CAM",
                "calibration_path": "configs/rectilinear_calib.npz",
                "model_path": "models/Final.pt",
            },
            "homography": homography_section,
            "traffic_light_monitor": {
                "description": "Optical Yellow Phase Visual Trigger with temporal debounce and cooldown lockout",
                "lamp_roi_pixel": {
                    "ymin": int(lamp_roi["ymin"]),
                    "xmin": int(lamp_roi["xmin"]),
                    "ymax": int(lamp_roi["ymax"]),
                    "xmax": int(lamp_roi["xmax"]),
                },
                "hsv_thresholds": {
                    "hue_min": 18,
                    "hue_max": 35,
                    "sat_min": 120,
                    "sat_max": 255,
                    "val_min": 150,
                    "val_max": 255,
                },
                "active_ratio_threshold": 0.25,
                "debounce_frame_count": 3,
                "cooldown_lockout_seconds": 15.0,
            },
            "lane_metrics": {
                "description": "Indonesian PKJI 2014 Passenger Car Equivalent (PCE) weighted flow",
                "corridor_polygon": corridor_polygon_pixel,
                "physical_road_area_m2": round(float(area_m2), 2),
                "pcu_weights": {
                    "motorcycle": 0.4,
                    "car": 1.0,
                    "truck": 1.6,
                    "bus": 1.6,
                },
                "stopped_speed_threshold_mps": 1.0,
                "green_time_bounds_sec": [10.0, 120.0],
            },
        }

        save_path = CORE_ROOT / "configs" / clean_name
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        return {
            "status": "saved",
            "config_path": str(save_path),
            "name": clean_name,
            "config": config_data,
        }


# Global instance
manager = SessionManager()
