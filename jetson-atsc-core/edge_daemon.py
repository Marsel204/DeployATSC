"""
Central Multi-Threaded Edge Orchestrator Daemon for Adaptive Traffic Signal Controller (ATSC).

Deployed on: NVIDIA Jetson Orin Nano (8GB, JetPack 6.x)
Coordinates:
- Ingest: GStreamer hardware acceleration for dual Brica B-PRO5 Alpha action cameras
- Rectification: Precomputed zero-latency Kannala-Brandt cv2.remap distortion flattening
- Perception: Batched YOLO11 TensorRT inference + ByteTrack tracking
- Analytics: Sidewalk-curb homography Inverse Perspective Mapping (IPM) & PKJI 2014 metrics
- Trigger: Optical Yellow Phase Visual Debouncer with 15s lockout cooldown
- Control: Pure Decoupled ATSCDecisionEngine (126-parameter Sugeno ANFIS ONNX)
- Hardware Actuation: RS-485 cabinet packet dispatch & 1.0s heartbeat watchdog
- Dynamic Thermal Guard: Non-destructive software fallback (DUAL_CAM -> SINGLE_CAM)
- Pluggable Sinks: Ledger CSV, Async MP4 Video Recorder, and Async LLM Quality Auditor
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import signal
import sys
import time
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

# Ensure project root is in python path
CORE_ROOT = os.path.dirname(os.path.abspath(__file__))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.common.types import (
    TrafficSnapshot,
    ActuationDecision,
    HardwareTelemetry,
    CycleEvent,
)
from src.perception.lens_rectifier import LensRectifier
from src.perception.phase_monitor import OpticalPhaseMonitor
from src.perception.frame_grabber import FrameGrabber
from src.perception.detector_tracker import DetectorTracker, TrackedVehicle
from src.analytics.homography_engine import HomographyEngine
from src.analytics.traffic_metrics import TrafficMetricsExtractor
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine
from src.control.signal_interface import SignalInterface
from src.monitoring.system_telemetry import SystemTelemetry
from src.sinks.base import CycleSink
from src.sinks.ledger_sink import LedgerSink
from src.sinks.recorder_sink import RecorderSink
from src.sinks.llm_sink import LLMAuditSink


class EdgeDaemon:
    """
    Central production daemon orchestrating real-time ATSC edge operations.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.is_running = False
        self.cycle_count = 0

        # Load configurations
        self.config_path = os.path.abspath(args.config)
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.hw_config_path = os.path.abspath(args.hardware_config)
        with open(self.hw_config_path, "r", encoding="utf-8") as f:
            self.hw_config = json.load(f)

        # Operating mode
        self.mode = args.mode or self.config.get("system", {}).get("mode", "DUAL_CAM")
        self.auto_thermal_fallback = self.config.get("system", {}).get("auto_thermal_fallback", True)
        self.acq_enabled = (self.mode == "DUAL_CAM")

        # Paths
        self.calib_path = os.path.join(CORE_ROOT, "configs/brica_fisheye_calib.npz")
        self.anfis_path = os.path.join(CORE_ROOT, "models/anfis_sugeno.onnx")
        self.yolo_engine_path = os.path.join(CORE_ROOT, "models/yolo11s_fp16.engine")
        self.yolo_pt_path = os.path.join(CORE_ROOT, "models/Final.pt")
        self.ledger_path = os.path.abspath(args.ledger)

        # Video sources
        cameras_cfg = self.config.get("cameras", {})
        self.source_sys = args.source_sys or cameras_cfg.get("kamera_sistem", {}).get("source", 0)
        self.source_acq = args.source_acq or cameras_cfg.get("kamera_akuisisi", {}).get("source", 1)

        # 1. Initialize Perception & Optics
        print("[EdgeDaemon] Initializing Lens Rectifier...")
        self.rectifier = LensRectifier(self.calib_path)

        print("[EdgeDaemon] Initializing Optical Yellow Phase Monitor...")
        self.phase_monitor = OpticalPhaseMonitor.from_config(self.config)

        print("[EdgeDaemon] Initializing Sidewalk-Curb Homography Engine...")
        self.homography = HomographyEngine.from_config(self.config)

        print("[EdgeDaemon] Initializing PKJI 2014 Traffic Metrics Extractor...")
        self.metrics_extractor = TrafficMetricsExtractor.from_config(self.config, self.homography)

        # 2. Initialize Pure Control Decision Engine
        print("[EdgeDaemon] Initializing 126-Parameter Sugeno ANFIS Engine & Decision Layer...")
        min_green = self.config.get("lane_metrics", {}).get("green_time_bounds_sec", [10.0, 120.0])[0]
        max_green = self.config.get("lane_metrics", {}).get("green_time_bounds_sec", [10.0, 120.0])[1]
        self.anfis_raw = ANFISInferenceEngine(
            self.anfis_path,
            prefer_cuda=True,
            min_green_sec=min_green,
            max_green_sec=max_green,
        )
        self.decision_engine = ATSCDecisionEngine(
            self.anfis_raw,
            min_green_sec=min_green,
            max_green_sec=max_green,
        )

        print("[EdgeDaemon] Initializing RS-485 Signal Interface & Watchdog...")
        hardware_settings = json.loads(json.dumps(self.hw_config))
        if args.mock_hardware:
            hardware_settings.setdefault("serial_interface", {})["mock_mode"] = True
        self.signal_interface = SignalInterface.from_config(hardware_settings)

        # 3. Initialize Monitoring & Telemetry
        print("[EdgeDaemon] Initializing System Telemetry & Thermal Guard...")
        sys_cfg = self.config.get("system", {})
        self.telemetry = SystemTelemetry(
            thermal_threshold_celsius=float(sys_cfg.get("thermal_threshold_celsius", 75.0)),
            min_fps_threshold=float(sys_cfg.get("min_fps_threshold", 15.0)),
            thermal_recovery_celsius=float(sys_cfg.get("thermal_recovery_celsius", 68.0)),
            recovery_fps_threshold=float(sys_cfg.get("recovery_fps_threshold", 20.0)),
        )

        # 4. Initialize Pluggable Sinks
        self.cycle_sinks: List[CycleSink] = []
        self.ledger_sink = LedgerSink(self.ledger_path)
        self.register_cycle_sink(self.ledger_sink)

        record_dir = os.path.join(CORE_ROOT, "recordings")
        self.recorder_sink = RecorderSink(output_dir=record_dir, filename_prefix="kamera_akuisisi")
        self.register_cycle_sink(self.recorder_sink)

        self.llm_sink = LLMAuditSink(on_audit_completed=self._handle_audit_completed)
        self.register_cycle_sink(self.llm_sink)

        # 5. Initialize YOLO11 Detector Tracker
        model_to_use = self.yolo_engine_path if os.path.exists(self.yolo_engine_path) else self.yolo_pt_path
        print(f"[EdgeDaemon] Initializing YOLO11 Detector & ByteTracker using: {model_to_use}...")
        self.detector = DetectorTracker(model_to_use, conf_threshold=0.28, imgsz=args.imgsz)

        # 6. Ingest Pipelines
        print(f"[EdgeDaemon] Initializing Ingest for Kamera Sistem: {self.source_sys}...")
        self.grabber_sys = FrameGrabber(
            source=self.source_sys,
            auto_loop=args.loop,
            name="KameraSistem",
        )

        self.grabber_acq: Optional[FrameGrabber] = None
        if self.mode == "DUAL_CAM" or args.source_acq:
            print(f"[EdgeDaemon] Initializing Ingest for Kamera Akuisisi: {self.source_acq}...")
            self.grabber_acq = FrameGrabber(
                source=self.source_acq,
                auto_loop=args.loop,
                name="KameraAkuisisi",
            )

        # Active state tracking
        self.last_actuated_green: Optional[float] = None
        self.last_decision: Optional[ActuationDecision] = None
        self.current_snapshot: TrafficSnapshot = TrafficSnapshot()
        self.current_metrics: Dict[str, Any] = self.current_snapshot.to_dict()

        # Setup OS Signal Intercepts
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def register_cycle_sink(self, sink: CycleSink) -> None:
        """Attaches an observer sink to handle completed cycle events."""
        if sink not in self.cycle_sinks:
            self.cycle_sinks.append(sink)

    def _signal_handler(self, sig, frame) -> None:
        """Handles clean shutdown on Ctrl+C (SIGINT) or SIGTERM."""
        print(f"\n[EdgeDaemon] Caught signal {sig}. Initiating graceful production shutdown...")
        self.is_running = False

    def _handle_audit_completed(self, audited_cycle: Dict[str, Any]) -> None:
        """Callback invoked when asynchronous LLM audit completes."""
        audit = audited_cycle.get("audit", {})
        cycle_id = audited_cycle.get("cycle_id", "N/A")
        severity = audit.get("severity", "NORMAL")
        critique = audit.get("critique", "")

        if audit.get("anomaly_detected", False):
            print(f"\n[AUDIT ALERT] Cycle {cycle_id} Flagged [{severity}]: {critique}")
        else:
            print(f"[AUDIT LOG] Cycle {cycle_id} Verified Nominal: {critique}")

        # Update CSV ledger row with LLM audit verdict via ledger sink
        self.ledger_sink.update_audit_result(cycle_id, audit)

    def start(self) -> None:
        """Starts all background support threads and enters the primary execution loop."""
        print("\n==================================================================")
        print("    NVIDIA JETSON ORIN NANO ATSC PRODUCTION RUNTIME DAEMON        ")
        print("==================================================================")
        print(f"Mode               : {self.mode}")
        print(f"Thermal Fallback   : {self.auto_thermal_fallback}")
        print(f"Kamera Sistem Ingest: {self.source_sys}")
        print(f"Kamera Akuisisi Ingest: {self.source_acq if self.acq_enabled else 'DISABLED'}")
        print(f"Ledger Output File : {self.ledger_path}")
        print("==================================================================\n")

        # Start components & sinks
        self.grabber_sys.start()
        if self.grabber_acq:
            self.grabber_acq.start()
        self.signal_interface.start_watchdog()

        for sink in self.cycle_sinks:
            sink.start()

        self.is_running = True
        self._run_loop()

    def _run_loop(self) -> None:
        """Main real-time edge control loop."""
        t_daemon_start = time.time()
        while self.is_running:
            if getattr(self.args, "max_cycles", None) and self.cycle_count >= self.args.max_cycles:
                print(f"\n[EdgeDaemon] Reached target cycle count ({self.args.max_cycles}). Initiating graceful shutdown...")
                self.is_running = False
                break
            if getattr(self.args, "duration", None) and (time.time() - t_daemon_start) >= self.args.duration:
                print(f"\n[EdgeDaemon] Reached target runtime duration ({self.args.duration}s). Initiating graceful shutdown...")
                self.is_running = False
                break

            t_loop_start = time.perf_counter()

            # 1. Grab Frame 1 (Kamera Sistem)
            ret1, raw_sys, ts1 = self.grabber_sys.read()
            if not ret1 or raw_sys is None:
                time.sleep(0.005)
                continue

            # 2. Grab Frame 2 (Kamera Akuisisi) if active
            ret2, raw_acq, ts2 = False, None, 0.0
            if self.acq_enabled and self.grabber_acq:
                ret2, raw_acq, ts2 = self.grabber_acq.read()

            # 3. Zero-Latency Fisheye Rectification
            rectified_sys = self.rectifier.rectify(raw_sys)

            # 4. Check Optical Yellow Phase Transition
            yellow_triggered = self.phase_monitor.process_frame(rectified_sys, ts1)

            # 5. Shared-Engine Batched Inference + Tracking
            frames_to_track = [rectified_sys]
            if self.acq_enabled and ret2 and raw_acq is not None:
                frames_to_track.append(raw_acq)

            batch_tracks = self.detector.track(frames_to_track, ts1)
            vehs_sys = batch_tracks[0] if len(batch_tracks) > 0 else []
            vehs_acq = batch_tracks[1] if len(batch_tracks) > 1 else []

            # 6. Update Kinematics & Traffic Metrics
            self.detector.update_velocities(vehs_sys, self.homography.pixel_to_ground)
            raw_metrics = self.metrics_extractor.extract_approach_metrics(vehs_sys)
            self.current_snapshot = TrafficSnapshot.from_dict(raw_metrics, timestamp=ts1)
            self.current_metrics = raw_metrics

            # 7. Update Kamera Akuisisi Stop-Line Discharge Tripwire
            if self.acq_enabled and vehs_acq:
                self.metrics_extractor.update_discharge_tripwire(vehs_acq)

            # 8. Dynamic Thermal Guard Evaluation (Non-destructive software gating)
            fps = self.telemetry.record_frame()
            if self.auto_thermal_fallback:
                new_mode, mode_changed, reason = self.telemetry.evaluate_fallback(self.mode)
                if mode_changed:
                    print(f"\n[DYNAMIC THERMAL GUARD] Mode transition: {self.mode} -> {new_mode}. Reason: {reason}")
                    self._transition_mode(new_mode)

            # 9. Yellow Phase Trigger Action: Actuation & Thesis Logging
            if yellow_triggered:
                self._handle_yellow_actuation(fps)

            # 10. Ground-Truth Recording Enqueue via RecorderSink
            if getattr(self.args, "record_annotated", False):
                annotated_frame = self.render_overlay(rectified_sys, vehs_sys)
                self.recorder_sink.enqueue_frame(annotated_frame)
            elif self.acq_enabled and ret2 and raw_acq is not None:
                watermark = f"ACQ | CYC:{self.cycle_count} | SERVED:{self.metrics_extractor.n_served} | {datetime.now().strftime('%H:%M:%S')}"
                self.recorder_sink.enqueue_frame(raw_acq, watermark)
            else:
                watermark = f"SYS | CYC:{self.cycle_count} | Q:{self.current_snapshot.queue_meters:.1f}m | {datetime.now().strftime('%H:%M:%S')}"
                self.recorder_sink.enqueue_frame(rectified_sys, watermark)

            # 11. Headed Debug Display (if not running headless)
            if not self.args.headless:
                self._render_gui(rectified_sys, vehs_sys)

            # Yield to prevent CPU thread starvation
            elapsed = (time.perf_counter() - t_loop_start) * 1000.0
            if elapsed < 5.0:
                time.sleep(0.002)

        self._cleanup()

    def _transition_mode(self, new_mode: str) -> None:
        """
        Transitions non-destructively between DUAL_CAM and SINGLE_CAM without tearing down threads.
        """
        self.mode = new_mode
        self.acq_enabled = (new_mode == "DUAL_CAM")
        if not self.acq_enabled:
            print("[EdgeDaemon] Kamera Akuisisi inference gated for thermal relief.")
        else:
            print("[EdgeDaemon] Kamera Akuisisi inference restored after thermal recovery.")

    def _handle_yellow_actuation(self, fps: float) -> None:
        """
        Coordinates pure ANFIS evaluation, serial actuation, and observer cycle event broadcast.
        """
        self.cycle_count += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        traffic = self.current_snapshot

        # 1. Evaluate Decision Engine
        decision = self.decision_engine.evaluate(traffic, self.cycle_count)
        self.last_decision = decision
        self.last_actuated_green = decision.green_seconds

        # 2. Dispatch RS-485 Actuation Packet to Cabinet
        dispatch_ok = self.signal_interface.dispatch_green_actuation(decision.green_seconds, phase_id=1)

        # 3. Tally Discharged Vehicles & Reset for Next Cycle
        n_served = self.metrics_extractor.reset_discharge_cycle()
        theor_capacity = round(decision.green_seconds / 2.0, 1)

        # 4. Hardware Telemetry Snapshot
        hw_metrics = self.telemetry.read_metrics()
        soc_temp = hw_metrics["soc_temp_c"]
        watchdog_status = "OK" if dispatch_ok and not self.signal_interface.is_in_failsafe else "FAILSAFE"

        telemetry = HardwareTelemetry(
            soc_temp_c=soc_temp,
            fps=fps,
            mode=self.mode,
            is_thermal_throttled=(self.mode == "SINGLE_CAM"),
        )

        print("\n=======================================================")
        print(f"           YELLOW PHASE TRIGGERED (CYCLE #{self.cycle_count})")
        print("=======================================================")
        print(f"Traffic State   : V_w={traffic.v_w_pcu:.2f} PCU | Q={traffic.queue_meters:.1f} m | L={traffic.occupancy_pct:.1f}%")
        print(f"Vehicles in ROI : Total={traffic.total_vehicles} (MC:{traffic.n_mc}, LV:{traffic.n_lv}, HV:{traffic.n_hv})")
        print(f"ANFIS Actuation : t_ANFIS = {decision.green_seconds:.1f} seconds (Latency: {decision.inference_latency_ms:.2f}ms)")
        print(f"Previous Served : N_served = {n_served} vehicles (Theor. Cap = {theor_capacity} PCU)")
        print(f"Hardware Status : Temp = {soc_temp:.1f}°C | FPS = {fps:.1f} | Mode = {self.mode}")
        print("=======================================================\n")

        # 5. Assemble CycleEvent and broadcast to all sinks
        event = CycleEvent(
            cycle_id=self.cycle_count,
            timestamp_iso=now_iso,
            traffic=traffic,
            decision=decision,
            telemetry=telemetry,
            n_served=n_served,
            theoretical_capacity_pcu=theor_capacity,
            watchdog_status=watchdog_status,
        )

        for sink in self.cycle_sinks:
            try:
                sink.on_cycle(event)
            except Exception as e:
                print(f"[EdgeDaemon] Error in sink {sink.__class__.__name__}: {e}")

    def render_overlay(self, frame: np.ndarray, vehicles: List[TrackedVehicle]) -> np.ndarray:
        """Renders perception overlays, corridor boundaries, and HUD."""
        display_frame = frame.copy()

        # Draw corridor IPM boundary
        self.homography.draw_corridor_overlay(display_frame)

        # Draw optical yellow monitor
        self.phase_monitor.draw_overlay(display_frame)

        # Draw tracked vehicle bounding boxes
        for v in vehicles:
            x1, y1, x2, y2 = [int(b) for b in v.bbox]
            color = (0, 0, 255) if v.is_stopped else (0, 255, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{v.track_id} {v.class_name} {v.velocity_mps:.1f}m/s"
            cv2.putText(display_frame, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Draw HUD Metrics Card
        self.metrics_extractor.draw_metrics_overlay(display_frame, self.current_metrics, self.last_actuated_green)

        # Draw system telemetry in top-right
        hw = self.telemetry.read_metrics()
        telemetry_str = f"MODE: {self.mode} | FPS: {hw['fps']:.1f} | TEMP: {hw['soc_temp_c']:.1f}C"
        cv2.putText(display_frame, telemetry_str, (display_frame.shape[1] - 420, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        return display_frame

    def _render_gui(self, frame: np.ndarray, vehicles: List[TrackedVehicle]) -> None:
        """Renders live perception overlays and HUD."""
        display_frame = self.render_overlay(frame, vehicles)

        # Show window
        cv2.imshow("ATSC Edge Perception - NVIDIA Jetson Orin Nano", cv2.resize(display_frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n[EdgeDaemon] 'q' pressed. Shutting down...")
            self.is_running = False

    def _cleanup(self) -> None:
        """Safely stops threads, releases sinks, and releases hardware descriptors."""
        print("\n[EdgeDaemon] Shutting down ingest pipelines and hardware interfaces...")
        self.grabber_sys.stop()
        if self.grabber_acq:
            self.grabber_acq.stop()
        self.signal_interface.stop()

        for sink in self.cycle_sinks:
            try:
                sink.close()
            except Exception as e:
                print(f"[EdgeDaemon] Error closing sink {sink.__class__.__name__}: {e}")

        if not self.args.headless:
            cv2.destroyAllWindows()
        print("[EdgeDaemon] Graceful shutdown complete. System safe.")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive Traffic Signal Controller Edge Daemon")
    parser.add_argument("--operation", choices=["observation", "legacy-control"], default="observation",
                        help="Observation never constructs cabinet hardware; controller operation is explicit")
    parser.add_argument("--source-mode", choices=["replay", "live"], default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", type=str, choices=["SINGLE_CAM", "DUAL_CAM"], default=None, help="Operating camera mode")
    parser.add_argument("--source_sys", type=str, default=None, help="Kamera Sistem source path, MP4 file, or RTSP URL")
    parser.add_argument("--source_acq", type=str, default=None, help="Kamera Akuisisi source path, MP4 file, or RTSP URL")
    parser.add_argument("--config", type=str, default=os.path.join(CORE_ROOT, "configs/intersection_roi.json"), help="Path to intersection ROI config")
    parser.add_argument("--hardware_config", type=str, default=os.path.join(CORE_ROOT, "configs/hardware_config.json"), help="Path to hardware config")
    parser.add_argument("--ledger", type=str, default=os.path.join(CORE_ROOT, "field_experiment_ledger.csv"), help="Path to experiment CSV ledger")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference resolution")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--record_annotated", action="store_true", help="Record HUD-annotated video frames instead of raw stream")
    parser.add_argument("--max_cycles", type=int, default=None, help="Stop daemon gracefully after N completed cycles")
    parser.add_argument("--duration", type=float, default=None, help="Stop daemon gracefully after N seconds")
    parser.add_argument("--loop", action="store_true", default=True, help="Auto-loop MP4 video sources")
    parser.add_argument("--mock_hardware", action="store_true", default=True, help="Operate serial signal interface in loopback mock mode")
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.operation == "observation":
        from pathlib import Path
        from scripts.launch_ui import serve_observation
        with open(args.config, encoding="utf-8") as stream:
            camera_config = json.load(stream).get("cameras", {})
        system = args.source_sys or camera_config.get("kamera_sistem", {}).get("source", "/dev/video0")
        acquisition = args.source_acq or camera_config.get("kamera_akuisisi", {}).get("source", "/dev/video1")
        source_mode = args.source_mode or ("replay" if Path(str(system)).is_file() and Path(str(acquisition)).is_file() else "live")
        serve_observation(host=args.host, port=args.port, source_mode=source_mode,
                          source_sys=str(system),source_acq=str(acquisition),config=args.config)
        return
    daemon = EdgeDaemon(args)
    daemon.start()


if __name__ == "__main__":
    main()
