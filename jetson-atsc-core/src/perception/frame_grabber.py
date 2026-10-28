"""
GStreamer Hardware-Accelerated Video and Stream Ingest Module.

Supports:
- USB Action Cameras (/dev/video*) via v4l2src and Jetson nvvidconv
- Local MP4 video files with hardware nvv4l2decoder and auto-looping
- Low-latency RTSP and municipal HLS network streams
- Thread-safe zero-latency buffer management preventing frame lag
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from typing import Optional, Tuple
import cv2
import numpy as np
from src.common.source_identity import public_source


class FrameGrabber:
    """
    Asynchronous, non-blocking frame capture engine.

    Launches a dedicated acquisition thread that pulls frames from GStreamer
    or standard V4L2 pipelines, always serving the latest frame to prevent
    the inference pipeline from accumulating stale buffered images.
    """

    def __init__(
        self,
        source: str | int,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        auto_loop: bool = True,
        name: str = "Camera",
    ) -> None:
        """
        Initialize the FrameGrabber.

        Args:
            source: Device path ('/dev/video0'), integer device index (0),
                    local video file path ('test.mp4'), or stream URL ('rtsp://...').
            width: Capture width in pixels (default 1920).
            height: Capture height in pixels (default 1080).
            fps: Frame rate (default 30).
            auto_loop: Whether to loop local video files infinitely.
            name: Identifier for logging.
        """
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.auto_loop = auto_loop
        self.name = name

        self.is_file: bool = False
        self.is_rtsp: bool = False
        self._inspect_source_type()

        # Thread synchronization
        self._lock = threading.Lock()
        self._running = False
        self._lifecycle_lock = threading.RLock()
        self._released = threading.Event()
        self._stop_requested = False
        self._thread: Optional[threading.Thread] = None

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_timestamp: float = 0.0
        self._latest_monotonic: float = 0.0
        self._observation_queue = deque(maxlen=16)
        self._frame_count: int = 0
        self._dropped_frames: int = 0
        self._is_opened: bool = False

        self.cap: Optional[cv2.VideoCapture] = None
        try:self._init_capture()
        except BaseException as error:
            self._stop_requested=True
            error.resource_owner=self
            try:self._release_capture()
            except Exception:pass
            raise

    def _inspect_source_type(self) -> None:
        """Detects whether source is a file, RTSP stream, or V4L2 device."""
        if isinstance(self.source, int):
            self.is_file = False
            self.is_rtsp = False
        else:
            s_str = str(self.source).strip()
            if s_str.startswith("rtsp://") or s_str.startswith("http://") or s_str.startswith("https://"):
                self.is_rtsp = True
                self.is_file = False
            elif os.path.isfile(s_str):
                self.is_file = True
                self.is_rtsp = False
            else:
                self.is_file = False
                self.is_rtsp = False

    def _build_gstreamer_pipeline(self) -> Optional[str]:
        """
        Constructs optimized GStreamer pipelines for NVIDIA Jetson hardware acceleration.
        Returns None if using standard OpenCV capture fallback.
        """
        if self.is_file:
            # Jetson hardware video file decoding pipeline
            return (
                f"filesrc location={self.source} ! "
                f"qtdemux ! h264parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true sync=false"
            )
        elif self.is_rtsp:
            # Low-latency RTSP pipeline
            return (
                f"rtspsrc location={self.source} latency=100 ! "
                f"rtph264depay ! h264parse ! nvv4l2decoder ! "
                f"nvvidconv ! video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true sync=false"
            )
        else:
            # V4L2 USB camera pipeline
            dev = f"/dev/video{self.source}" if isinstance(self.source, int) else self.source
            return (
                f"v4l2src device={dev} ! "
                f"video/x-raw, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
                f"videoconvert ! video/x-raw, format=BGR ! "
                f"appsink drop=true sync=false"
            )

    def _init_capture(self) -> None:
        """Attempts to open capture using GStreamer, falling back to standard OpenCV."""
        gst_pipeline = self._build_gstreamer_pipeline()

        # Try GStreamer hardware acceleration first
        if gst_pipeline:
            self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                self._is_opened = True
                print(f"[{self.name}] Opened GStreamer hardware pipeline successfully.")
                return
            else:
                if self.cap:
                    self.cap.release()
                    self.cap = None
                print(f"[{self.name}] GStreamer pipeline unavailable. Falling back to standard backend...")

        # Fallback to standard OpenCV capture
        src_val = int(self.source) if str(self.source).isdigit() else self.source
        self.cap = cv2.VideoCapture(src_val)
        if not self.is_file and not self.is_rtsp:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._is_opened = self.cap.isOpened()
        if self._is_opened:
            print(f"[{self.name}] Opened standard capture device: {public_source(self.source)}")
        else:
            print(f"[{self.name}] WARNING: Failed to open capture source: {public_source(self.source)}")

    def start(self) -> FrameGrabber:
        """Starts the background acquisition thread."""
        with self._lifecycle_lock:
            if self._stop_requested:raise RuntimeError("Capture has been stopped")
            if self._running:return self
            self._running = True
            self._thread = threading.Thread(target=self._capture_worker, daemon=True, name=f"Grabber-{self.name}")
            try:self._thread.start()
            except BaseException as error:
                self._running = False
                self._stop_requested = True
                error.resource_owner=self
                try:self._release_capture()
                except Exception:pass
                raise
        return self

    def _release_capture(self):
        with self._lifecycle_lock:
            if self._released.is_set():return
            if self.cap is not None:self.cap.release()
            self._is_opened = False
            self._released.set()

    @property
    def resources_released(self):
        return self._released.is_set()

    def _capture_worker(self) -> None:
        """Continuously pulls frames from the device into the latest frame buffer."""
        try:
            while self._running:
                if self.cap is None or not self.cap.isOpened():
                    time.sleep(0.1)
                    continue

                ret, frame = self.cap.read()
                now = time.time()
                acquired = time.monotonic()

                if not ret or frame is None:
                    if self.is_file and self.auto_loop:
                        # Auto-loop video file for continuous testing
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.01)
                        continue
                    else:
                        # Temporary read glitch or stream disconnect
                        time.sleep(0.02)
                        continue

                with self._lock:
                    self._latest_frame = frame
                    self._latest_timestamp = now
                    self._frame_count += 1
                    self._latest_monotonic = acquired
                    if len(self._observation_queue) == self._observation_queue.maxlen:
                        self._dropped_frames += 1
                    self._observation_queue.append({"sequence":self._frame_count,
                        "timestamp":acquired,"frame":frame})

                # Prevent 100% CPU spinning on fast disk reads
                if self.is_file:
                    time.sleep(1.0 / (self.fps * 1.5))
        finally:
            self._running = False
            # A failed release stays unconfirmed; stop() retries it and reports the error to its caller.
            try:self._release_capture()
            except Exception:pass

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        """
        Retrieves the latest ingested frame without blocking.

        Returns:
            Tuple of (success: bool, frame: Optional[np.ndarray], timestamp: float)
        """
        with self._lock:
            if self._latest_frame is None:
                return False, None, 0.0
            return True, self._latest_frame.copy(), self._latest_timestamp

    def is_opened(self) -> bool:
        """Returns whether the capture stream is open and functional."""
        return self._is_opened

    def read_observation(self):
        """Single-consumer, sequence-aware read; the legacy latest-frame API remains."""
        with self._lock:
            return self._observation_queue.popleft() if self._observation_queue else None

    def discard_observation_queue(self):
        with self._lock:self._observation_queue.clear()

    def observation_health(self):
        with self._lock:
            age = time.monotonic()-self._latest_monotonic if self._latest_monotonic else None
            return {"state":"ready" if age is not None and age<=1 else "stale",
                    "age_sec":age,"sequence":self._frame_count,
                    "release_pending":self._stop_requested and not self.resources_released,
                    "resources_released":self.resources_released,
                    "timestamp_origin":"host_monotonic_receive","timestamp_fallback_count":0,
                    "dropped_frames":self._dropped_frames,"queue_depth":len(self._observation_queue),
                    "image_size":[self.width,self.height],"fps":self.fps}

    def stop(self, timeout=1.0) -> None:
        """Request exit; only the read owner releases a started capture."""
        with self._lifecycle_lock:
            self._stop_requested = True
            self._running = False
            thread = self._thread
            if thread is None or thread.ident is None:
                self._release_capture()
                return
        if thread is not threading.current_thread():thread.join(timeout=max(0,timeout))
        if not thread.is_alive():self._release_capture()


if __name__ == "__main__":
    # Test synthetic or fallback grabber
    grabber = FrameGrabber(source=0, name="TestCam")
    print(f"Grabber opened: {grabber.is_opened()}")
    grabber.stop()
