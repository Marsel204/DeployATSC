"""
Asynchronous Ring-Buffered Ground-Truth MP4 Video Recorder.

Writes Kamera Akuisisi (and/or Kamera Sistem) video to disk in a dedicated
background thread using a ring-buffer queue, ensuring file I/O operations
never stall real-time edge perception or actuation FPS.
"""

from __future__ import annotations

from datetime import datetime
import os
import queue
import threading
import time
from typing import Optional, Tuple
import cv2
import numpy as np


class AsyncRecorder:
    """
    High-throughput non-blocking video recorder with timestamp watermarking.
    """

    def __init__(
        self,
        output_dir: str = "recordings",
        filename_prefix: str = "kamera_akuisisi",
        width: int = 1920,
        height: int = 1080,
        fps: float = 30.0,
        max_buffer_size: int = 120,
        codec: str = "mp4v",
    ) -> None:
        """
        Args:
            output_dir: Directory where MP4 segments are saved.
            filename_prefix: Prefix identifier for the output file.
            width: Frame width in pixels.
            height: Frame height in pixels.
            fps: Video recording frame rate.
            max_buffer_size: Maximum frames allowed in ring buffer before dropping oldest.
            codec: FourCC codec string ('mp4v' or 'avc1').
        """
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.width = width
        self.height = height
        self.fps = fps
        self.max_buffer_size = max_buffer_size
        self.codec = codec

        os.makedirs(self.output_dir, exist_ok=True)

        self._frame_queue: queue.Queue[Tuple[np.ndarray, Optional[str]]] = queue.Queue(
            maxsize=self.max_buffer_size
        )
        self._running = False
        self._write_lock = threading.Lock()
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None

        self.current_filepath: str = ""
        self.recorded_frames: int = 0
        self.dropped_frames: int = 0

    def start(self) -> AsyncRecorder:
        """Initializes the VideoWriter and starts the writer worker thread."""
        if self._running:
            return self

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.filename_prefix}_{timestamp_str}.mp4"
        self.current_filepath = os.path.join(self.output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self._writer = cv2.VideoWriter(
            self.current_filepath,
            fourcc,
            float(self.fps),
            (self.width, self.height),
        )

        if not self._writer.isOpened():
            print(f"[AsyncRecorder] WARNING: Could not open VideoWriter for {self.current_filepath} with codec {self.codec}. Trying MJPG...")
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            filename = f"{self.filename_prefix}_{timestamp_str}.avi"
            self.current_filepath = os.path.join(self.output_dir, filename)
            self._writer = cv2.VideoWriter(
                self.current_filepath,
                fourcc,
                float(self.fps),
                (self.width, self.height),
            )

        self._running = True
        self._thread = threading.Thread(target=self._writer_worker, daemon=True, name="AsyncVideoWriter")
        self._thread.start()
        print(f"[AsyncRecorder] Started recording to: {self.current_filepath}")
        return self

    def enqueue_frame(self, frame: np.ndarray, watermark_text: Optional[str] = None) -> bool:
        """
        Enqueues a frame for asynchronous encoding and writing.
        Non-blocking: if queue is full, drops the oldest frame.
        """
        if not self._running:
            return False

        # If buffer is completely full, pop oldest frame to maintain zero-latency guarantee
        if self._frame_queue.full():
            try:
                _ = self._frame_queue.get_nowait()
                self.dropped_frames += 1
            except queue.Empty:
                pass

        try:
            self._frame_queue.put_nowait((frame.copy(), watermark_text))
            return True
        except queue.Full:
            self.dropped_frames += 1
            return False

    def _writer_worker(self) -> None:
        """Pulls frames from the ring-buffer queue and writes to MP4 container."""
        while self._running or not self._frame_queue.empty():
            try:
                frame, watermark = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if frame is None or self._writer is None:
                continue

            # Overlay optional timestamp/cycle watermark
            if watermark:
                cv2.putText(
                    frame,
                    watermark,
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                )

            # Ensure correct resolution
            if (frame.shape[1], frame.shape[0]) != (self.width, self.height):
                frame = cv2.resize(frame, (self.width, self.height))

            with self._write_lock:
                if self._writer is not None:
                    self._writer.write(frame)
                    self.recorded_frames += 1
            self._frame_queue.task_done()

    def stop(self) -> None:
        """Flushes buffered frames and safely finalizes MP4 container."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

        with self._write_lock:
            if self._writer:
                self._writer.release()
                self._writer = None

        print(
            f"[AsyncRecorder] Finalized {self.current_filepath} (Recorded: {self.recorded_frames}, Dropped: {self.dropped_frames})"
        )
