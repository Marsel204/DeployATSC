from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time

import cv2
import numpy as np
from src.perception.frame_grabber import FrameGrabber


@dataclass(frozen=True)
class FramePacket:
    role: str
    sequence: int
    source_timestamp: float
    timestamp: float
    frame: np.ndarray


class FileSource:
    def __init__(self, source, role, offset=0.):
        if not Path(source).is_file(): raise ValueError(f"{role} video file does not exist")
        self.cap = cv2.VideoCapture(str(source))
        if not self.cap.isOpened():
            self.cap.release(); raise ValueError(f"Cannot open {role} video")
        self.role = role; self.offset = offset
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(self.fps) or self.fps <= 0:
            self.cap.release(); raise ValueError(f"{role} video has no valid time base")
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.sequence = 0; self.last_pts = None; self.ended = False
        self.timestamp_origin = "presentation_time"

    def read(self):
        if self.ended: return None
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True; return None
        self.sequence += 1
        pts = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        if not math.isfinite(pts) or pts < 0 or (self.last_pts is not None and pts <= self.last_pts):
            pts = (self.sequence-1)/self.fps
            self.timestamp_origin = "validated_fps_fallback"
        self.last_pts = pts
        return FramePacket(self.role, self.sequence, pts, pts-self.offset, frame)

    def close(self):
        self.cap.release()

    def health(self):
        return {"state":"ended" if self.ended else "ready", "fps":self.fps,
                "image_size":self.size, "sequence":self.sequence, "total_frames":self.total_frames,
                "timestamp_origin":self.timestamp_origin}


class ReplayPair:
    def __init__(self, system_source, acquisition_source, offset_sec=0., **kwargs):
        self.system = FileSource(system_source, "system")
        try:self.acquisition = FileSource(acquisition_source, "acquisition", offset_sec)
        except Exception:
            self.system.close(); raise
        self.peek = None; self.current_system = None
        self.ended = False; self.has_backlog = False

    def step(self):
        if self.ended:return []
        packets = []
        if not self.has_backlog:
            self.current_system = self.system.read()
            if self.current_system is None:
                self.ended = True; return []
            packets.append(self.current_system)
        due = self.current_system.timestamp
        self.has_backlog = False
        for _ in range(8):
            if self.peek is None:self.peek = self.acquisition.read()
            if self.peek is None or self.peek.timestamp > due+1e-7: break
            if self.peek.timestamp >= -1e-7:packets.append(self.peek)
            self.peek = None
        if self.peek is None and not self.acquisition.ended:
            self.peek = self.acquisition.read()
        self.has_backlog = self.peek is not None and self.peek.timestamp <= due+1e-7
        return packets

    def close(self):
        self.system.close(); self.acquisition.close()

    def health(self):
        return {"system":self.system.health(),"acquisition":self.acquisition.health()}


class LivePair:
    def __init__(self, system_source, acquisition_source, offset_sec=0., sizes=None, **kwargs):
        self.origin = time.monotonic(); self.offset = offset_sec
        sizes = sizes or {"system":(1920,1080), "acquisition":(1920,1080)}
        self.system = FrameGrabber(system_source, width=sizes["system"][0], height=sizes["system"][1],auto_loop=False,name="System")
        try:
            if not self.system.is_opened():raise ValueError("Cannot open system camera")
            self.acquisition = FrameGrabber(acquisition_source,width=sizes["acquisition"][0],height=sizes["acquisition"][1],auto_loop=False,name="Acquisition")
            if not self.acquisition.is_opened():raise ValueError("Cannot open acquisition camera")
        except Exception:
            self.system.stop()
            if hasattr(self,"acquisition"):self.acquisition.stop()
            raise
        self.system.start(); self.acquisition.start()
        self.ended = False; self.has_backlog = False

    def step(self):
        packets = []
        for role, source in [("system",self.system),("acquisition",self.acquisition)]:
            for _ in range(8):
                raw = source.read_observation()
                if raw is None:break
                mapped = raw["timestamp"]-self.origin-(self.offset if role=="acquisition" else 0)
                if mapped >= 0:
                    packets.append(FramePacket(role,raw["sequence"],raw["timestamp"],mapped,raw["frame"]))
        return sorted(packets,key=lambda p:p.timestamp)

    def close(self):
        self.system.stop(); self.acquisition.stop(); self.ended = True

    def discard_pending(self):
        self.system.discard_observation_queue(); self.acquisition.discard_observation_queue()

    def health(self):
        return {"system":self.system.observation_health(), "acquisition":self.acquisition.observation_health()}
