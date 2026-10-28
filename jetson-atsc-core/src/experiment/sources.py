from __future__ import annotations

from dataclasses import dataclass
import math
import hashlib
from pathlib import Path
import time

import cv2
import numpy as np
from src.perception.frame_grabber import FrameGrabber
from src.common.source_identity import public_source, safe_text


def release_all(steps):
    """Attempt every independent release, then re-raise so the owner keeps and retries what failed."""
    errors = []
    for step in steps:
        try:step()
        except Exception as error:errors.append(error)
    if len(errors) == 1:raise errors[0]
    if errors:raise RuntimeError("; ".join(safe_text(error) for error in errors))


@dataclass(frozen=True)
class FramePacket:
    role: str
    sequence: int
    source_timestamp: float
    timestamp: float
    frame: np.ndarray
    timestamp_origin: str = "unknown"


class FileSource:
    def __init__(self, source, role, offset=0.):
        self.cap=None;self.resources_released=False
        self.role=role;self.offset=offset;self.fps=None;self.size=None;self.total_frames=None
        self.sequence=0;self.last_pts=None;self.ended=False;self.initialized=False;self.timestamp_origin="presentation_time";self.timestamp_fallback_count=0
        try:
            if not Path(source).is_file():raise ValueError(f"{role} video file does not exist")
            # Identity is file content, never a digest of a source URL/string.
            digest=hashlib.sha256()
            with open(source,"rb") as stream:
                for block in iter(lambda:stream.read(1024*1024),b""):digest.update(block)
            self.media={"source":public_source(source),"sha256":digest.hexdigest(),"bytes":Path(source).stat().st_size}
            self.cap=cv2.VideoCapture(str(source))
            if not self.cap.isOpened():raise ValueError(f"Cannot open {role} video")
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(self.fps) or self.fps <= 0:raise ValueError(f"{role} video has no valid time base")
            self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.initialized=True
        except Exception as error:
            # Assignment in the pair may never complete. Propagate this leaf before cleanup,
            # and preserve the acquisition error even when native release also fails.
            error.resource_owner=self
            try:self.close()
            except Exception:pass
            raise

    def read(self):
        if self.ended: return None
        ok, frame = self.cap.read()
        if not ok:
            self.ended = True; return None
        self.sequence += 1
        pts = self.cap.get(cv2.CAP_PROP_POS_MSEC)/1000
        self.timestamp_origin = "presentation_time"
        if not math.isfinite(pts) or pts < 0 or (self.last_pts is not None and pts <= self.last_pts):
            pts = (self.sequence-1)/self.fps
            self.timestamp_origin = "validated_fps_fallback"
            self.timestamp_fallback_count += 1
        self.last_pts = pts
        return FramePacket(self.role, self.sequence, pts, pts-self.offset, frame, self.timestamp_origin)

    def close(self):
        if not self.resources_released:
            if self.cap is not None:self.cap.release()
            self.resources_released = True

    def health(self):
        state=("ended" if self.ended else "ready") if self.initialized else ("released" if self.resources_released else "release_pending")
        return {"state":state, "fps":self.fps,
                "image_size":self.size, "sequence":self.sequence, "total_frames":self.total_frames,
                "timestamp_origin":self.timestamp_origin,"timestamp_fallback_count":self.timestamp_fallback_count,
                "offset_sec":self.offset,"resources_released":self.resources_released}


class ReplayPair:
    def __init__(self, system_source, acquisition_source, offset_sec=0., **kwargs):
        self.system=self.acquisition=None
        try:
            for role,source,offset in (("system",system_source,0.),("acquisition",acquisition_source,offset_sec)):
                setattr(self,role,FileSource(source,role,offset))
        except Exception as error:
            leaf=getattr(error,"resource_owner",None)
            if leaf is not None:setattr(self,role,leaf)
            error.resource_owner=self
            try:self.close()
            except Exception:pass
            raise
        self.peek = None; self.current_system = None
        self.ended = False; self.has_backlog = False

    def media_identity(self):
        return {role:getattr(self,role).media for role in ("system","acquisition")}

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

    @property
    def resources_released(self):
        return all(source.resources_released for source in self._sources())

    def _sources(self):
        return [source for source in (self.system,self.acquisition) if source is not None]

    def close(self):
        release_all([source.close for source in self._sources()])

    def health(self):
        return {role:source.health() if source is not None else {"state":"not_started","resources_released":True}
                for role,source in (("system",self.system),("acquisition",self.acquisition))}


class LivePair:
    def __init__(self, system_source, acquisition_source, offset_sec=0., sizes=None, **kwargs):
        self.origin = time.monotonic(); self.offset = offset_sec
        sizes = sizes or {"system":(1920,1080), "acquisition":(1920,1080)}
        self.system=self.acquisition=None
        try:
            for role,source in (("system",system_source),("acquisition",acquisition_source)):
                setattr(self,role,FrameGrabber(source,width=sizes[role][0],height=sizes[role][1],auto_loop=False,name=role.title()))
                if not getattr(self,role).is_opened():raise ValueError(f"Cannot open {role} camera")
            for role in ("system","acquisition"):getattr(self,role).start()
        except Exception as error:
            leaf=getattr(error,"resource_owner",None)
            if leaf is not None:setattr(self,role,leaf)
            error.resource_owner = self
            # An unconfirmed release stays visible through resources_released; the owner retries close().
            try:release_all([source.stop for source in self._sources()])
            except Exception:pass
            raise
        self.ended = False; self.has_backlog = False

    def step(self):
        packets = []
        for role, source in [("system",self.system),("acquisition",self.acquisition)]:
            for _ in range(8):
                raw = source.read_observation()
                if raw is None:break
                mapped = raw["timestamp"]-self.origin-(self.offset if role=="acquisition" else 0)
                if mapped >= 0:
                    packets.append(FramePacket(role,raw["sequence"],raw["timestamp"],mapped,raw["frame"],"host_monotonic_receive"))
        return sorted(packets,key=lambda p:p.timestamp)

    @property
    def resources_released(self):
        return all(source.resources_released for source in self._sources())

    def _sources(self):
        return [source for source in (self.system,self.acquisition) if source is not None]

    def close(self, timeout=1.):
        deadline=time.monotonic()+timeout
        try:release_all([lambda source=source:source.stop(timeout=max(0,deadline-time.monotonic())) for source in self._sources()])
        finally:self.ended = True

    def discard_pending(self):
        self.system.discard_observation_queue(); self.acquisition.discard_observation_queue()

    def health(self):
        return {role:getattr(self,role).observation_health() if getattr(self,role) is not None else
                {"state":"not_started","resources_released":True,"release_pending":False}
                for role in ("system","acquisition")}
