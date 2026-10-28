"""One owner for two observation cameras; no controller imports or hardware calls."""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache, partial
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import tempfile
import threading
import time
import uuid

import cv2
import numpy as np

from src.ui.runtime_ownership import runtime_ownership, RuntimeConflict
from src.ui.runtime_metrics import AcquisitionRates
from src.common.exception_cleanup import clear_exception_frames
from src.analytics.traffic_metrics import TrafficMetricsExtractor
from src.common.types import TrafficSnapshot
from src.common.source_identity import public_data, public_source, resolve_source, safe_text
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine
from src.experiment.calibration import CameraCalibration, legacy_system_payload
from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.predictions import PredictionConflict
from src.experiment.retention import validate_budget
from src.experiment.sources import ReplayPair, LivePair
from src.perception.detector_tracker import DetectorTracker
from src.perception.observation_model import model_for_factory
from src.perception.lens_rectifier import LensRectifier

ROOT=Path(__file__).resolve().parents[2]


@lru_cache(maxsize=16)
def file_digest(path, modified):
    h=hashlib.sha256()
    with open(path,"rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def source_identity(value):
    value=resolve_source(value)
    if value.isdigit():value="/dev/video"+value
    return str(Path(value).resolve()) if "://" not in value else value


class ObservationConflict(RuntimeConflict):
    pass


class ReplacementFailed(RuntimeError):
    pass


class ObservationSession:
    @staticmethod
    def preflight(payload,storage_root,detector_factory=None,anfis_factory=None):
        """Validate static inputs without opening video devices or allocating models."""
        RunPolicy(**payload.get("policy",{}))
        validate_budget(payload.get("retention_budget",64))
        mode=payload.get("source_mode","replay")
        if mode not in ("live","replay"):raise ValueError("Source mode must be replay or live")
        calibrations={role:CameraCalibration(payload[role]) for role in ("system","acquisition")}
        if any(c.role!=role for role,c in calibrations.items()):raise ValueError("Calibration camera roles are mismatched")
        # The same pure extractor construction (numeric conversions included) that the session performs.
        system=calibrations["system"]
        try:TrafficMetricsExtractor.from_config(system.legacy_metrics_config(),system.homography,strict_motion=True)
        except (TypeError,ValueError) as error:raise ValueError(f"Invalid lane metrics configuration: {error}") from error
        sources=[c.source for c in calibrations.values()]
        shared=source_identity(sources[0])==source_identity(sources[1])
        if shared and (not payload.get("shared_test",False) or mode!="replay"):
            raise ValueError("A shared source requires explicit shared-video test mode; live inputs must differ")
        if mode=="replay":
            for source in sources:
                if not Path(source).is_file():raise ValueError("Replay video file does not exist")
                with open(source,"rb") as stream:stream.read(1)
        selection=model_for_factory(ROOT/"models",detector_factory)
        model=Path(selection["path"])
        for path,factory in ((model,detector_factory),(ROOT/"models/anfis_sugeno.onnx",anfis_factory)):
            hook=getattr(factory,"preflight",None)
            if hook:hook(str(path))
            elif factory is None or (path==model and selection["backend"]!="injected"):
                if not path.is_file() or not path.stat().st_size:raise ValueError(f"Required model is unavailable: {path.name}")
                with path.open("rb") as stream:stream.read(1)
        root=Path(storage_root);root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryFile(dir=root) as probe:probe.write(b"preflight");probe.flush()
        return calibrations

    def __init__(self,payload,storage_root,detector_factory=None,anfis_factory=None):
        self.preflight(payload,storage_root,detector_factory,anfis_factory)
        self.policy=RunPolicy(**payload.get("policy",{}))
        self.source_mode=payload.get("source_mode","replay")
        if self.source_mode not in ("replay","live"):raise ValueError("Source mode must be replay or live")
        self.calibrations={role:CameraCalibration(payload[role]) for role in ("system","acquisition")}
        if any(c.role!=role for role,c in self.calibrations.items()):raise ValueError("Calibration camera roles are mismatched")
        self.shared_test=bool(payload.get("shared_test",False))
        sources={role:c.source for role,c in self.calibrations.items()}
        shared=source_identity(sources["system"])==source_identity(sources["acquisition"])
        if shared and (not self.shared_test or self.source_mode!="replay"):
            raise ValueError("A shared source requires explicit shared-video test mode; live inputs must differ")
        self.shared_test=shared
        self.run_id="run_"+uuid.uuid4().hex[:16]
        self.lock=threading.RLock();self.stopping=threading.Event();self.wake=threading.Event()
        self.state="paused" if payload.get("step_mode",False) else "running"
        self.applied_state=self.state;self.control_seq=0;self.applied_control_seq=0
        self.controls=deque();self.packets=deque()
        self.error=None;self.storage_error=None;self.jpeg_frames={};self.clean_frames={}
        self.clean_jpeg_cache={};self.acquisition_rates=AcquisitionRates()
        self.preview_width=640;self.preview_published_at={}
        self.frame_times={};self.steps=0;self.created_monotonic=time.monotonic()
        # Camera 1 "frames" counts one-shot still-image inferences; its decoded frames are preview_frames.
        self.runtime={role:{"frames":0,"inference_total_ms":0.,"last_inference_ms":0.,"preview_frames":0,
            "inference_mode":"on_demand_single_frame" if role=="system" else "continuous"} for role in self.calibrations}
        self.camera1_latest=None
        self.initializing=True;self.ready=threading.Event();self.startup_error=None
        self.durable_revision=None;self.durable_generation=None
        self.pending_commit=None;self.storage_suspended=False;self.commit_sequence=0
        self.finalized=False;self.terminal_state="active";self.last_persisted_revision=None
        self._resources_released=False;self.release_error=None;self.persist_lock=threading.RLock();self.cleanup_lock=threading.Lock()
        self.cleanup_schedule_lock=threading.Lock();self.cleanup_schedule_error=None;self.thread=None
        self.detectors={};self.pair=None;self.decision=None
        self.last_persisted=0.;self.acquisition_ended=False;self.live_stale=False
        factory=ReplayPair if self.source_mode=="replay" else LivePair
        try:
            self.pair=factory(sources["system"],sources["acquisition"],self.policy.offset_sec,
                              sizes={role:c.size for role,c in self.calibrations.items()})
            if self.source_mode=="replay":
                for role,data in self.pair.health().items():
                    if tuple(data["image_size"])!=self.calibrations[role].size:
                        raise ValueError(f"{role} source dimensions do not match calibration")
            self.model_selection=model_for_factory(ROOT/"models",detector_factory)
            detector_factory=detector_factory or DetectorTracker
            anfis_factory=anfis_factory or ANFISInferenceEngine
            model=Path(self.model_selection["path"]);anfis=ROOT/"models/anfis_sugeno.onnx"
            for role in self.calibrations:
                self.detectors[role]=detector_factory(str(model),conf_threshold=.28,imgsz=640)
            for role in self.detectors:
                self.detectors[role].max_gap_sec=self.policy.max_gap_sec
            self.decision=ATSCDecisionEngine(anfis_factory(str(anfis),prefer_cuda=False))
            acquisition=self.calibrations["acquisition"]
            self.engine=ObservationEngine(self.run_id,self.policy,acquisition.roi,acquisition.side,self.decision.evaluate,acquisition.lamp_rois,
                scale_evidence={role:{"revision":c.revision,**c.scale_status} for role,c in self.calibrations.items()})
            self.engine.enable_retention(payload.get("retention_budget",64),Path(storage_root))
            if self.source_mode=="live":
                self.engine.predictions.clock=lambda:time.monotonic()-self.pair.origin
            system=self.calibrations["system"]
            self.metrics=TrafficMetricsExtractor.from_config(system.legacy_metrics_config(),system.homography,strict_motion=True)
            versions={}
            for package in ("torch","ultralytics","onnxruntime","opencv-python"):
                try:versions[package]=importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:versions[package]=None
            manifest={"operation":"observation","movement_id":self.policy.movement_id,
                "source_mode":self.source_mode,"shared_source_test":shared,"policy":asdict(self.policy),
                "sources":{role:c.payload["source"] for role,c in self.calibrations.items()},"calibrations":{role:{"revision":c.revision,"payload":c.payload,"scale":c.scale_status} for role,c in self.calibrations.items()},
                "model_hashes_scope":"configured_files",
                "model_hashes":{p.name:file_digest(str(p),p.stat().st_mtime_ns) for p in {model,ROOT/"models/Final.pt",anfis}},
                "detector_selection":self.model_selection,
                "package_versions":versions, "execution":self.execution_provenance(),
                "media_identity":self.pair.media_identity() if hasattr(self.pair,"media_identity") else {"state":"unknown"},
                "clock_provenance":self.clock_provenance()}
            manifest["inference_components"]={"detector":getattr(detector_factory,"__name__",type(detector_factory).__name__),
                "anfis":getattr(anfis_factory,"__name__",type(anfis_factory).__name__)}
            manifest["source_hashes"]={str(path.relative_to(ROOT)):file_digest(str(path),path.stat().st_mtime_ns)
                for path in sorted((ROOT/"src").rglob("*.py"))}
            self.store=ExperimentStore(storage_root,self.run_id,manifest)
            self.thread=threading.Thread(target=self._worker,daemon=True,name="Observation-"+self.run_id)
            self.thread.start()
        except Exception as error:
            # Preserve traceback locations without retaining exited model/capture frames.
            clear_exception_frames(error)
            if self.pair is None:self.pair=getattr(error,"resource_owner",None)
            self.error=safe_text(error);self.state=self.applied_state="error";self.stopping.set()
            error.session_owner=self
            # No measurement worker was started. This cleanup thread becomes the owner.
            self._ensure_cleanup()
            raise
        self.ready.wait()
        if self.startup_error is not None:
            self.startup_error.session_owner=self
            raise self.startup_error

    def _process_batch(self):
        if not self.packets:self.packets.extend(self.pair.step())
        had_packets=bool(self.packets)
        while self.packets:
            self._apply_controls()
            if not self.packets or self.stopping.is_set() or (self.applied_state=="paused" and not self.steps and not self.initializing):break
            packet=self.packets.popleft()
            generation=self.engine.generation
            calibration=self.calibrations[packet.role]
            frame=calibration.transform(packet.frame)
            if packet.role=="system":
                # Camera 1: decode/preview only. YOLO runs once per accepted capture (predict).
                self._observe_camera1(packet,frame,generation)
                continue
            detector=self.detectors[packet.role]
            started=time.perf_counter()
            tracks=detector.track([frame],timestamp=packet.timestamp)[0]
            if self.stopping.is_set():break
            detector.update_velocities(tracks,calibration.homography.pixel_to_ground,timestamp=packet.timestamp)
            self._read_execution_provenance()
            elapsed=(time.perf_counter()-started)*1000
            predictor=getattr(getattr(detector,"model",None),"predictor",None)
            neural_device=str(getattr(predictor,"device","injected/unknown"))
            with self.lock:
                observations=[VehicleObservation(v.track_id,v.class_name,
                    v.velocity_mps if getattr(v,"velocity_valid",False) else None,
                    v.bottom_center,stable_id=getattr(v,"stable_id",False),
                    velocity_valid=getattr(v,"velocity_valid",False),
                    in_region=calibration.homography.is_in_approach_corridor(*v.bottom_center),
                    epoch=getattr(v,"epoch",detector.epoch)) for v in tracks]
                self.engine.observe_acquisition(frame,packet.timestamp,packet.sequence,observations,generation=generation)
                data=self.runtime[packet.role]
                data["frames"]+=1;data["preview_frames"]+=1;data["inference_total_ms"]+=elapsed;data["last_inference_ms"]=elapsed
                data["neural_device"]=neural_device
                data["last_source_timestamp"]=packet.source_timestamp;data["last_mapped_timestamp"]=packet.timestamp
                data["last_timestamp_origin"]=packet.timestamp_origin;data["last_sequence"]=packet.sequence
                self.acquisition_rates.record(time.monotonic(),packet.timestamp,elapsed)
            self._publish_frame(packet,frame,tracks)
        self._apply_controls()
        if self.source_mode=="replay" and not self.packets and self.pair.acquisition.ended and not self.acquisition_ended:
            self.acquisition_ended=True;self.engine.invalidate("acquisition_ended")
        if self.source_mode=="live":
            health=self.pair.health()
            age=health["acquisition"].get("age_sec")
            stale=age is None or age>self.policy.freshness_sec
            if stale and not self.live_stale:self.engine.invalidate("acquisition_stale")
            self.live_stale=stale
        if not self.storage_suspended and (self.engine.time-self.last_persisted>=5 or
                self.engine.retention_due()) and self.persist(wait=False) is not None:
            self.last_persisted=self.engine.time
        return had_packets

    def _publish_frame(self,packet,frame,tracks):
        with self.lock:
            # The transformed frame is owned by this packet and never modified after publication.
            # Calibration previews encode it only when requested, rather than twice per frame.
            self.clean_frames[packet.role]=frame
            self.clean_jpeg_cache.pop(packet.role,None)
        # Preview cadence never gates Camera 1's latest click candidate or Camera 2 measurements.
        now=time.monotonic()
        if packet.role=="system" and getattr(self,"applied_state",None)=="running":
            previous=self.preview_published_at.get(packet.role)
            if previous is not None and now-previous<.1:return
        visual=self._render(packet.role,frame,tracks)
        width=getattr(self,"preview_width",640)
        if visual.shape[1]>width:
            visual=cv2.resize(visual,(width,round(visual.shape[0]*width/visual.shape[1])),interpolation=cv2.INTER_LINEAR)
        ok,jpeg=cv2.imencode(".jpg",visual,[cv2.IMWRITE_JPEG_QUALITY,82])
        with self.lock:
            if ok:self.jpeg_frames[packet.role]=jpeg.tobytes()
            if hasattr(self,"preview_published_at"):self.preview_published_at[packet.role]=now
            self.frame_times[packet.role]=packet.timestamp
        # Routine commits never wait behind another commit; packets keep flowing.
        if not self.storage_suspended and self.engine.retention_due() and self.persist(wait=False) is not None:
            self.last_persisted=self.engine.time

    def _observe_camera1(self,packet,frame,generation):
        with self.lock:
            if not self.engine.observe_system_frame(packet.timestamp,generation=generation):return
            # The latest decoded frame (owned copy from transform) is the only click candidate.
            self.camera1_latest={"frame":frame,"sequence":packet.sequence,"timestamp":packet.timestamp,
                "source_timestamp":packet.source_timestamp,"timestamp_origin":packet.timestamp_origin,
                "generation":generation,"received_monotonic":time.monotonic()}
            data=self.runtime["system"];data["preview_frames"]+=1
            data["last_source_timestamp"]=packet.source_timestamp;data["last_mapped_timestamp"]=packet.timestamp
            data["last_timestamp_origin"]=packet.timestamp_origin;data["last_sequence"]=packet.sequence
        self._publish_frame(packet,frame,[])

    def _camera1_clock(self):
        """Run clock on the mapped-timestamp axis: host monotonic for live, replay time otherwise."""
        origin=getattr(self.pair,"origin",None) if self.source_mode=="live" else None
        return time.monotonic()-origin if origin is not None else self.engine.time

    def _camera1_frame_status(self):
        """Caller holds self.lock. Freshness of the latest Camera 1 preview frame."""
        latest=self.camera1_latest
        if latest is None:return None,"Camera 1 frame is unavailable; play or step the run first"
        age=self._camera1_clock()-latest["timestamp"]
        if not 0<=age<=self.policy.freshness_sec:return age,"Camera 1 frame is stale"
        if latest["generation"]!=self.engine.generation:return age,"Camera 1 frame predates a measurement discontinuity"
        return age,None

    def _analyze_camera1(self,frame,timestamp):
        """Runs on the prediction worker, outside every lock: ONE native still-image call."""
        detector=self.detectors["system"]
        detect=getattr(detector,"detect_once",None)
        started=time.perf_counter()
        # Injected components without a still-image API get exactly one single-frame call.
        try:
            vehicles=detect(frame,timestamp=timestamp) if detect else detector.track([frame],timestamp=timestamp)[0]
        finally:
            elapsed=(time.perf_counter()-started)*1000
            predictor=getattr(getattr(detector,"model",None),"predictor",None)
            device=str(getattr(predictor,"device","injected/unknown"))
            with self.lock:
                data=self.runtime["system"]
                data["frames"]+=1;data["inference_total_ms"]+=elapsed;data["last_inference_ms"]=elapsed
                data["neural_device"]=device
        for vehicle in vehicles:
            vehicle.stable_id=False;vehicle.velocity_valid=False;vehicle.obs_count=0
        raw=self.metrics.extract_single_frame_metrics(vehicles)
        return TrafficSnapshot.from_dict(raw,timestamp=timestamp),{"inference_calls":1,
            "inference_api":"detect_once" if detect else "track_single_frame_fallback",
            "latency_ms":round(elapsed,3),"detections":len(vehicles),"neural_device":device}

    def _render(self,role,frame,tracks):
        visual=frame.copy();calibration=self.calibrations[role]
        color=(220,210,90) if role=="system" else (70,200,235)
        calibration.homography.draw_corridor_overlay(visual)
        if role=="acquisition":
            line=np.array(calibration.stop_line,np.int32)
            cv2.line(visual,tuple(line[0]),tuple(line[1]),(240,240,240),2)
        for v in tracks:
            p=tuple(int(x) for x in v.bbox)
            cv2.rectangle(visual,p[:2],p[2:],color,2)
            label=f"#{v.track_id} {v.class_name}" if getattr(v,"stable_id",False) else v.class_name+" (ID unknown)"
            if role=="acquisition":
                state=self.engine.waits.active.get((getattr(v,"epoch",0),v.track_id))
                if state and "red_wait_sec" in state["row"]:label+=f" | red wait {state['row']['red_wait_sec']:.1f}s"
            cv2.putText(visual,label,(p[0],max(16,p[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.putText(visual,"CAMERA 1 / SYSTEM - YOLO on demand" if role=="system" else "CAMERA 2 / ACQUISITION",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        return visual

    def _control_status(self):
        return {"state":self.state,"requested_state":self.state,"applied_state":self.applied_state,
                "control_seq":self.control_seq,"applied_control_seq":self.applied_control_seq}

    def _apply_controls(self):
        """Only the measurement owner calls this, at packet boundaries."""
        while True:
            with self.lock:
                if not self.controls:return
                sequence,action=self.controls.popleft()
                if action!="play" or self.applied_state!="running":
                    self.acquisition_rates.reset_playback()
            if action=="pause":
                if self.source_mode=="live":self.engine.invalidate("live_pause")
                self.applied_state="paused";self.steps=0
            elif action=="play":
                if self.source_mode=="live" and self.applied_state=="paused":
                    # Clear BOTH the fetched local batch and the capture backlog before reset.
                    self.packets.clear();self.pair.discard_pending()
                    # Only Camera 2 tracks; the Camera 1 model may be inside a one-shot call.
                    self.detectors["acquisition"].reset_tracking()
                    self.engine.invalidate("live_resume")
                self.applied_state="running";self.steps=0
            elif action=="step":self.applied_state="paused";self.steps+=1
            elif action=="stop":self.applied_state="stopped";self.steps=0;self.packets.clear()
            with self.lock:self.applied_control_seq=sequence

    def _worker(self):
        try:
            self._process_batch()
            self.initializing=False
            self.persist()
            self.ready.set()
            while not self.stopping.is_set():
                self._apply_controls()
                with self.lock:state=self.applied_state;stepping=self.steps>0
                if state=="paused" and not stepping:
                    self.wake.wait(.1);self.wake.clear();continue
                started=time.perf_counter();self._process_batch()
                if self.pair.ended and not self.packets:
                    self.state=self.applied_state="ended";break
                if self.steps and not self.packets and not self.pair.has_backlog:
                    with self.lock:self.steps=max(0,self.steps-1)
                delay=0. if self.packets or self.pair.has_backlog or self.applied_state=="paused" else (1/self.pair.system.fps if self.source_mode=="replay" else .01)
                self.wake.wait(max(0,delay-(time.perf_counter()-started)));self.wake.clear()
        except Exception as error:
            clear_exception_frames(error)
            if self.initializing:self.startup_error=error
            self.error=safe_text(error);self.state=self.applied_state="error"
        finally:
            self.ready.set()
            self._cleanup()

    def _ensure_cleanup(self):
        # Preserve the acquisition error/session owner if scheduling fails. A later
        # public stop can retry without releasing beneath an active native worker.
        with self.cleanup_schedule_lock:
            if self.resources_released:return
            if self.thread is not None and self.thread.is_alive():return
            try:
                self.thread=threading.Thread(target=self._cleanup,daemon=True,name="Observation-cleanup-"+self.run_id)
                self.thread.start()
                self.cleanup_schedule_error=None
            except Exception as error:
                self.thread=None;self.cleanup_schedule_error=safe_text(error)

    def _cleanup(self):
        """Keep the owning session alive until native work exits; never cancel it.
        Every independent release is attempted even if another raised; failed releases are retried."""
        with self.cleanup_lock:
            if self.finalized:return
            sealing=self._attempt(self._apply_controls,self._seal)
            self.terminal_state="releasing";delay=.02
            while True:
                # A close request is not release evidence for blocked reads or predictions. Model owners
                # wait only for native tracking/prediction calls (tracking has exited here), not captures.
                steps=[] if self._captures_released() else [self.pair.close]
                if self._predictions_released() and (self.detectors or self.decision is not None):steps.append(self._release_models)
                errors=self._attempt(*steps)
                if errors:self.release_error="; ".join(errors)
                elif self._captures_released() and self._predictions_released() and not self.detectors and self.decision is None:break
                # Paced; stop() wakes a retry, a persistent failure backs off.
                self.wake.wait(delay if errors else .02);self.wake.clear()
                delay=min(.5,delay*2) if errors else .02
            self.release_error="; ".join(sealing) or None;self._resources_released=True
            if hasattr(self,"store") and not sealing:
                self.terminal_state="finalizing"
                self.finalized=self.persist(final=True)
                self.terminal_state="finalized" if self.finalized else "unclean"
            else:self.terminal_state="unclean"

    @staticmethod
    def _attempt(*steps):
        errors=[]
        for step in steps:
            try:step()
            except Exception as error:errors.append(safe_text(error))
        return errors

    def _seal(self):
        if hasattr(self,"engine"):
            self.engine.close("recording_end" if self.state=="ended" else "session_error" if self.state=="error" else "operator_stop")

    def _release_models(self):
        # Keep local/bound method references inside this scope, which exits before
        # resources_released is published (even if final persistence blocks).
        errors=[]
        for role,detector in list(self.detectors.items()):
            close=getattr(detector,"close",None)
            try:
                if close:close()
            except Exception as error:errors.append(safe_text(error));continue  # unconfirmed: retained for retry
            del self.detectors[role]
        if hasattr(self,"engine"):self.engine.predictions.evaluate=None
        self.decision=None
        if errors:raise RuntimeError("; ".join(errors))

    def _captures_released(self):
        return self.pair is None or getattr(self.pair,"resources_released",True)

    def _predictions_released(self):
        return not hasattr(self,"engine") or self.engine.predictions.resources_released

    @property
    def resources_released(self):
        return self._resources_released

    def lifecycle(self):
        with self.lock:
            return {"stop_requested":self.stopping.is_set(),
                    "measurement_sealed":hasattr(self,"engine") and self.engine.closed,
                    "capture_release_pending":self.pair is not None and not getattr(self.pair,"resources_released",True),
                    "prediction_release_pending":hasattr(self,"engine") and not self.engine.predictions.resources_released,
                    "resources_released":self.resources_released,"finalized":self.finalized,
                    "terminal_state":self.terminal_state,"release_error":self.release_error,
                    "cleanup_schedule_error":self.cleanup_schedule_error,
                    "last_persisted_revision":self.last_persisted_revision,
                    "durable_revision":self.durable_revision,"durable_generation":self.durable_generation,
                    "durable_finalized":self.finalized,"storage_protocol":"typed-archive-v1",
                    "storage_suspended":self.storage_suspended,"pending_commit":self.pending_commit is not None}


    def control(self,action):
        if action not in ("pause","play","step","stop"):raise ValueError("Unknown control action")
        with self.lock:
            if self.state in ("stopped","ended","error"):
                if action=="stop":return self._control_status()
                raise ValueError("Run has ended; start a new observation")
            if action=="step" and self.source_mode!="replay":raise ValueError("Step is available only for file replay")
            self.control_seq+=1;self.controls.append((self.control_seq,action))
            self.state={"pause":"paused","play":"running","step":"paused","stop":"stopped"}[action]
            if action=="stop":self.stopping.set()
            result=self._control_status()
        self.wake.set()
        return result

    def predict(self,request_id):
        # Historical retries are explicit reads, outside engine/control locks.
        with self.engine.predictions.lock:
            existing=deepcopy(self.engine.predictions.records.get(request_id))
        if existing is not None:
            if self.pending_commit is not None and self.storage_error:raise OSError(self.storage_error)
            return existing
        # Only a retry of a capture this run already archived reads history; a new
        # capture is admitted with its inputs/time frozen without an archive scan.
        if self.engine.retention is not None and self.engine.retention.requested(request_id):
            return self.store.lookup(request_id)
        with self.lock:
            if self.storage_suspended:raise ValueError("Evidence storage suspended; retry storage and start a new run")
            if self.stopping.is_set() or self.state in ("stopped","ended","error"):
                raise ValueError("Run has ended or is stopping")
            timestamp=None
            if self.source_mode=="live":
                age=self.pair.health()["system"].get("age_sec")
                if age is None or age>self.policy.freshness_sec:raise ValueError("System camera is stale")
                timestamp=time.monotonic()-self.pair.origin
            _,problem=self._camera1_frame_status()
            if problem:raise ValueError(problem)
            latest=self.camera1_latest;frame=latest["frame"].copy()
            evidence={key:latest[key] for key in ("sequence","timestamp","source_timestamp","timestamp_origin","generation")}
            evidence.update(shape=list(frame.shape),sha256=hashlib.sha256(frame.tobytes()).hexdigest())
            record=self.engine.predict(request_id,timestamp,provenance={"execution":self.execution_provenance(),
                "clock_provenance":self.clock_provenance(),"sources":{role:c.payload["source"] for role,c in self.calibrations.items()}},
                frame=evidence,analyze=partial(self._analyze_camera1,frame,latest["timestamp"]))
        if not self.persist():raise OSError(self.storage_error)
        return record

    def cancel(self,request_id):
        if self.storage_suspended:raise OSError(self.storage_error or "Evidence storage suspended")
        try:result=self.engine.predictions.cancel(request_id)
        except KeyError:
            if self.engine.retention is not None and self.engine.retention.requested(request_id):
                raise PredictionConflict("Archived prediction is immutable") from None
            raise
        if not self.persist():raise OSError(self.storage_error)
        return result

    def stop(self,timeout=5.):
        self.control("stop");self.wake.set()  # also wakes a cleanup retry after a failed release
        self._ensure_cleanup()
        thread=self.thread
        if thread is not None and threading.current_thread()!=thread and thread.ident is not None:
            thread.join(timeout=max(0,timeout))
            if thread.is_alive():raise RuntimeError("Observation is stopping; wait for resource release")
        if not self.resources_released:raise RuntimeError("Observation is stopping; wait for resource release")

    def telemetry(self,offset=0,limit=100,evidence=False,*,_snapshot=None):
        snapshot=_snapshot if _snapshot is not None else self.engine.snapshot(offset,limit,evidence) if hasattr(self,"engine") else {
            "schema_version":2,"run_id":self.run_id,"operation":"observation","metrics":None,"closed":False}
        with self.lock:
            elapsed=max(.001,time.monotonic()-self.created_monotonic)
            runtime=deepcopy(self.runtime)
            health=self.pair.health() if self.pair is not None else {}
            for role,stats in runtime.items():
                stats["mean_inference_ms"]=round(stats["inference_total_ms"]/max(1,stats["frames"]),2)
                stats["processed_fps"]=round(stats["frames"]/elapsed,2)
                if role=="acquisition":
                    stats.update(self.acquisition_rates.snapshot(time.monotonic(),self.applied_state=="running" and not self.stopping.is_set()))
                    source=getattr(self.pair,"acquisition",None)
                    stats["source_fps"]=getattr(source,"fps",None) if self.source_mode=="replay" else None
                if self.source_mode=="live" and role in self.frame_times:
                    age=max(0,time.monotonic()-self.pair.origin-self.frame_times[role]-(self.policy.offset_sec if role=="acquisition" else 0))
                    health[role]["processed_frame_age_sec"]=age
                    if age>self.policy.freshness_sec and health[role]["state"]=="ready":health[role]["state"]="lagging"
            snapshot.update(**self._control_status(),lifecycle=self.lifecycle(),error=self.error,storage_error=self.storage_error,
                source_mode=self.source_mode,shared_source_test=self.shared_test,
                camera_calibrations={role:deepcopy(c.payload) for role,c in self.calibrations.items()},
                policy=asdict(self.policy),
                cameras=health,runtime=runtime,frame_times=dict(self.frame_times),
                detector_selection=deepcopy(getattr(self,"model_selection",None)))
            # Camera 1 metrics are the last analysed still image, never a live measurement.
            age,problem=self._camera1_frame_status() if hasattr(self,"engine") else (None,"Camera 1 is unavailable")
            last=snapshot.pop("camera1_last_analysis",None)
            now=self._camera1_clock() if hasattr(self,"engine") else 0.
            snapshot["camera1"]={"inference_mode":"on_demand_single_frame",
                "frame_available":self.camera1_latest is not None,"frame_fresh":problem is None,
                "frame_age_sec":age,"frame_problem":problem,
                "frame_sequence":self.camera1_latest["sequence"] if self.camera1_latest else None,
                "last_analysis":None if last is None else {"request_id":last["request_id"],
                    "frame_sequence":last["frame"].get("sequence"),"frame_timestamp":last["frame"].get("timestamp"),
                    "completed_at":last["completed_at"],"age_sec":max(0.,now-last["frame"]["timestamp"])}}
            snapshot["snapshot_fresh"]=problem is None
        snapshot.update(execution=self.execution_provenance(),clock_provenance=self.clock_provenance())
        return public_data(snapshot)

    def execution_provenance(self):
        # Sample models only on their owner thread, never concurrently with release.
        if getattr(self,"last_execution",None) is None:self._read_execution_provenance()
        return deepcopy(self.last_execution)

    def _read_execution_provenance(self):
        def artifact(component):
            path=getattr(component,"model_path",None)
            if not path:return {"path":None,"sha256":None}
            candidate=Path(path)
            return {"path":public_source(path),"sha256":file_digest(str(candidate),candidate.stat().st_mtime_ns) if candidate.is_file() else None}
        detectors={}
        for role,detector in self.detectors.items():
            predictor=getattr(getattr(detector,"model",None),"predictor",None)
            args=getattr(predictor,"args",None)
            settings={key:getattr(args,key,None) for key in ("conf","iou","imgsz","half","tracker")}
            if getattr(predictor,"imgsz",None) is not None:settings["imgsz"]=predictor.imgsz
            backend=getattr(predictor,"model",None)
            device=getattr(backend,"device",getattr(predictor,"device",None))
            detectors[role]={"component":type(detector).__name__,
                "requested_device":getattr(detector,"device",None),
                "effective_device":str(device) if device is not None else None,
                "model":artifact(detector),
                "effective_precision":("fp16" if backend.fp16 else "fp32") if hasattr(backend,"fp16") else None,
                "precision_scope":"input_tensor",
                "engine_build_precision":self.model_selection.get("engine_build_precision") if getattr(detector,"is_tensorrt",False) else None,
                "requested_settings":{key:getattr(detector,key,None) for key in ("conf_threshold","iou_threshold","imgsz","max_gap_sec")},
                "effective_settings":settings}
        anfis=getattr(self.decision,"anfis",None)
        session=getattr(anfis,"session",None)
        providers=session.get_providers() if callable(getattr(session,"get_providers",None)) else ([anfis.active_provider] if getattr(anfis,"active_provider",None) else [])
        result=public_data({"detectors":detectors,"anfis":{"component":type(anfis).__name__,
            "providers":providers,"requested_prefer_cuda":False,"model":artifact(anfis),
            "min_green_sec":getattr(anfis,"min_green",None),"max_green_sec":getattr(anfis,"max_green",None),
            "model_domain":"unknown"}})
        self.last_execution=deepcopy(result)
        return result

    def clock_provenance(self):
        health=self.pair.health() if self.pair is not None else {}
        with self.lock:processed=deepcopy(self.runtime)
        result={}
        for role in ("system","acquisition"):
            values=health.get(role,{})
            result[role]={"timestamp_origin":values.get("timestamp_origin","unknown"),
                "timestamp_fallback_count":values.get("timestamp_fallback_count"),
                "fallback_count_scope":"read packets including replay prefetch",
                "last_processed":{"origin":processed.get(role,{}).get("last_timestamp_origin","unknown"),
                    "source_timestamp":processed.get(role,{}).get("last_source_timestamp"),
                    "mapped_timestamp":processed.get(role,{}).get("last_mapped_timestamp"),
                    "sequence":processed.get(role,{}).get("last_sequence")},
                "offset_sec":self.policy.offset_sec if role=="acquisition" else 0.,
                "mapping":"source_timestamp - offset_sec" if self.source_mode=="replay" else "host_monotonic_receive - pair_origin - offset_sec",
                "pair_origin_monotonic":getattr(self.pair,"origin",None) if self.source_mode=="live" else None}
        return public_data(result)

    def jpeg(self,role,clean=False):
        with self.lock:
            if not clean:return self.jpeg_frames.get(role)
            frame=self.clean_frames.get(role)
            cached=self.clean_jpeg_cache.get(role)
            if cached is not None or frame is None:return cached
        ok,jpeg=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:return None
        result=jpeg.tobytes()
        with self.lock:
            if self.clean_frames.get(role) is frame:self.clean_jpeg_cache[role]=result
        return result

    def persist(self,final=False,wait=True):
        # Disk IO uses only persist_lock/store.lock. Telemetry, measurement admission
        # and native cleanup never require either lock. One retained payload survives
        # every failure. wait=False returns None if another commit is in flight.
        if not self.persist_lock.acquire(blocking=wait):return None
        try:
            try:
                if self.pending_commit is None:
                    with self.engine.lock:
                        if self.last_persisted_revision==self.engine.revision+self.engine.predictions.view()["revision"] and not final:
                            return True
                        state,token=self.engine.prepare_commit()
                    state=self.telemetry(_snapshot=state)
                    self.commit_sequence+=1
                    state["commit_sequence"]=self.commit_sequence
                    if final:
                        state["lifecycle"].update(finalization_requested=True,finalized=True,
                            durable_finalized=True,terminal_state="finalized")
                    self.pending_commit=(state,token,final)
                state,token,pending_final=self.pending_commit
                acknowledgment=self.store.persist(state)
                # Derived lookups are written outside the engine lock, before eviction.
                if token is not None:self.engine.retention.index_token(token)
                self.engine.acknowledge(token)
                self.last_persisted_revision=self.durable_revision=state["snapshot_revision"]
                self.durable_generation=acknowledgment["generation"]
                self.pending_commit=None;self.storage_error=None
                if pending_final:self.finalized=True
                elif final:return self.persist(final=True)
                return True
            except Exception as error:
                self.storage_error=safe_text(error);self.storage_suspended=True
                # Sealing rejects asynchronous completion and new measurements.
                # Its bounded terminal delta is committed AFTER the retained
                # pre-failure payload has been durably acknowledged.
                self.engine.close("storage_error")
                return False
        finally:self.persist_lock.release()

    def retry_storage(self):
        if not self.persist():return False
        if self.storage_suspended:
            self.control("stop");self.wake.set()
            if self.resources_released:
                if not self.finalized and not self.persist(final=True):return False
                self.terminal_state="finalized" if self.finalized else "unclean"
            else:self._ensure_cleanup()
        return True

    def export(self):
        if not self.persist():raise OSError(self.storage_error or "Evidence commit failed")
        return self.store.export()


class ObservationManager:
    def __init__(self,storage_root=None,profiles_root=None,session_factory=None,preflight=None,shutdown_timeout=5.):
        self.storage_root=Path(storage_root or ROOT/"data/experiments")
        self.profiles_root=Path(profiles_root or ROOT/"data/observation_profiles")
        self.session_factory=session_factory or ObservationSession
        self.current=None;self.lock=threading.RLock();self.starting=False
        self.preflight=preflight;self.shutdown_timeout=shutdown_timeout;self.last_start=None
        self.closing=False;self.start_complete=threading.Event();self.start_complete.set()

    def _preflight(self,payload):
        if self.preflight is not None:return self.preflight(payload,self.storage_root)
        factory=self.session_factory
        keywords={}
        if isinstance(factory,partial):keywords=factory.keywords or {};factory=factory.func
        hook=getattr(factory,"preflight",ObservationSession.preflight)
        return hook(payload,self.storage_root,**{k:v for k,v in keywords.items() if k in ("detector_factory","anfis_factory")})

    def start(self,payload,before_handoff=None):
        with self.lock:
            if self.closing:raise ObservationConflict("Observation manager is shutting down")
            if self.starting:raise ObservationConflict("An observation start is already in progress")
            self.starting=True;self.start_complete.clear()
        try:
            with runtime_ownership.reserve("observation"):
                # The reservation is held through static validation, handoff and acquisition.
                self._preflight(payload)
                for recovered in self.runs():
                    if recovered.get("storage_protocol")=="typed-archive-v1" and not recovered.get("durable_finalized") and (
                            self.current is None or recovered["run_id"]!=self.current.run_id):
                        self._recover_interrupted(recovered["run_id"])
                if self.closing:raise ObservationConflict("Shutdown requested during static preflight")
                previous=self.current
                kind,owner=runtime_ownership.current_owner()
                if kind=="observation" and owner is not None and owner is not previous and not owner.resources_released:
                    raise ObservationConflict("Another observation owner still owns resources")
                self.last_start={"previous_run_id":previous.run_id if previous else None,
                                 "previous_ended":False,"replacement_state":"reserved"}
                if previous:
                    try:previous.stop(timeout=self.shutdown_timeout)
                    except RuntimeError as error:raise ObservationConflict(safe_text(error)) from error
                    if not previous.resources_released:raise ObservationConflict("Previous observation still owns resources")
                    if getattr(previous,"storage_suspended",False) and (not previous.retry_storage() or not previous.finalized):
                        raise ObservationConflict("Previous run evidence is not durably finalized; retry storage")
                    self.last_start["previous_ended"]=True
                if before_handoff is None:
                    from src.ui.session_manager import manager as legacy_manager
                    before_handoff=legacy_manager.stop_session
                if before_handoff:
                    try:before_handoff()
                    except RuntimeError as error:raise ObservationConflict(safe_text(error)) from error
                if kind=="legacy" and owner is not None and not owner.resources_released:
                    raise ObservationConflict("Legacy runtime owner still owns resources after handoff")
                try:session=self.session_factory(payload,self.storage_root)
                except Exception as error:
                    failed=getattr(error,"session_owner",None)
                    if failed is not None:
                        self.current=failed;runtime_ownership.claim("observation",failed)
                    self.last_start["replacement_state"]="failed"
                    message="Previous run " + (previous.run_id+" ended; " if previous else "absent; ") + "replacement failed: " + safe_text(error)
                    raise ReplacementFailed(message) from error
                self.current=session;runtime_ownership.claim("observation",session)
                if self.closing:
                    session.control("stop")
                    raise ObservationConflict("Shutdown requested during observation startup")
                self.last_start.update(replacement_state="started",run_id=session.run_id)
                return session
        finally:
            with self.lock:self.starting=False;self.start_complete.set()

    def shutdown(self,timeout=None):
        budget=self.shutdown_timeout if timeout is None else max(0,timeout)
        deadline=time.monotonic()+budget
        with self.lock:self.closing=True
        self.start_complete.wait(max(0,deadline-time.monotonic()))
        previews_released=runtime_ownership.stop_previews(max(0,deadline-time.monotonic()))
        session=self.current
        if session is not None:
            try:session.stop(timeout=max(0,deadline-time.monotonic()))
            except RuntimeError:pass
            result={**session.lifecycle(),"start_pending":self.starting}
            if self.starting:result.update(resources_released=False,finalized=False,terminal_state="startup_pending")
            result["preview_release_pending"]=not previews_released
            if not previews_released:result.update(resources_released=False,finalized=False)
            result["shutdown_outcome"]="clean" if result["finalized"] else "unclean" if result["resources_released"] else "abandoned_pending_resources"
            return result
        return {"resources_released":not self.starting and previews_released,"finalized":False,"start_pending":self.starting,
                "preview_release_pending":not previews_released,
                "terminal_state":"startup_pending" if self.starting else "idle",
                "shutdown_outcome":"abandoned_pending_resources" if self.starting or not previews_released else "idle"}

    def require(self,run_id=None):
        if self.current is None:raise ValueError("No observation run is active")
        if run_id is not None and run_id!=self.current.run_id:raise ValueError("Run identity changed; refresh the dashboard")
        return self.current

    def save_calibration(self,payload):
        calibration=CameraCalibration(payload)
        self.profiles_root.mkdir(parents=True,exist_ok=True)
        name=re.sub(r"[^A-Za-z0-9_-]","_",str(calibration.payload.get("name","profile")))[:60] or "profile"
        profile_id=f"{name}_{calibration.role}_{calibration.revision}.json"
        path=self.profiles_root/profile_id
        path.write_text(json.dumps(calibration.payload,indent=2,allow_nan=False))
        return {"profile_id":profile_id,"revision":calibration.revision,"calibration":calibration.payload,"scale":calibration.scale_status}

    def evidence_store(self,run_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]+",run_id):raise ValueError("Invalid run identity")
        if self.current is not None and self.current.run_id==run_id:return self.current.store
        return ExperimentStore.open(self.storage_root/run_id)

    def runs(self):
        result=[]
        for folder in sorted(self.storage_root.glob("*/manifest.json")):
            try:
                store=ExperimentStore.open(folder.parent);status=store.recovery_status()
                if self.current is not None and folder.parent.name==self.current.run_id:
                    status.update(status=self.current.terminal_state,**self.current.lifecycle())
                result.append(status)
            except (OSError,ValueError,KeyError) as error:
                # Keep a readable protocol marker even if committed bytes are
                # missing. New admission must not bypass interrupted evidence.
                try:protocol=json.loads(folder.read_text()).get("storage_protocol")
                except (OSError,ValueError):protocol=None
                result.append({"run_id":folder.parent.name,"status":"unavailable","error":safe_text(error),
                               "storage_protocol":protocol})
        return result

    def recover_interrupted(self,run_id):
        with runtime_ownership.reserve("evidence recovery"):
            return self._recover_interrupted(run_id)

    def _recover_interrupted(self,run_id):
        _,owner=runtime_ownership.current_owner()
        if owner is not None and owner is not self.current and getattr(owner,"run_id",None)==run_id:
            thread=getattr(owner,"thread",None)
            if not owner.resources_released or (thread is not None and thread.is_alive()):
                raise ObservationConflict("Another observation owner still owns this run")
        if self.current is not None and self.current.run_id==run_id:
            if not self.current.resources_released and not self.current.storage_suspended:
                raise ObservationConflict("Run still owns resources")
            if not self.current.retry_storage():raise OSError(self.current.storage_error)
            if not self.current.resources_released:return self.current.lifecycle()
            if not self.current.finalized and not self.current.persist(final=True):raise OSError(self.current.storage_error)
            return self.current.store.recovery_status()
        return self.evidence_store(run_id).recover()

    def profiles(self):
        return [{"profile_id":p.name,"revision":p.stem.rsplit("_",1)[-1],"calibration":public_data(json.loads(p.read_text()))} for p in sorted(self.profiles_root.glob("*.json"))]

    def _owned_preview(self,role,source,rectification_path):
        """Cached frame of the unreleased local or process-wide owner of this source, else None."""
        identity=source_identity(source);_,registered=runtime_ownership.current_owner()
        for owner in (self.current,registered):
            if owner is None or owner.resources_released:continue
            # Legacy JPEGs contain annotations and have no verified calibration binding.
            legacy_source=getattr(owner,"video_path",None)
            if legacy_source is not None and identity==source_identity(legacy_source):
                raise ObservationConflict("Legacy replay owns this source; stop it before calibration preview")
            calibrations=getattr(owner,"calibrations",None)
            if not calibrations:continue
            if identity not in {source_identity(c.payload["source"]) for c in calibrations.values()}:continue
            if identity!=source_identity(calibrations[role].payload["source"]):
                raise ObservationConflict("Source is owned by the other observation role")
            if rectification_path!=calibrations[role].payload.get("rectification_path"):
                raise ValueError("Stop the run before changing lens calibration")
            result=owner.jpeg(role,True)
            if result:return result
            raise ValueError("This live camera has not produced a valid preview yet")
        return None

    def preview(self,role,source,rectification_path=None,source_mode="replay"):
        source=resolve_source(source)
        if role not in ("system","acquisition"):raise ValueError("Unknown camera role")
        with self.lock:
            if self.closing:raise ObservationConflict("Observation manager is shutting down")
        result=self._owned_preview(role,source,rectification_path)
        if result is not None:return result
        # A new capture holds a shared lease from open until release; starts conflict meanwhile.
        with runtime_ownership.lease(source_identity(source)) as pending:
            # Shutdown may have begun between the first lookup and lease registration.
            # It either observes this owner, or this check prevents any acquisition.
            with self.lock:
                if self.closing:raise ObservationConflict("Observation manager is shutting down")
            result=self._owned_preview(role,source,rectification_path)
            if result is not None:return result
            pending.cap=cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
            if not pending.cap.isOpened():raise ValueError("Cannot open camera/video preview")
            ok,frame=pending.cap.read()
            if not ok:raise ValueError("No frame available from this source")
        if rectification_path:
            path=Path(rectification_path)
            if not path.is_absolute():path=ROOT/path
            rectifier=LensRectifier(str(path))
            if tuple(frame.shape[1::-1])!=rectifier.dim:raise ValueError("Lens calibration dimensions differ from source")
            frame=rectifier.rectify(frame)
        ok,result=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise ValueError("Could not encode preview")
        return result.tobytes()
