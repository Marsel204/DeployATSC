"""One owner for two observation cameras; no controller imports or hardware calls."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import threading
import time
import uuid

import cv2
import numpy as np

from src.analytics.traffic_metrics import TrafficMetricsExtractor
from src.common.types import TrafficSnapshot
from src.control.anfis_inference import ANFISInferenceEngine
from src.control.decision_engine import ATSCDecisionEngine
from src.experiment.calibration import CameraCalibration, legacy_system_payload
from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.sources import ReplayPair, LivePair
from src.perception.detector_tracker import DetectorTracker
from src.perception.lens_rectifier import LensRectifier

ROOT=Path(__file__).resolve().parents[2]


@lru_cache(maxsize=16)
def file_digest(path, modified):
    h=hashlib.sha256()
    with open(path,"rb") as source:
        for block in iter(lambda:source.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def source_identity(value):
    value=str(value).strip()
    if value.isdigit():value="/dev/video"+value
    return str(Path(value).resolve()) if "://" not in value else value


class ObservationSession:
    def __init__(self,payload,storage_root,detector_factory=None,anfis_factory=None):
        self.policy=RunPolicy(**payload.get("policy",{}))
        self.source_mode=payload.get("source_mode","replay")
        if self.source_mode not in ("replay","live"):raise ValueError("Source mode must be replay or live")
        self.calibrations={role:CameraCalibration(payload[role]) for role in ("system","acquisition")}
        if any(c.role!=role for role,c in self.calibrations.items()):raise ValueError("Calibration camera roles are mismatched")
        self.shared_test=bool(payload.get("shared_test",False))
        sources={role:c.payload["source"] for role,c in self.calibrations.items()}
        shared=source_identity(sources["system"])==source_identity(sources["acquisition"])
        if shared and (not self.shared_test or self.source_mode!="replay"):
            raise ValueError("A shared source requires explicit shared-video test mode; live inputs must differ")
        self.shared_test=shared
        self.run_id="run_"+uuid.uuid4().hex[:16]
        self.lock=threading.RLock();self.stopping=threading.Event();self.wake=threading.Event()
        self.state="paused" if payload.get("step_mode",False) else "running"
        self.error=None;self.storage_error=None;self.jpeg_frames={};self.clean_frames={}
        self.frame_times={};self.steps=0;self.created_monotonic=time.monotonic()
        self.runtime={role:{"frames":0,"inference_total_ms":0.,"last_inference_ms":0.} for role in self.calibrations}
        self.last_persisted=0.;self.acquisition_ended=False;self.live_stale=False
        factory=ReplayPair if self.source_mode=="replay" else LivePair
        self.pair=factory(sources["system"],sources["acquisition"],self.policy.offset_sec,
                          sizes={role:c.size for role,c in self.calibrations.items()})
        try:
            if self.source_mode=="replay":
                for role,data in self.pair.health().items():
                    if tuple(data["image_size"])!=self.calibrations[role].size:
                        raise ValueError(f"{role} source dimensions do not match calibration")
            detector_factory=detector_factory or DetectorTracker
            anfis_factory=anfis_factory or ANFISInferenceEngine
            model=ROOT/"models/Final.pt";anfis=ROOT/"models/anfis_sugeno.onnx"
            self.detectors={role:detector_factory(str(model),conf_threshold=.28,imgsz=640) for role in self.calibrations}
            decision=ATSCDecisionEngine(anfis_factory(str(anfis),prefer_cuda=False))
            acquisition=self.calibrations["acquisition"]
            self.engine=ObservationEngine(self.run_id,self.policy,acquisition.roi,acquisition.side,decision.evaluate,acquisition.lamp_rois)
            if self.source_mode=="live":
                self.engine.predictions.clock=lambda:time.monotonic()-self.pair.origin
            system=self.calibrations["system"]
            self.metrics=TrafficMetricsExtractor.from_config(system.legacy_metrics_config(),system.homography)
            versions={}
            for package in ("torch","ultralytics","onnxruntime","opencv-python"):
                try:versions[package]=importlib.metadata.version(package)
                except importlib.metadata.PackageNotFoundError:versions[package]=None
            manifest={"operation":"observation","movement_id":self.policy.movement_id,
                "source_mode":self.source_mode,"shared_source_test":shared,"policy":asdict(self.policy),
                "sources":sources,"calibrations":{role:{"revision":c.revision,"payload":c.payload} for role,c in self.calibrations.items()},
                "model_hashes":{p.name:file_digest(str(p),p.stat().st_mtime_ns) for p in (model,anfis)},
                "package_versions":versions}
            manifest["inference_components"]={"detector":getattr(detector_factory,"__name__",type(detector_factory).__name__),
                "anfis":getattr(anfis_factory,"__name__",type(anfis_factory).__name__)}
            manifest["source_hashes"]={str(path.relative_to(ROOT)):file_digest(str(path),path.stat().st_mtime_ns)
                for path in sorted((ROOT/"src").rglob("*.py"))}
            self.store=ExperimentStore(storage_root,self.run_id,manifest)
            self._process_batch()
            self.persist()
        except Exception:
            self.pair.close()
            if hasattr(self,"engine"):self.engine.close("startup_failure")
            raise
        self.thread=threading.Thread(target=self._worker,daemon=True,name="Observation-"+self.run_id)
        self.thread.start()

    def _process_batch(self):
        packets=self.pair.step()
        for packet in packets:
            if self.stopping.is_set():break
            calibration=self.calibrations[packet.role]
            frame=calibration.transform(packet.frame)
            detector=self.detectors[packet.role]
            started=time.perf_counter()
            tracks=detector.track([frame],timestamp=packet.timestamp)[0]
            detector.update_velocities(tracks,calibration.homography.pixel_to_ground,timestamp=packet.timestamp)
            elapsed=(time.perf_counter()-started)*1000
            predictor=getattr(getattr(detector,"model",None),"predictor",None)
            neural_device=str(getattr(predictor,"device","injected/unknown"))
            if packet.role=="system":
                raw=self.metrics.extract_approach_metrics(tracks)
                snapshot=TrafficSnapshot.from_dict(raw,timestamp=packet.timestamp)
                self.engine.update_system(snapshot)
            else:
                observations=[VehicleObservation(v.track_id,v.class_name,
                    v.velocity_mps if getattr(v,"velocity_valid",False) else None,
                    v.bottom_center,stable_id=getattr(v,"stable_id",False),
                    velocity_valid=getattr(v,"velocity_valid",False),
                    in_region=calibration.homography.is_in_approach_corridor(*v.bottom_center),
                    epoch=getattr(v,"epoch",detector.epoch)) for v in tracks]
                self.engine.observe_acquisition(frame,packet.timestamp,packet.sequence,observations)
            visual=self._render(packet.role,frame,tracks)
            ok,jpeg=cv2.imencode(".jpg",visual,[cv2.IMWRITE_JPEG_QUALITY,82])
            ok_clean,clean=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
            with self.lock:
                if ok:self.jpeg_frames[packet.role]=jpeg.tobytes()
                if ok_clean:self.clean_frames[packet.role]=clean.tobytes()
                self.frame_times[packet.role]=packet.timestamp
                data=self.runtime[packet.role]
                data["frames"]+=1;data["inference_total_ms"]+=elapsed;data["last_inference_ms"]=elapsed
                data["neural_device"]=neural_device
        if self.source_mode=="replay" and self.pair.acquisition.ended and not self.acquisition_ended:
            self.acquisition_ended=True;self.engine.invalidate("acquisition_ended")
        if self.source_mode=="live":
            health=self.pair.health()
            age=health["acquisition"].get("age_sec")
            stale=age is None or age>self.policy.freshness_sec
            if stale and not self.live_stale:self.engine.invalidate("acquisition_stale")
            self.live_stale=stale
        if self.engine.time-self.last_persisted>=5:
            self.persist();self.last_persisted=self.engine.time
        return bool(packets)

    def _render(self,role,frame,tracks):
        visual=frame.copy();calibration=self.calibrations[role]
        color=(220,210,90) if role=="system" else (70,200,235)
        calibration.homography.draw_corridor_overlay(visual)
        if role=="acquisition":
            line=np.array(calibration.stop_line,np.int32)
            cv2.line(visual,tuple(line[0]),tuple(line[1]),(240,240,240),2)
            roi=calibration.roi
            cv2.rectangle(visual,(roi["xmin"],roi["ymin"]),(roi["xmax"],roi["ymax"]),color,2)
            cv2.putText(visual,"SIGNAL "+str(self.engine.phase.phase),(roi["xmin"],max(20,roi["ymin"]-10)),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        for v in tracks:
            p=tuple(int(x) for x in v.bbox)
            cv2.rectangle(visual,p[:2],p[2:],color,2)
            label=f"#{v.track_id} {v.class_name}" if getattr(v,"stable_id",False) else v.class_name+" (ID unknown)"
            if role=="acquisition":
                state=self.engine.waits.active.get((getattr(v,"epoch",0),v.track_id))
                if state:label+=f" | red wait {state['row']['red_wait_sec']:.1f}s"
            cv2.putText(visual,label,(p[0],max(16,p[1]-6)),cv2.FONT_HERSHEY_SIMPLEX,.45,color,1)
        cv2.putText(visual,"CAMERA 1 / SYSTEM" if role=="system" else "CAMERA 2 / ACQUISITION",(12,24),cv2.FONT_HERSHEY_SIMPLEX,.6,color,2)
        return visual

    def _worker(self):
        try:
            while not self.stopping.is_set():
                with self.lock:state=self.state;stepping=self.steps>0
                if state=="paused" and not stepping:
                    self.wake.wait(.1);self.wake.clear();continue
                started=time.perf_counter();self._process_batch()
                if self.pair.ended:
                    self.state="ended";break
                if stepping and not self.pair.has_backlog:
                    with self.lock:self.steps=max(0,self.steps-1)
                delay=0. if self.pair.has_backlog else (1/self.pair.system.fps if self.source_mode=="replay" else .01)
                self.stopping.wait(max(0,delay-(time.perf_counter()-started)))
        except Exception as error:
            self.error=str(error);self.state="error"
        finally:
            self.pair.close()
            self.engine.close("recording_end" if self.state=="ended" else "session_error" if self.state=="error" else "operator_stop")
            self.persist()

    def control(self,action):
        if action=="stop":self.stop();return {"state":self.state}
        with self.lock:
            if self.state in ("stopped","ended","error"):raise ValueError("Run has ended; start a new observation")
            if action=="pause":
                self.state="paused"
                if self.source_mode=="live":self.engine.invalidate("live_pause")
            elif action=="play":
                if self.source_mode=="live" and self.state=="paused":
                    self.pair.discard_pending()
                    for detector in self.detectors.values():detector.reset_tracking()
                    self.engine.invalidate("live_resume")
                self.state="running";self.steps=0
            elif action=="step":
                if self.source_mode!="replay":raise ValueError("Step is available only for file replay")
                self.state="paused";self.steps+=1
            else:raise ValueError("Unknown control action")
        self.wake.set();return {"state":self.state}

    def predict(self,request_id):
        timestamp=None
        if self.source_mode=="live":
            age=self.pair.health()["system"].get("age_sec")
            if age is None or age>self.policy.freshness_sec:raise ValueError("System camera is stale")
            timestamp=time.monotonic()-self.pair.origin
        record=self.engine.predict(request_id,timestamp);self.persist();return record

    def stop(self):
        if self.state not in ("ended","error"):self.state="stopped"
        self.stopping.set();self.wake.set()
        if hasattr(self,"thread") and threading.current_thread()!=self.thread:
            self.thread.join(timeout=5)
            if self.thread.is_alive():raise RuntimeError("Observation is stopping; wait for the active inference to finish")
        if self.state not in ("ended","error"):self.state="stopped"

    def telemetry(self,offset=0,limit=100,evidence=False):
        snapshot=self.engine.snapshot(offset,limit,evidence)
        with self.lock:
            elapsed=max(.001,time.monotonic()-self.created_monotonic)
            runtime=deepcopy(self.runtime)
            health=self.pair.health()
            for role,stats in runtime.items():
                stats["mean_inference_ms"]=round(stats["inference_total_ms"]/max(1,stats["frames"]),2)
                stats["processed_fps"]=round(stats["frames"]/elapsed,2)
                if self.source_mode=="live" and role in self.frame_times:
                    age=max(0,time.monotonic()-self.pair.origin-self.frame_times[role]-(self.policy.offset_sec if role=="acquisition" else 0))
                    health[role]["processed_frame_age_sec"]=age
                    if age>self.policy.freshness_sec and health[role]["state"]=="ready":health[role]["state"]="lagging"
            snapshot.update(state=self.state,error=self.error,storage_error=self.storage_error,
                source_mode=self.source_mode,shared_source_test=self.shared_test,
                camera_calibrations={role:deepcopy(c.payload) for role,c in self.calibrations.items()},
                policy=asdict(self.policy),
                cameras=health,runtime=runtime,frame_times=dict(self.frame_times),
                snapshot_fresh=bool(snapshot["metrics"] and (self.source_mode!="live" or
                    time.monotonic()-self.pair.origin-snapshot["metrics"]["timestamp"]<=self.policy.freshness_sec)))
        return snapshot

    def jpeg(self,role,clean=False):
        with self.lock:return (self.clean_frames if clean else self.jpeg_frames).get(role)

    def persist(self):
        try:self.store.persist(self.telemetry(0,len(self.engine.waits.records),True))
        except Exception as error:self.storage_error=str(error)

    def export(self):
        return self.store.export(self.telemetry(0,len(self.engine.waits.records),True))


class ObservationManager:
    def __init__(self,storage_root=None,profiles_root=None,session_factory=None):
        self.storage_root=Path(storage_root or ROOT/"data/experiments")
        self.profiles_root=Path(profiles_root or ROOT/"data/observation_profiles")
        self.session_factory=session_factory or ObservationSession
        self.current=None;self.lock=threading.RLock()

    def start(self,payload):
        with self.lock:
            if self.current:self.current.stop()
            session=self.session_factory(payload,self.storage_root)
            self.current=session
            return session

    def require(self,run_id=None):
        if self.current is None:raise ValueError("No observation run is active")
        if run_id is not None and run_id!=self.current.run_id:raise ValueError("Run identity changed; refresh the dashboard")
        return self.current

    def save_calibration(self,payload):
        calibration=CameraCalibration(payload)
        self.profiles_root.mkdir(parents=True,exist_ok=True)
        name=re.sub(r"[^A-Za-z0-9_-]","_",str(payload.get("name","profile")))[:60] or "profile"
        profile_id=f"{name}_{calibration.role}_{calibration.revision}.json"
        path=self.profiles_root/profile_id
        path.write_text(json.dumps(calibration.payload,indent=2,allow_nan=False))
        return {"profile_id":profile_id,"revision":calibration.revision,"calibration":calibration.payload}

    def profiles(self):
        return [{"profile_id":p.name,"calibration":json.loads(p.read_text())} for p in sorted(self.profiles_root.glob("*.json"))]

    def preview(self,role,source,rectification_path=None,source_mode="replay"):
        if role not in ("system","acquisition"):raise ValueError("Unknown camera role")
        if self.current and self.current.state in ("running","paused") and source_identity(source)==source_identity(self.current.calibrations[role].payload["source"]):
            if rectification_path!=self.current.calibrations[role].payload.get("rectification_path"):
                raise ValueError("Stop the run before changing lens calibration")
            result=self.current.jpeg(role,True)
            if result:return result
            raise ValueError("This live camera has not produced a valid preview yet")
        cap=cv2.VideoCapture(int(source) if str(source).isdigit() else str(source))
        try:
            if not cap.isOpened():raise ValueError("Cannot open camera/video preview")
            ok,frame=cap.read()
            if not ok:raise ValueError("No frame available from this source")
        finally:cap.release()
        if rectification_path:
            path=Path(rectification_path)
            if not path.is_absolute():path=ROOT/path
            rectifier=LensRectifier(str(path))
            if tuple(frame.shape[1::-1])!=rectifier.dim:raise ValueError("Lens calibration dimensions differ from source")
            frame=rectifier.rectify(frame)
        ok,result=cv2.imencode(".jpg",frame,[cv2.IMWRITE_JPEG_QUALITY,85])
        if not ok:raise ValueError("Could not encode preview")
        return result.tobytes()
