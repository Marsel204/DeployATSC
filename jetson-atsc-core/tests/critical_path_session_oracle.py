"""Verbatim HEAD methods for differential compatibility tests."""
from copy import deepcopy
from dataclasses import asdict
import time
import cv2
import numpy as np
from src.common.source_identity import public_data

class OriginalSessionMethods:
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
