"""Deterministic subprocess runner: no GPU, no wall-time evidence differences."""
import sys,json,io,zipfile,time,uuid
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import Future


def run(tree,core,clip,output,storage,persist_mode='inline',blas='default'):
    sys.path.insert(0,tree)
    # Load the candidate's additive startup policy on both trees, before numpy.
    import importlib.util,os
    spec=importlib.util.spec_from_file_location('src.common.runtime_tuning',Path(core)/'src/common/runtime_tuning.py')
    tuning=importlib.util.module_from_spec(spec);sys.modules[spec.name]=tuning;spec.loader.exec_module(tuning)
    sysfs=Path(storage)/'sys/devices/system/cpu/cpu0/cpufreq';sysfs.mkdir(parents=True)
    for name,value in [('scaling_governor','schedutil'),('scaling_min_freq','1510400'),('scaling_max_freq','1510400'),('scaling_cur_freq','1510400')]:
        (sysfs/name).write_text(value)
    sysfs=Path(storage)/'sys'
    os.environ['ATSC_BLAS_THREADS']=blas;os.environ.pop('OPENBLAS_NUM_THREADS',None)
    tuning.apply_blas_policy(sysfs_root=sysfs)
    host=tuning.host_provenance(sysfs)
    import cv2,numpy as np
    import src.ui.observation_session as m
    import src.experiment.predictions as predictions
    import src.experiment.export as export
    from src.perception.detector_tracker import TrackedVehicle
    m.ROOT=Path(core);cv2.setNumThreads(1)
    class Clock(datetime):
        @classmethod
        def now(cls,tz=None):return datetime(2026,10,29,0,0,tzinfo=timezone.utc)
    for module in (m,predictions,export):module.datetime=Clock
    time.monotonic=lambda:100.;time.perf_counter=lambda:100.
    m.uuid.uuid4=lambda:uuid.UUID(int=12345)
    class ScriptedDetector:
        def __init__(self,*a,**kw):self.epoch=0
        def track(self,frames,timestamp=None):
            return [[TrackedVehicle(int(timestamp*15)//20*10+j,0,'car',[50+j*35,80,70+j*35,120+(int(timestamp*15)%20)*3],.9,timestamp,obs_count=6) for j in range(4)] for _ in frames]
        def update_velocities(self,vs,project,timestamp=None):
            for v in vs:v.velocity_valid=True;v.velocity_mps=0. if timestamp<2 else 1.;v.ground_pos=project(*v.bottom_center)
        def reset_tracking(self):self.epoch+=1
    class ScriptedANFIS:
        def __init__(self,*a,**kw):pass
        def evaluate_green_time(self,**kw):return 35
    class Inline:
        def submit(self,fn,*args):
            future=Future()
            try:future.set_result(fn(*args))
            except BaseException as error:future.set_exception(error)
            return future
        def shutdown(self,*a,**kw):pass
    class Manual(m.ObservationSession):
        def _worker(self):self.ready.set()
        def _read_execution_provenance(self):
            self.sysfs_root=sysfs
            result=super()._read_execution_provenance();result.setdefault('host',host)
            self.last_execution=result
            return result
        def _observe_camera1(self,*a,**kw):
            # Golden compares every preview rather than running-preview wall cadence.
            self.preview_published_at.pop('system',None)
            return super()._observe_camera1(*a,**kw)
        def _request_persist(self):
            super()._request_persist()
            if persist_mode=='background':
                with self.persist_condition:
                    assert self.persist_condition.wait_for(lambda:not self.persist_pending and not self.persist_inflight,5)

        def telemetry(self,*a,**kw):
            value=super().telemetry(*a,**kw)
            # Add the requested additive counters to the original synchronous implementation.
            value.setdefault('preview_encoded_frames',sum(self.engine.processed.values()))
            value.setdefault('preview_coalesced_frames',0);value.setdefault('preview_errors',0)
            value.setdefault('performance_advice',tuning.performance_advice(host))
            return value
    def calibration(role):
        return {'role':role,'source':clip,'image_size':[320,240],'anchors':[[20,30],[300,30],[310,220],[10,220]],'polygon':[[20,30],[300,30],[310,220],[10,220]],'road_width_m':8,'road_length_m':40,'rectification_path':None,'signal_roi':{'xmin':10,'ymin':10,'xmax':40,'ymax':40},'stop_line':[[10,210],[310,210]],'crossing_direction':1}
    payload={'system':calibration('system'),'acquisition':calibration('acquisition'),'source_mode':'replay','step_mode':True,'persist_mode':persist_mode,'shared_test':True,'retention_budget':8,'policy':{'debounce_frames':1,'stop_persistence_sec':.1}}
    s=Manual(payload,storage,detector_factory=ScriptedDetector,anfis_factory=ScriptedANFIS)
    s.engine.predictions.executor.shutdown();s.engine.predictions.executor=Inline()
    samples=[];images=[]
    try:
        for step in range(1,91):
            if persist_mode=='background':s.applied_state='running'
            s._process_batch()
            s.applied_state='paused'
            images.append({role:s.jpeg(role).hex() for role in ('system','acquisition')})
            if step in (12,78):s.predict('click-'+str(step));s.engine.predictions.await_idle();s.persist()
            if step in (1,12,30,60,78,90):samples.append(s.telemetry(evidence=True))
        s.state=s.applied_state='ended';s._cleanup()
        with zipfile.ZipFile(io.BytesIO(s.export())) as bundle:
            files={name:json.loads(bundle.read(name)) if name.endswith('.json') else bundle.read(name).decode() for name in bundle.namelist()}
        files['manifest.json'].setdefault('host',host)
        Path(output).write_text(json.dumps({'telemetry':samples,'exports':files,'jpegs':images},allow_nan=False))
    finally:
        if not s.resources_released:s.state=s.applied_state='stopped';s._cleanup()


if __name__=='__main__':run(*sys.argv[1:])
