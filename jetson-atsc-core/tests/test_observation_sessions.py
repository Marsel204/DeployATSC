from copy import deepcopy
import threading
import time
import numpy as np
import pytest
from src.ui.observation_session import ObservationSession
from test_observation_calibration import payload


class FakeDetector:
    def __init__(self,*args,**kwargs):self.epoch=0;self.track_trajectories={}
    def track(self,frames,timestamp=None):return [[] for _ in frames]
    def update_velocities(self,*args,**kwargs):pass
    def reset_tracking(self):self.epoch+=1


class FakeANFIS:
    def __init__(self,*args,**kwargs):pass
    def evaluate_green_time(self,**kwargs):return 35


def run_payload(video_factory,paused=True):
    a=payload();a["source"]=str(video_factory("road.mp4",color=(40,40,40)))
    b=payload("acquisition");b["source"]=str(video_factory("signal.mp4",color=(0,0,255)))
    return {"system":a,"acquisition":b,"source_mode":"replay","step_mode":paused,
            "policy":{"debounce_frames":1},"shared_test":False}


def test_role_streams_pause_step_and_cleanup(video_factory,tmp_path):
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    initial=session.telemetry()
    assert initial["state"]=="paused"
    assert session.jpeg("system")!=session.jpeg("acquisition")
    time.sleep(.05)
    assert session.telemetry()["processed_frames"]==initial["processed_frames"]
    session.control("step")
    deadline=time.monotonic()+1
    while session.telemetry()["processed_frames"]["system"]==1 and time.monotonic()<deadline:time.sleep(.01)
    assert session.telemetry()["processed_frames"]["system"]==2
    session.stop()
    assert session.state=="stopped" and not session.pair.system.cap.isOpened()
    assert (tmp_path/session.run_id/"manifest.json").exists()


def test_shared_source_requires_explicit_test_mode(video_factory,tmp_path):
    data=run_payload(video_factory)
    data["acquisition"]["source"]=data["system"]["source"]
    with pytest.raises(ValueError,match="shared"):
        ObservationSession(data,tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)


def test_live_prediction_uses_monotonic_completion_and_resume_discards_backlog(video_factory,tmp_path,monkeypatch):
    import src.ui.observation_session as module
    from src.experiment.sources import FramePacket
    class Pair:
        def __init__(self,*args,**kwargs):
            self.origin=time.monotonic()-5;self.first=True;self.ended=False;self.has_backlog=False;self.discarded=False
        def step(self):
            if not self.first:return []
            self.first=False
            return [FramePacket("system",1,4.8,4.8,np.full((240,320,3),40,np.uint8)),
                    FramePacket("acquisition",1,4.8,4.8,np.full((240,320,3),(0,0,255),np.uint8))]
        def health(self):
            return {r:{"state":"ready","age_sec":0,"image_size":[320,240]} for r in ("system","acquisition")}
        def close(self):self.ended=True
        def discard_pending(self):self.discarded=True
    monkeypatch.setattr(module,"LivePair",Pair)
    data=run_payload(video_factory);data["source_mode"]="live"
    session=ObservationSession(data,tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    capture=session.predict("live-click");session.engine.predictions.await_idle()
    # Camera 1 analyses the frozen 4.8 s still image after the click (on-demand contract).
    analysed=session.engine.predictions.records["live-click"]
    assert capture["clicked_at"]>=5 and capture["camera1_frame"]["timestamp"]==4.8
    assert analysed["snapshot"]["timestamp"]==4.8
    session.engine.predictions.on_green(4.99,1)
    assert session.engine.predictions.rows()[0]["status"]=="late"
    result=session.control("play");wait_applied(session,result["control_seq"])
    assert session.pair.discarded
    assert session.detectors["acquisition"].epoch==1
    session.stop()


def wait_applied(session,sequence):
    deadline=time.monotonic()+2
    while session.telemetry().get('applied_control_seq',-1)<sequence and time.monotonic()<deadline:
        time.sleep(.005)
    assert session.telemetry()['applied_control_seq']>=sequence


def test_live_rapid_pause_play_during_track_uses_worker_boundary(video_factory,tmp_path,monkeypatch):
    import src.ui.observation_session as module
    from src.experiment.sources import FramePacket
    entered=threading.Event();release=threading.Event();calls=[];resets=[]
    class Pair:
        def __init__(self,*args,**kwargs):
            self.origin=time.monotonic();self.batch=0;self.ended=False;self.has_backlog=False;self.discards=[]
        def step(self):
            self.batch+=1
            if self.batch>2:return []
            return [FramePacket(role,seq,t,t,np.full((240,320,3),(0,0,255),np.uint8))
                for role,seq,t in [('system',self.batch,self.batch/10),('acquisition',self.batch,self.batch/10),('acquisition',99,.3)]]
        def health(self):return {r:{'state':'ready','age_sec':0} for r in ('system','acquisition')}
        def close(self):self.ended=True
        def discard_pending(self):self.discards.append(threading.current_thread().name)
    class Detector(FakeDetector):
        def track(self,frames,timestamp=None):
            calls.append((timestamp,self.epoch,threading.current_thread().name))
            # Camera 1 is never tracked, so the third call is Camera 2's second-batch frame.
            if timestamp==.2 and len(calls)>2 and not entered.is_set():
                entered.set();assert release.wait(5)
            return super().track(frames,timestamp)
        def reset_tracking(self):resets.append(threading.current_thread().name);super().reset_tracking()
    monkeypatch.setattr(module,'LivePair',Pair)
    data=run_payload(video_factory);data['source_mode']='live'
    session=ObservationSession(data,tmp_path,detector_factory=Detector,anfis_factory=FakeANFIS)
    try:
        first=session.control('play');wait_applied(session,first['control_seq'])
        assert entered.wait(1)
        paused=session.control('pause');playing=session.control('play')
        assert playing['requested_state']=='running' and playing['applied_control_seq']<playing['control_seq']
        assert len(resets)==1 # first play, Camera 2 only; HTTP cannot reset blocked inference
        release.set();wait_applied(session,playing['control_seq'])
        assert all(name.startswith('Observation-') for name in resets+session.pair.discards)
        assert all(epoch==1 for t,epoch,name in calls if t==.2)
        # Remaining .3 packet from the blocked local batch was discarded on resume.
        assert not any(t==.3 and epoch==2 for t,epoch,name in calls)
        assert session.engine.processed['acquisition']==2 # first batch's two packets only
        assert session.detectors['acquisition'].epoch==2
        assert session.detectors['system'].epoch==0  # Camera 1 model never reset by the worker
    finally:release.set();session.stop()


def test_replay_pause_preserves_local_batch_and_step_drains_exact_backlog(video_factory,tmp_path,monkeypatch):
    import src.ui.observation_session as module
    from src.experiment.sources import FramePacket
    entered=threading.Event();release=threading.Event()
    class Pair:
        def __init__(self,*a,**kw):
            self.batch=0;self.ended=False;self.has_backlog=False
            self.system=type('Source',(),{'fps':100})()
            self.acquisition=type('Source',(),{'ended':False})()
        def step(self):
            self.batch+=1
            self.has_backlog=self.batch==2
            t=0 if self.batch==1 else 1 if self.batch==2 else 1.1
            if self.batch>3:return []
            return [FramePacket('system' if self.batch<3 else 'acquisition',21 if self.batch==3 else self.batch,t,t,np.zeros((240,320,3),np.uint8)),
                    FramePacket('acquisition',self.batch*10,t+.01,t+.01,np.zeros((240,320,3),np.uint8))]
        def close(self):self.ended=True
        def health(self):return {r:{'image_size':[320,240]} for r in ('system','acquisition')}
    # Camera 1 has no per-packet detector call; block its preview admission instead.
    observe=module.ObservationSession._observe_camera1
    def blocking(self,packet,frame,generation):
        if packet.timestamp==1:entered.set();assert release.wait(5)
        return observe(self,packet,frame,generation)
    monkeypatch.setattr(module.ObservationSession,'_observe_camera1',blocking)
    monkeypatch.setattr(module,'ReplayPair',Pair)
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        session.control('play');assert entered.wait(1)
        pause=session.control('pause');release.set();wait_applied(session,pause['control_seq'])
        assert session.engine.processed=={'system':2,'acquisition':1}
        paused_time=session.engine.time;time.sleep(.05);assert session.engine.time==paused_time
        step=session.control('step');wait_applied(session,step['control_seq'])
        deadline=time.monotonic()+1
        while session.engine.processed['acquisition']<4 and time.monotonic()<deadline:time.sleep(.005)
        assert session.engine.processed=={'system':2,'acquisition':4}
        assert session.pair.batch==3 and session.engine.time==pytest.approx(1.11)
        assert all(detector.epoch==0 for detector in session.detectors.values())
    finally:release.set();session.stop()
