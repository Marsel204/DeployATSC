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
    assert capture["clicked_at"]>=5 and capture["snapshot"]["timestamp"]==4.8
    session.engine.predictions.on_green(4.99,1)
    assert session.engine.predictions.rows()[0]["status"]=="late"
    session.control("play");assert session.pair.discarded
    assert session.detectors["acquisition"].epoch==1
    session.stop()
