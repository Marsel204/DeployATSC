import io,json,zipfile
import cv2,numpy as np
import pytest
from src.perception.detector_tracker import TrackedVehicle
from src.control.signal_interface import SignalInterface
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeANFIS


class KnownVehicleDetector:
    def __init__(self,*args,**kwargs):self.epoch=0
    def track(self,frames,timestamp=None):
        stopped=10<=timestamp<34
        bottom=20 if timestamp<40 else 35
        vehicle=TrackedVehicle(7,1,"car",[15,bottom-10,25,bottom],1,timestamp)
        vehicle.velocity_mps=0 if stopped else 2
        vehicle.velocity_valid=True
        return [[vehicle]]
    def update_velocities(self,*args,**kwargs):pass
    def reset_tracking(self):self.epoch+=1


def make_clip(path,fps,signal=False):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(64,48))
    assert writer.isOpened()
    for index in range(fps*60):
        t=index/fps
        color=((0,0,255) if t<30 else (0,255,0) if t<55 else (0,255,255)) if signal else (40,40,40)
        writer.write(np.full((48,64,3),color,np.uint8))
    writer.release()


def test_paired_video_pipeline_timing_waits_exports_and_no_hardware(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Observation invoked cabinet hardware')
    for method in ('__init__','from_config','start_watchdog','dispatch_green_actuation','_send_heartbeat'):
        monkeypatch.setattr(SignalInterface,method,forbidden)
    system=tmp_path/'system.mp4';acquisition=tmp_path/'acquisition.mp4'
    make_clip(system,25);make_clip(acquisition,30,True)
    def calibration(role,source):
        return {'role':role,'source':str(source),'image_size':[64,48],
                'anchors':[[3,3],[60,3],[60,44],[3,44]],'polygon':[[3,3],[60,3],[60,44],[3,44]],
                'road_width_m':8,'road_length_m':40,'signal_roi':{'xmin':0,'ymin':0,'xmax':64,'ymax':48},
                'stop_line':[[3,30],[60,30]],'crossing_direction':1}
    session=ObservationSession({'system':calibration('system',system),'acquisition':calibration('acquisition',acquisition),
        'step_mode':True,'policy':{'debounce_frames':3}},tmp_path/'runs',detector_factory=KnownVehicleDetector,anfis_factory=FakeANFIS)
    captured=False
    while not session.pair.ended:
        session._process_batch()
        if not captured and session.engine.time>=29:
            session.predict('before-green');session.engine.predictions.await_idle();captured=True
    session.stop()
    data=session.telemetry()
    cycle=data['comparisons'][0];vehicle=data['vehicles'][0]
    assert cycle['actual_green_sec']==pytest.approx(25,abs=.04)
    assert cycle['recommended_green_sec']==35 and cycle['difference_sec']==pytest.approx(10,abs=.04)
    assert cycle['eligible']
    assert vehicle['red_wait_sec']==pytest.approx(20,abs=.04)
    assert vehicle['total_stopped_sec']==pytest.approx(24,abs=.04)
    assert vehicle['crossed_stop_line'] and cycle['n_served']==1
    assert data['los']['grade'] is None
    assert not session.pair.system.cap.isOpened() and not session.pair.acquisition.cap.isOpened()
    with zipfile.ZipFile(io.BytesIO(session.export())) as archive:
        evidence=json.loads(archive.read('evidence.json'))
        assert evidence['vehicles'][0]['intervals'] and evidence['phase_events']
        assert evidence['predictions'][0]['snapshot']['timestamp']<cycle['green_start']
