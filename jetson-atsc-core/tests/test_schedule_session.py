import cv2
import numpy as np

from src.perception.detector_tracker import TrackedVehicle
from src.ui.observation_session import ObservationSession
from test_observation_calibration import payload
from test_observation_sessions import FakeANFIS, FakeDetector


def test_two_real_sources_capture_a_once_and_save_per_vehicle_result(tmp_path):
    paths={}
    for role in ('system','acquisition'):
        path=tmp_path/(role+'.mp4')
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),5,(320,240))
        assert writer.isOpened()
        for index in range(120):
            t=index/5
            if role=='system':phase='RED' if t<1 or t>=21.6 else 'GREEN' if t<21 else 'YELLOW'
            else:phase='GREEN' if t<.2 or t>=23 else 'YELLOW' if t<.4 else 'RED'
            image=np.full((240,320,3),40,np.uint8)
            image[10:40,10:40]={'RED':(0,0,255),'GREEN':(0,255,0),'YELLOW':(0,255,255)}[phase]
            image[70:120,40:80]=(200,100,40)
            writer.write(image)
        writer.release()
        paths[role]=str(path)
    class Detector(FakeDetector):
        calls=0
        def track(self,frames,timestamp=None):
            return [[TrackedVehicle(4 if timestamp<8 else 228,1,'car',[40,70,80,120],.9,timestamp,stable_id=True)]]
        def detect_once(self,frame,timestamp=None):
            Detector.calls+=1
            return []
    class ANFIS(FakeANFIS):
        def evaluate_green_time(self,**kwargs):return 10
    data={role:{**payload(role),'source':paths[role]} for role in paths}
    data.update(policy={'debounce_frames':1},vehicle_identity={'mode':'persistent_v1'},
        schedule_comparison={'enabled':True,'unchanged_clearance':True},retention_budget=1)
    session=ObservationSession(data,tmp_path/'runs',detector_factory=Detector,anfis_factory=ANFIS)
    try:
        session.thread.join(timeout=35)
        assert not session.thread.is_alive(),session.error
        assert session.error is None
        assert Detector.calls==1
        saved=session.store.read()
        assert len(saved['schedule_episodes'])==1
        assert len(saved['vehicle_savings'])==1
        result=saved['vehicle_savings'][0]
        assert result['available'] and 9.5<=result['potential_saving_sec']<=10.5
        assert len(saved['vehicles'])==1
        assert saved['schedule_episodes'][0]['camera1_frame']['sha256']
    finally:
        session.stop()
