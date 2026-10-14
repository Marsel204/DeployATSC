from types import SimpleNamespace
import numpy as np
from src.perception.detector_tracker import DetectorTracker, TrackedVehicle


def tracker(monkeypatch):
    monkeypatch.setattr(DetectorTracker, "_load_model", lambda self: None)
    return DetectorTracker("unused.pt")


def test_two_camera_histories_and_reset_are_independent(monkeypatch):
    system, acquisition = tracker(monkeypatch), tracker(monkeypatch)
    for t in [0,.1,.2,.3]:
        a = TrackedVehicle(7,0,"car",[0,0,10,10],1,t)
        b = TrackedVehicle(7,0,"car",[20,20,30,30],1,t)
        system.update_velocities([a],lambda x,y:(x+t*10,y),timestamp=t)
        acquisition.update_velocities([b],lambda x,y:(x,y),timestamp=t)
    assert a.velocity_valid and a.velocity_mps > 1
    assert b.velocity_valid and b.velocity_mps == 0
    assert system.track_trajectories[7] != acquisition.track_trajectories[7]
    acquisition.reset_tracking()
    assert acquisition.epoch == 1 and not acquisition.track_trajectories
    assert system.epoch == 0 and system.track_trajectories


def test_warmup_is_not_valid_zero_velocity(monkeypatch):
    a=tracker(monkeypatch)
    vehicle=TrackedVehicle(7,0,"car",[0,0,10,10],1,0)
    a.update_velocities([vehicle],lambda x,y:(x,y),timestamp=0)
    assert not vehicle.velocity_valid


def test_detector_fallback_ids_are_not_stable(monkeypatch):
    class Tensor:
        def __init__(self,x):self.x=np.array(x)
        def cpu(self):return self
        def numpy(self):return self.x
    class Boxes:
        id=None
        xyxy=Tensor([[0,0,10,10]]);conf=Tensor([1]);cls=Tensor([0])
        def __len__(self):return 1
    a=tracker(monkeypatch)
    a.model=SimpleNamespace(names={0:"car"},track=lambda **kw:[SimpleNamespace(boxes=Boxes())])
    vehicle=a.track([np.zeros((20,20,3),np.uint8)],0)[0][0]
    assert not vehicle.stable_id
