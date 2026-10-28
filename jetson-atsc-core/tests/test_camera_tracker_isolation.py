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


def point(t, y=30, stable=True, epoch=0):
    return TrackedVehicle(7, 1, "car", [0,y-4,10,y], 1, t, stable_id=stable, epoch=epoch)


def test_fallback_collision_never_changes_stable_history(monkeypatch):
    clean, polluted = tracker(monkeypatch), tracker(monkeypatch)
    polluted.update_velocities([point(0,0,False)],lambda x,y:(x,y),timestamp=0)
    assert not polluted.track_trajectories and not polluted.track_obs_counts
    for t in [.1,.2,.3]:
        a,b=point(t),point(t)
        clean.update_velocities([a],lambda x,y:(x,y),timestamp=t)
        polluted.update_velocities([b],lambda x,y:(x,y),timestamp=t)
    assert (a.velocity_mps,a.velocity_valid,a.obs_count)==(b.velocity_mps,b.velocity_valid,b.obs_count)==(0,True,3)
    before=list(polluted.track_trajectories[7])
    fallback=point(.4,0,False)
    polluted.update_velocities([fallback],lambda x,y:(x,y),timestamp=.4)
    assert polluted.track_trajectories[7]==before and fallback.obs_count==0
    b=point(.5);polluted.update_velocities([b],lambda x,y:(x,y),timestamp=.5)
    assert (b.velocity_mps,b.velocity_valid,b.obs_count)==(0,True,4)


def test_source_gap_rewarms_and_duplicate_reverse_or_old_epoch_cannot_append(monkeypatch):
    a=tracker(monkeypatch)
    a.max_gap_sec=.5
    for t in [0,.1,.2]:
        v=point(t);a.update_velocities([v],lambda x,y:(x,y),timestamp=t)
    assert v.velocity_valid
    before=list(a.track_trajectories[7])
    for t in [.2,.15]:
        v=point(t,100);a.update_velocities([v],lambda x,y:(x,y),timestamp=t)
        assert not v.velocity_valid
        assert a.track_trajectories[7]==before
    v=point(10);a.update_velocities([v],lambda x,y:(x,y),timestamp=10)
    assert not v.velocity_valid and v.obs_count==1
    a.reset_tracking()
    a.update_velocities([point(10.1,epoch=0)],lambda x,y:(x,y),timestamp=10.1)
    assert not a.track_trajectories
    for t in [10.2,10.3,10.4]:
        v=point(t,epoch=1);a.update_velocities([v],lambda x,y:(x,y),timestamp=t)
    assert v.velocity_valid


def test_wall_clock_pause_does_not_break_source_time_history(monkeypatch):
    a=tracker(monkeypatch);a.max_gap_sec=.5
    for t in [0,.1,.2]:
        monkeypatch.setattr('src.perception.detector_tracker.time.time',lambda:999999+t*10000)
        v=point(t);a.update_velocities([v],lambda x,y:(x,y),timestamp=t)
    assert v.velocity_valid and v.obs_count==3
