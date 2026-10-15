import numpy as np
import pytest
import threading
from src.common.types import TrafficSnapshot, ActuationDecision
from src.experiment.contracts import RunPolicy,VehicleObservation
from src.experiment.engine import ObservationEngine


def test_link_prediction_with_actual_cycle_and_vehicle_waits():
    policy=RunPolicy(debounce_frames=1,stop_persistence_sec=0,max_gap_sec=1)
    engine=ObservationEngine("run",policy,{"xmin":0,"ymin":0,"xmax":40,"ymax":40},lambda p:p[1]-100,lambda s,c:ActuationDecision(c,35))
    engine.update_system(TrafficSnapshot(timestamp=0,v_w_pcu=5))
    for sequence in range(61):
        t=sequence/10
        engine.update_system(TrafficSnapshot(timestamp=t,v_w_pcu=5))
        color=(0,0,255) if t<3 else (0,255,0) if t<5.5 else (0,255,255)
        frame=np.full((40,40,3),color,np.uint8)
        car=VehicleObservation(7,"car",0,(10,10),velocity_valid=True)
        engine.observe_acquisition(frame,t,sequence+1,[car])
        if sequence==29:
            engine.predict("click");engine.predictions.await_idle()
    state=engine.snapshot()
    comparison=state["comparisons"][0]
    assert comparison["actual_green_sec"]==pytest.approx(2.5)
    assert comparison["recommended_green_sec"]==35
    assert comparison["difference_sec"]==pytest.approx(32.5)
    assert comparison["eligible"]
    assert state["vehicles"][0]["red_wait_sec"]==pytest.approx(3)
    engine.close()


def test_red_crossing_is_not_counted_as_served_during_green():
    policy=RunPolicy(debounce_frames=1,stop_persistence_sec=0,max_gap_sec=1)
    engine=ObservationEngine("run",policy,{"xmin":0,"ymin":0,"xmax":40,"ymax":40},lambda p:p[1]-20,lambda s,c:ActuationDecision(c,35))
    for seq,(t,color,point) in enumerate([(0,(0,0,255),(10,10)),(.1,(0,0,255),(10,25)),(.2,(0,255,0),(10,30)),(.3,(0,255,255),(10,30))],1):
        engine.observe_acquisition(np.full((40,40,3),color,np.uint8),t,seq,[VehicleObservation(7,"car",2,point,velocity_valid=True)])
    assert engine.snapshot()["comparisons"][0]["n_served"]==0
    engine.close()


@pytest.mark.parametrize("gap", ["live_pause", "signal_gap", "ambiguous_signal"])
@pytest.mark.parametrize("in_flight", [False, True])
def test_prediction_cannot_match_a_green_after_observation_continuity_is_lost(gap, in_flight):
    release = threading.Event()
    started = threading.Event()

    def evaluate(snapshot, cycle):
        started.set()
        assert release.wait(5)
        return ActuationDecision(cycle, 35)

    policy = RunPolicy(debounce_frames=1)
    engine = ObservationEngine("run", policy, {"xmin": 0, "ymin": 0, "xmax": 40, "ymax": 40},
                               lambda p: p[1]-100, evaluate)

    def observe(t, sequence, color):
        engine.update_system(TrafficSnapshot(timestamp=t))
        engine.observe_acquisition(np.full((40, 40, 3), color, np.uint8), t, sequence, [])

    try:
        observe(0, 1, (0, 0, 255))
        engine.predict("before-gap")
        assert started.wait(1)
        if not in_flight:
            release.set()
            engine.predictions.await_idle()
        if gap == "live_pause":
            engine.invalidate("live_pause", .1)
            engine.invalidate("live_resume", 40)
            resumed = 40
        elif gap == "signal_gap":
            resumed = 40
        else:
            observe(.1, 2, (0, 0, 0))
            resumed = .2
        observe(resumed, 3, (0, 0, 255))
        observe(resumed+.1, 4, (0, 255, 0))
        observe(resumed+.2, 5, (0, 255, 255))
        release.set()
        engine.predictions.await_idle()
        state = engine.snapshot()
        prediction = state["predictions"][0]
        assert prediction["status"] == "unmatched"
        assert prediction["cycle_id"] is None
        assert not prediction["eligible"]
        assert prediction["recommended_green_sec"] == 35
        assert not state["prediction_busy"]
        assert state["comparisons"][0]["complete"]
        assert not state["comparisons"][0]["eligible"]
        observe(resumed+.3, 6, (0, 0, 255))
        engine.predict("after-gap")
        engine.predictions.await_idle()
        observe(resumed+.4, 7, (0, 255, 0))
        observe(resumed+.5, 8, (0, 255, 255))
        recovered = engine.snapshot()
        assert recovered["predictions"][0]["status"] == "unmatched"
        assert recovered["predictions"][1]["eligible"]
        assert recovered["summary"]["valid_cycle_comparisons"] == 1
    finally:
        release.set()
        engine.predictions.await_idle()
        engine.close()
