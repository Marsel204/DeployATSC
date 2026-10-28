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


def qualified_snapshot(t,quality=True):
    from src.analytics.traffic_metrics import TrafficMetricsExtractor
    from test_homography_metrics import IdentityGeometry
    raw=TrafficMetricsExtractor(IdentityGeometry(),strict_motion=True).extract_approach_metrics([])
    if not quality:raw.pop('quality',None)
    return TrafficSnapshot.from_dict(raw,timestamp=t)


def feed(engine,t,seq,phase,quality=True):
    colors={'RED':(0,0,255),'GREEN':(0,255,0),'YELLOW':(0,255,255),'UNKNOWN':(0,0,0)}
    engine.update_system(qualified_snapshot(t,quality))
    engine.observe_acquisition(np.full((20,20,3),colors[phase],np.uint8),t,seq,[])


def make_engine(policy=None,evaluate=None):
    return ObservationEngine('phase-a',policy or RunPolicy(debounce_frames=1),
        {'xmin':0,'ymin':0,'xmax':20,'ymax':20},lambda p:p[1]-20,
        evaluate or (lambda s,c:ActuationDecision(c,35)))


def test_debounce_capture_uses_measurement_phase_and_freezes_both_phases():
    engine=make_engine(RunPolicy(debounce_frames=3))
    try:
        trace=[(0,'RED'),(.1,'RED'),(.2,'RED'),(.3,'GREEN'),(.4,'RED'),(.5,'RED'),
               (.6,'GREEN'),(.7,'GREEN'),(.8,'GREEN'),(1,'YELLOW'),(1.1,'YELLOW'),(1.2,'YELLOW')]
        for seq,(t,phase) in enumerate(trace,1):
            feed(engine,t,seq,phase)
            if t==.3:
                record=engine.predict('ambiguous');engine.predictions.await_idle()
                assert record['display_phase_at_capture']=='RED'
                assert record['measurement_phase_at_capture']=='UNKNOWN'
                assert record['phase_at_capture']=='UNKNOWN'
                assert record['phase_evidence']['candidate']=='GREEN'
        state=engine.snapshot();comparison=state['comparisons'][0]
        assert comparison['complete'] and not comparison['eligible'] and not comparison['qualified']
        assert 'measurement_phase_not_red' in comparison['qualification']['reasons']
    finally:engine.close()


@pytest.mark.parametrize('quality_state',['good','missing','partial','unavailable'])
def test_qualification_policy_is_separate_from_timing_eligibility(quality_state):
    calls=[]
    def evaluate(s,c):calls.append(s);return ActuationDecision(c,35)
    engine=make_engine(evaluate=evaluate)
    engine.predictions.scale_evidence=synthetic_scale_evidence()
    try:
        feed(engine,0,1,'RED',quality_state!='missing')
        if quality_state in ('partial','unavailable'):
            from dataclasses import replace
            from copy import deepcopy
            snap=engine.system_snapshot;quality=deepcopy(snap.quality)
            quality['Q'].update(state=quality_state,complete=False,available=quality_state=='partial')
            engine.update_system(replace(snap,quality=quality,queue_meters=None if quality_state=='unavailable' else 4))
        engine.predict('click');engine.predictions.await_idle()
        # Retry must return frozen inputs despite a newer frame.
        feed(engine,.1,2,'RED')
        retry=engine.predict('click')
        assert retry['snapshot']['timestamp']==0
        feed(engine,.2,3,'GREEN');feed(engine,.3,4,'YELLOW')
        state=engine.snapshot();p=state['predictions'][0];c=state['comparisons'][0]
        assert p['eligible']==(quality_state!='unavailable')
        assert c['eligible']==p['eligible']
        assert p['qualified']==c['qualified']==(quality_state=='good')
        assert p['qualification']['policy']==c['qualification']['policy']=='motion-phase-scale-v2'
        assert p['qualification']['scale']=='verified'
        if quality_state=='missing':assert p['qualification']['state']=='unknown'
        if quality_state=='unavailable':assert not calls
    finally:engine.close()


def test_snapshot_captures_one_prediction_view_even_if_callback_completes(monkeypatch):
    release=threading.Event();started=threading.Event()
    def evaluate(s,c):
        started.set();assert release.wait(5);return ActuationDecision(c,35)
    engine=make_engine(evaluate=evaluate)
    try:
        feed(engine,0,1,'RED');engine.predict('click');assert started.wait(1)
        feed(engine,.1,2,'GREEN');feed(engine,.2,3,'YELLOW')
        before=engine.snapshot()
        # Interleave immediately after the sole atomic view has been captured.
        original=getattr(engine.predictions,'view',engine.predictions.rows)
        calls=[]
        def interleave():
            result=original();calls.append(True)
            release.set();engine.predictions.await_idle()
            return result
        method='view' if hasattr(engine.predictions,'view') else 'rows'
        monkeypatch.setattr(engine.predictions,method,interleave)
        state=engine.snapshot()
        assert state['predictions'][0]==state['comparisons'][0]['prediction']
        assert len(calls)==1
        assert state['prediction_busy'] and state['predictions'][0]['status']=='predicting'
        assert state['snapshot_revision']==before['snapshot_revision']
        after=engine.snapshot()
        assert after['snapshot_revision']>state['snapshot_revision']
        assert after['prediction_revision']>state['prediction_revision']
        assert not after['prediction_busy']
        assert after['summary']['qualified_prediction_count']==sum(p['qualified'] for p in after['predictions'])
        assert after['summary']['qualified_cycle_comparisons']==sum(c['qualified'] for c in after['comparisons'])
    finally:release.set();engine.predictions.await_idle();engine.close()


@pytest.mark.parametrize('phase',['UNKNOWN','GREEN'])
def test_engine_crossing_attribution_uses_measurement_evidence(phase):
    engine=make_engine()
    try:
        for seq,(t,color,y) in enumerate([(0,'GREEN',10),(.1,phase,25),(.2,'GREEN',30),(.3,'YELLOW',30)],1):
            colors={'GREEN':(0,255,0),'YELLOW':(0,255,255),'UNKNOWN':(0,0,0)}
            engine.observe_acquisition(np.full((20,20,3),colors[color],np.uint8),t,seq,
                [VehicleObservation(7,'car',2,(10,y),velocity_valid=True)])
        state=engine.snapshot();row=state['vehicles'][0]
        assert row['crossing_cycle']==(None if phase=='UNKNOWN' else 1)
        assert sum(c['n_served'] for c in state['comparisons'])==(0 if phase=='UNKNOWN' else 1)
    finally:engine.close()


def test_closed_prediction_view_does_not_retain_a_busy_admission_slot():
    release=threading.Event();started=threading.Event()
    def evaluate(s,c):started.set();assert release.wait(5);return ActuationDecision(c,35)
    engine=make_engine(evaluate=evaluate)
    try:
        feed(engine,0,1,'RED');engine.predict('click');assert started.wait(1)
        engine.close();release.set();engine.predictions.await_idle()
        assert not engine.snapshot()['prediction_busy']
    finally:release.set();engine.predictions.await_idle();engine.close()


def test_green_unknown_green_remains_two_incomplete_segments():
    engine=make_engine()
    try:
        for seq,(t,phase) in enumerate([(0,'RED'),(.1,'GREEN'),(.2,'UNKNOWN'),(.3,'GREEN'),(.4,'YELLOW')],1):
            feed(engine,t,seq,phase)
        state=engine.snapshot()
        assert len(state['comparisons'])==2
        assert all(not c['complete'] and c['actual_green_sec'] is None for c in state['comparisons'])
        assert not any(c['eligible'] or c['qualified'] for c in state['comparisons'])
    finally:engine.close()


def test_async_capture_freezes_nested_quality_for_inference_and_retry():
    release=threading.Event();started=threading.Event();seen=[]
    def evaluate(s,c):
        started.set();assert release.wait(5)
        seen.append(s.quality['Q']['state']);return ActuationDecision(c,35)
    engine=make_engine(evaluate=evaluate)
    try:
        feed(engine,0,1,'RED');first=engine.predict('frozen');assert started.wait(1)
        engine.system_snapshot.quality['Q']['state']='changed'
        first['snapshot']['quality']['Q']['state']='response-mutated'
        release.set();engine.predictions.await_idle()
        assert seen==['measured_zero']
        assert engine.predict('frozen')['snapshot']['quality']['Q']['state']=='measured_zero'
    finally:release.set();engine.predictions.await_idle();engine.close()


def test_sealed_engine_and_book_reject_all_late_mutations():
    release=threading.Event();started=threading.Event()
    def evaluate(s,c):started.set();assert release.wait(5);return ActuationDecision(c,35)
    engine=make_engine(evaluate=evaluate)
    try:
        feed(engine,0,1,'RED');engine.predict('blocked');assert started.wait(1)
        engine.close('operator_stop')
        before=engine.snapshot(evidence=True)
        assert not engine.predictions.resources_released
        engine.update_system(qualified_snapshot(20))
        engine.observe_acquisition(np.full((20,20,3),(0,255,0),np.uint8),20,2,[])
        engine.invalidate('late',20)
        engine.predictions.invalidate('late',20)
        engine.predictions.on_green(20,9)
        with pytest.raises(ValueError):engine.predictions.cancel('blocked')
        engine.close('again')
        release.set();engine.predictions.await_idle()
        assert engine.predictions.resources_released
        assert engine.snapshot(evidence=True)==before
    finally:release.set();engine.predictions.await_idle();engine.close()


def test_retimestamped_duplicate_does_not_change_engine_time_or_revision():
    engine=make_engine()
    try:
        frame=np.full((20,20,3),(0,255,0),np.uint8)
        engine.observe_acquisition(frame,0,1,[])
        before=engine.snapshot(evidence=True)
        engine.observe_acquisition(frame,.2,1,[])
        assert engine.snapshot(evidence=True)==before
    finally:engine.close()


def test_old_generation_packet_cannot_commit_after_boundary():
    engine=make_engine()
    try:
        generation=engine.generation
        engine.invalidate('live_resume')
        before=engine.snapshot(evidence=True)
        engine.update_system(qualified_snapshot(20),generation=generation)
        engine.observe_acquisition(np.zeros((20,20,3),np.uint8),20,1,[],generation=generation)
        assert engine.snapshot(evidence=True)==before
    finally:engine.close()


def synthetic_scale_evidence():
    from test_observation_calibration import payload
    from src.experiment.calibration import CameraCalibration
    result={}
    for role in ('system','acquisition'):
        data=payload(role)
        data.update(scale_evidence={'method':'measured_dimensions','reference':'synthetic fixture survey'},binding={'reviewed':True})
        calibration=CameraCalibration(data)
        result[role]={'revision':calibration.revision,**calibration.scale_status}
    return result


def test_new_v2_default_scale_is_diagnostic_but_timing_stays_eligible():
    engine=make_engine()
    try:
        feed(engine,0,1,'RED');engine.predict('legacy-scale');engine.predictions.await_idle()
        feed(engine,.1,2,'GREEN');feed(engine,.2,3,'YELLOW')
        state=engine.snapshot();row=state['comparisons'][0]
        assert row['eligible'] and not row['qualified']
        assert row['qualification']['policy']=='motion-phase-scale-v2'
        assert row['prediction']['qualification']['policy']=='motion-phase-scale-v2'
        assert row['qualification']['scale']=='unverified'
        assert state['summary']['valid_cycle_comparisons']==1
        assert state['summary']['qualified_cycle_comparisons']==0
    finally:engine.close()


@pytest.mark.parametrize('missing_role',[None,'system','acquisition'])
def test_v2_requires_reviewed_scale_for_each_camera(missing_role):
    evidence=synthetic_scale_evidence()
    if missing_role:evidence[missing_role]['state']='review_required'
    engine=ObservationEngine('scale',RunPolicy(debounce_frames=1),
        {'xmin':0,'ymin':0,'xmax':20,'ymax':20},lambda p:-1,lambda s,c:ActuationDecision(c,10),scale_evidence=evidence)
    try:
        feed(engine,0,1,'RED');engine.predict('click');engine.predictions.await_idle()
        feed(engine,.1,2,'GREEN');feed(engine,.2,3,'YELLOW')
        state=engine.snapshot();row=state['comparisons'][0]
        assert row['eligible'] and row['qualified']==(missing_role is None)
        assert row['prediction']['scale_evidence']==evidence
        assert row['qualification']['model_domain']=='unknown'
    finally:engine.close()


def test_old_record_policy_is_not_reinterpreted_by_v2_reader():
    from copy import deepcopy
    engine=make_engine()
    try:
        feed(engine,0,1,'RED');engine.predict('old');engine.predictions.await_idle()
        feed(engine,.1,2,'GREEN');feed(engine,.2,3,'YELLOW')
        with engine.predictions.lock:
            row=engine.predictions.records['old']
            row['qualification']={'policy':'motion-phase-v1','state':'qualified','reasons':[],'scale':'unverified'}
            row['qualified']=True
        before=deepcopy(engine.predictions.rows())
        comparison=engine.snapshot()['comparisons'][0]
        assert comparison['qualified'] and comparison['qualification']['policy']=='motion-phase-v1'
        assert engine.predictions.rows()==before
        with engine.predictions.lock:engine.predictions.records['old'].pop('qualification')
        assert engine.snapshot()['comparisons'][0]['qualification']['state']=='unknown'
    finally:engine.close()


def test_null_legacy_qualification_is_unknown():
    engine=make_engine()
    try:
        feed(engine,0,1,'RED');engine.predict('old-null');engine.predictions.await_idle()
        feed(engine,.1,2,'GREEN');feed(engine,.2,3,'YELLOW')
        with engine.predictions.lock:engine.predictions.records['old-null']['qualification']=None
        row=engine.snapshot()['comparisons'][0]
        assert row['eligible'] and not row['qualified']
        assert row['qualification']['state']=='unknown'
    finally:engine.close()
