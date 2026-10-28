"""Barrier tests use synthetic captures/inference; no native work is cancelled."""
import threading
import time
import pytest
from fastapi.testclient import TestClient
from src.common.types import ActuationDecision
from src.ui.observation_session import ObservationSession,ObservationManager
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload


def test_stop_timeout_keeps_detector_owner_and_discards_late_packet(video_factory,tmp_path):
    entered=threading.Event();release=threading.Event()
    class Detector(FakeDetector):
        def track(self,frames,timestamp=None):
            if timestamp>0:entered.set();assert release.wait(5)
            return super().track(frames,timestamp)
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=Detector,anfis_factory=FakeANFIS)
    try:
        # Camera 1 admits its preview frame without YOLO; the blocked native call is Camera 2's.
        session.control('play');assert entered.wait(1);before=session.engine.processed.copy()
        with pytest.raises(RuntimeError):session.stop(timeout=.01)
        status=session.lifecycle()
        assert status['stop_requested'] and not status['resources_released'] and not status['finalized']
        assert session.engine.processed==before
        release.set();session.stop(timeout=2)
        assert session.engine.processed==before
        assert session.lifecycle()['measurement_sealed'] and session.lifecycle()['finalized']
        assert session.lifecycle()['durable_finalized'] is True
        assert session.store.recovery_status()['durable_finalized']
    finally:release.set();session.stop()


def test_prediction_admission_closed_still_retains_native_resource_owner(video_factory,tmp_path):
    entered=threading.Event();release=threading.Event()
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    def evaluate(s,c):entered.set();assert release.wait(5);return ActuationDecision(c,35)
    session.engine.predictions.evaluate=evaluate
    try:
        session.predict('blocked');assert entered.wait(1)
        with pytest.raises(RuntimeError):session.stop(timeout=.01)
        status=session.lifecycle()
        assert status['measurement_sealed'] and status['prediction_release_pending']
        assert not session.engine.snapshot()['prediction_busy']
        assert not status['resources_released'] and not status['finalized']
        sealed=session.engine.snapshot(evidence=True)
        release.set();session.stop(timeout=2)
        assert session.engine.snapshot(evidence=True)==sealed
        assert session.lifecycle()['resources_released'] and session.lifecycle()['finalized']
    finally:release.set();session.stop()


def test_failed_final_store_reports_unclean_and_last_success(video_factory,tmp_path,monkeypatch):
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    def failure(state):raise OSError('disk unavailable')
    monkeypatch.setattr(session.store,'persist',failure)
    session.stop()
    status=session.lifecycle()
    assert status['resources_released'] and status['measurement_sealed']
    assert not status['finalized'] and status['terminal_state']=='unclean'
    assert session.storage_error=='disk unavailable'
    assert status['last_persisted_revision'] is not None and status['durable_revision']==status['last_persisted_revision']
    assert not status['durable_finalized']


def test_lifespan_closes_both_managers_and_reports_deadline(monkeypatch):
    import src.ui.app as application
    import src.ui.observation_api as api
    calls=[]
    class Manager:
        def shutdown(self,timeout):calls.append(timeout);return {'resources_released':False,'finalized':False}
    monkeypatch.setattr(api,'observations',Manager())
    monkeypatch.setattr(application,'manager',Manager())
    monkeypatch.setenv('ATSC_SHUTDOWN_TIMEOUT_SEC','.03')
    with TestClient(application.app):pass
    assert len(calls)==2 and all(0<=value<=.03 for value in calls)
    report=application.app.state.shutdown_report
    assert not report['observation']['resources_released'] and not report['legacy']['finalized']


def test_manager_retains_owner_while_capture_release_pending(video_factory,tmp_path,monkeypatch):
    import src.ui.observation_session as module
    from functools import partial
    from src.ui.observation_session import ObservationConflict
    released=threading.Event()
    class Pair:
        def __init__(self,*args,**kw):self.ended=False;self.has_backlog=False
        def step(self):return []
        def health(self):return {r:{'state':'ready','age_sec':0} for r in ('system','acquisition')}
        def close(self):self.ended=True
        @property
        def resources_released(self):return released.is_set()
    monkeypatch.setattr(module,'LivePair',Pair)
    manager=ObservationManager(tmp_path,session_factory=partial(ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS),shutdown_timeout=.02)
    data=run_payload(video_factory);data['source_mode']='live'
    session=manager.start(data)
    try:
        result=manager.shutdown(timeout=.02)
        assert not result['resources_released'] and result['capture_release_pending']
        assert manager.current is session
        with pytest.raises(ObservationConflict):manager.start(data)
    finally:released.set();session.stop(timeout=2)


def test_legacy_stop_does_not_release_under_blocked_worker():
    from src.ui.session_manager import ReplaySession
    from test_observation_live_sources import BlockingCapture
    cap=BlockingCapture()
    session=ReplaySession.__new__(ReplaySession)
    session.cap=cap;session.state='running';session.native_fps=30;session._is_stopped=False
    session._play_event=threading.Event();session._step_event=threading.Event()
    session._released=threading.Event();session._release_lock=threading.Lock()
    def process():cap.read();return False
    session._process_single_frame=process
    session._thread=threading.Thread(target=session._worker_loop,name='Legacy-owner')
    session._thread.start()
    try:
        assert cap.entered.wait(1)
        with pytest.raises(RuntimeError):session.stop(timeout=.01)
        assert not cap.releases and not session.resources_released
    finally:
        cap.unblock.set();session.stop(timeout=2)
    assert cap.releases==['Legacy-owner'] and session.resources_released


def test_lifespan_cleanup_runs_when_serving_context_raises(monkeypatch):
    import src.ui.app as application
    import src.ui.observation_api as api
    calls=[]
    class Manager:
        def shutdown(self,timeout):calls.append(True);return {'resources_released':True,'finalized':True}
    monkeypatch.setattr(api,'observations',Manager());monkeypatch.setattr(application,'manager',Manager())
    with pytest.raises(RuntimeError,match='serving failed'):
        with TestClient(application.app):raise RuntimeError('serving failed')
    assert len(calls)==2


def test_failed_startup_retains_capture_owner_and_exposes_telemetry(video_factory,tmp_path,monkeypatch):
    import src.ui.observation_session as module
    from functools import partial
    from src.ui.observation_session import ReplacementFailed
    released=threading.Event()
    class Pair:
        def __init__(self,*a,**kw):
            self.ended=False;self.has_backlog=False
            error=RuntimeError('capture startup failed');error.resource_owner=self;raise error
        def close(self):self.ended=True
        def health(self):return {'system':{'state':'failed','release_pending':not released.is_set()}}
        @property
        def resources_released(self):return released.is_set()
    monkeypatch.setattr(module,'LivePair',Pair)
    manager=ObservationManager(tmp_path,session_factory=partial(ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
    data=run_payload(video_factory);data['source_mode']='live'
    try:
        with pytest.raises(ReplacementFailed):manager.start(data)
        failed=manager.current
        assert failed is not None and not failed.resources_released
        telemetry=failed.telemetry()
        assert telemetry['state']=='error' and telemetry['lifecycle']['capture_release_pending']
        assert telemetry['error']=='capture startup failed'
    finally:
        released.set()
        if manager.current:manager.current.stop(timeout=2)


def test_shutdown_during_final_write_reports_pending_without_false_finalization(video_factory,tmp_path,monkeypatch):
    entered=threading.Event();release=threading.Event()
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    original=session.store.persist
    def blocked(state):entered.set();assert release.wait(5);return original(state)
    monkeypatch.setattr(session.store,'persist',blocked)
    try:
        session.control('stop');assert entered.wait(1)
        with pytest.raises(RuntimeError):session.stop(timeout=.01)
        assert session.lifecycle()['terminal_state']=='finalizing'
        assert not session.lifecycle()['finalized']
        release.set();session.stop(timeout=2)
        assert session.lifecycle()['finalized']
    finally:release.set();session.stop()


def test_models_are_dropped_before_reporting_release_during_final_write(video_factory,tmp_path,monkeypatch):
    import weakref
    refs=[];entered=threading.Event();release=threading.Event()
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=Detector,anfis_factory=FakeANFIS)
    def blocked(state):entered.set();assert release.wait(5)
    monkeypatch.setattr(session.store,'persist',blocked)
    try:
        session.control('stop');assert entered.wait(1)
        assert session.lifecycle()['resources_released']
        assert all(ref() is None for ref in refs)
    finally:release.set();session.stop()


def test_shutdown_during_reserved_start_does_not_report_old_owner_as_all_released(tmp_path):
    entered=threading.Event();release=threading.Event();errors=[]
    class Owner:
        run_id='old';resources_released=True
        def stop(self,**kw):pass
        def lifecycle(self):return {'resources_released':True,'finalized':True}
    def preflight(*a):entered.set();assert release.wait(5)
    def factory(*a):raise RuntimeError('should not acquire after shutdown')
    manager=ObservationManager(tmp_path,session_factory=factory,preflight=preflight)
    manager.current=Owner()
    def start():
        try:manager.start({})
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=start);thread.start()
    try:
        assert entered.wait(1)
        result=manager.shutdown(timeout=.01)
        assert result['start_pending'] and not result['resources_released'] and not result['finalized']
    finally:release.set();thread.join(2)
    assert errors


def test_stop_requested_rejects_prediction_before_native_tracker_exits(video_factory,tmp_path):
    entered=threading.Event();release=threading.Event()
    class Detector(FakeDetector):
        def track(self,frames,timestamp=None):
            if timestamp>0:entered.set();assert release.wait(5)
            return super().track(frames,timestamp)
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=Detector,anfis_factory=FakeANFIS)
    try:
        session.control('play');assert entered.wait(1);session.control('stop')
        with pytest.raises(ValueError,match='ended|stopping'):session.predict('too-late')
    finally:release.set();session.stop()


def test_after_replace_uncertainty_does_not_publish_live_durable_success(video_factory,tmp_path,monkeypatch):
    import json
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    original=session.store._write
    def fail_after_evidence(name,state):
        original(name,state)
        if name=='evidence.json':raise OSError('after evidence replacement')
    monkeypatch.setattr(session.store,'_write',fail_after_evidence)
    session.stop()
    persisted=session.store.read()
    # The checkpoint replace and directory fsync completed before this injected
    # caller error. Recovery can select the complete new generation; the live
    # session must retain its old acknowledgment and report uncertainty.
    assert persisted['lifecycle']['finalized']
    assert persisted['lifecycle']['durable_finalized']
    assert persisted['durability']['generation']==session.durable_generation+1
    assert session.pending_commit is not None
    assert not session.lifecycle()['durable_finalized']
    assert not session.lifecycle()['finalized'] and session.lifecycle()['terminal_state']=='unclean'


@pytest.mark.parametrize("failure",["store","track"])
def test_startup_exception_traceback_does_not_retain_released_models(video_factory,tmp_path,monkeypatch,failure):
    import weakref
    import src.ui.observation_session as module
    refs=[]
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
    class ANFIS(FakeANFIS):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
    def failed_store(*args):raise OSError('store initialization failed')
    if failure=='store':monkeypatch.setattr(module,'ExperimentStore',failed_store)
    else:
        def failed_track(self,*a,**kw):raise OSError('first tracking call failed')
        monkeypatch.setattr(Detector,'track',failed_track)
    with pytest.raises(OSError) as captured:
        ObservationSession(run_payload(video_factory),tmp_path,detector_factory=Detector,anfis_factory=ANFIS)
    owner=captured.value.session_owner
    owner.stop(timeout=2)
    assert owner.resources_released and all(ref() is None for ref in refs)


@pytest.mark.parametrize('budget',['not-a-number','nan','inf','-1'])
def test_invalid_shutdown_deadline_is_rejected_before_serving(monkeypatch,budget):
    import src.ui.app as application
    monkeypatch.setenv('ATSC_SHUTDOWN_TIMEOUT_SEC',budget)
    entered=[]
    with pytest.raises(ValueError,match='shutdown|Shutdown|convert'):
        with TestClient(application.app):entered.append(True)
    assert not entered


# Phase B corrections: one process-wide owner, failed legacy startup, isolated cleanup.
@pytest.fixture
def ownership(monkeypatch):
    import src.ui.observation_session as observation
    import src.ui.session_manager as legacy
    from src.ui.runtime_ownership import RuntimeOwnership
    shared=RuntimeOwnership()
    monkeypatch.setattr(observation,'runtime_ownership',shared);monkeypatch.setattr(legacy,'runtime_ownership',shared)
    return shared


class LegacyOwner:
    def __init__(self,**kw):self.resources_released=False;self.run_id='legacy'
    def stop(self,timeout=1.):self.resources_released=True


def test_distinct_legacy_manager_cannot_replace_an_unreleased_owner(tmp_path,monkeypatch,ownership):
    import src.ui.session_manager as legacy
    from src.ui.runtime_ownership import RuntimeConflict
    made=[]
    class Replay(LegacyOwner):
        def __init__(self,**kw):super().__init__(**kw);made.append(self)
    monkeypatch.setattr(legacy,'ReplaySession',Replay)
    first=legacy.SessionManager();second=legacy.SessionManager()
    old=first.start_session('unused','unused');owner=first.start_session('unused','unused')
    assert old.resources_released and ownership.current_owner()==('legacy',owner)
    try:
        with pytest.raises(RuntimeConflict):second.start_session('unused','unused')
        assert made==[old,owner] and ownership.current_owner()==('legacy',owner)
        second.stop_session()
        assert ownership.current_owner()==('legacy',owner) and not owner.resources_released
        with pytest.raises(RuntimeConflict):second.start_session('unused','unused')
        observation=ObservationManager(tmp_path,session_factory=lambda *a:made.append('observation'),preflight=lambda *a:None)
        with pytest.raises(RuntimeConflict):observation.start({},before_handoff=lambda:None)
        with pytest.raises(RuntimeConflict):ownership.claim('legacy',LegacyOwner())
        assert made==[old,owner] and ownership.owner is owner
    finally:first.stop_session()
    replacement=second.start_session('unused','unused')
    assert ownership.current_owner()==('legacy',replacement) and len(made)==3
    second.stop_session();assert replacement.resources_released


class ReleaseCapture:
    def __init__(self,failures=0):self.releases=0;self.failures=failures
    def isOpened(self):return True
    def get(self,prop):return 30
    def read(self):return True,None
    def release(self):
        if self.failures:self.failures-=1;raise OSError('native capture release failed')
        self.releases+=1


def inject_legacy_components(monkeypatch,refs,failure,fail,cap):
    """The real ReplaySession constructor runs; only its components are injected."""
    import weakref
    import src.ui.session_manager as legacy
    class Detector:
        def __init__(self,*a,**kw):refs.append(weakref.ref(self))
        def track(self,*a,**kw):
            if fail[0] and failure=='first_track':raise RuntimeError('first tracking failed')
    class ANFIS:
        def __init__(self,*a,**kw):
            refs.append(weakref.ref(self))
            if fail[0] and failure=='anfis':raise RuntimeError('ANFIS startup failed after detector allocation')
    monkeypatch.setattr(legacy,'LensRectifier',lambda *a:None)
    monkeypatch.setattr(legacy.OpticalPhaseMonitor,'from_config',lambda *a:None)
    monkeypatch.setattr(legacy.HomographyEngine,'from_config',lambda *a:None)
    monkeypatch.setattr(legacy.TrafficMetricsExtractor,'from_config',lambda *a:None)
    monkeypatch.setattr(legacy,'DetectorTracker',Detector)
    monkeypatch.setattr(legacy,'ANFISInferenceEngine',ANFIS)
    monkeypatch.setattr(legacy,'ATSCDecisionEngine',lambda anfis:anfis)
    monkeypatch.setattr(legacy.cv2,'VideoCapture',lambda *a:cap)
    def process(self):self.detector.track();return True
    monkeypatch.setattr(legacy.ReplaySession,'_process_single_frame',process)


@pytest.mark.parametrize('failure',['anfis','first_track'])
def test_failed_legacy_startup_disposes_models_before_reporting_release(tmp_path,monkeypatch,ownership,failure):
    import src.ui.session_manager as legacy
    refs=[];fail=[True];cap=ReleaseCapture()
    inject_legacy_components(monkeypatch,refs,failure,fail,cap)
    config=tmp_path/'config.json';config.write_text('{}')
    manager=legacy.SessionManager()
    with pytest.raises(RuntimeError,match='failed') as caught:manager.start_session('fake',str(config),step_mode=True)
    # The raised error (and its traceback) stays referenced; no collector run is needed.
    assert refs and all(ref() is None for ref in refs),'failed startup still retains a model owner'
    failed=getattr(caught.value,'session_owner',None)
    assert failed is not None and failed.resources_released
    assert cap.releases==(1 if failure=='first_track' else 0)
    assert manager.active_session is None and ownership.owner is None
    fail[0]=False
    replacement=manager.start_session('fake',str(config),step_mode=True)
    try:assert sum(ref() is not None for ref in refs)==2 and ownership.owner is replacement
    finally:manager.stop_session()
    assert replacement.resources_released and all(ref() is None for ref in refs)


def test_pending_legacy_startup_release_conflicts_before_any_constructor(tmp_path,monkeypatch,ownership):
    import src.ui.session_manager as legacy
    from src.ui.runtime_ownership import RuntimeConflict
    refs=[];fail=[True];cap=ReleaseCapture(failures=2);constructed=[]
    inject_legacy_components(monkeypatch,refs,'first_track',fail,cap)
    original=legacy.ReplaySession.__init__
    def counted(self,*a,**kw):constructed.append(self);original(self,*a,**kw)
    monkeypatch.setattr(legacy.ReplaySession,'__init__',counted)
    config=tmp_path/'config.json';config.write_text('{}')
    first=legacy.SessionManager();other=legacy.SessionManager()
    with pytest.raises(RuntimeError,match='first tracking failed') as caught:first.start_session('fake',str(config),step_mode=True)
    failed=caught.value.session_owner
    # Models are disposed, but the unconfirmed capture release keeps the owner registered.
    assert all(ref() is None for ref in refs) and not failed.resources_released and failed.release_error
    assert ownership.current_owner()==('legacy',failed) and first.active_session is failed
    fail[0]=False
    for manager in (first,other):
        with pytest.raises(RuntimeConflict):manager.start_session('fake',str(config),step_mode=True)
    observation=ObservationManager(tmp_path/'runs',session_factory=lambda *a:constructed.append('observation'),preflight=lambda *a:None)
    with pytest.raises(RuntimeConflict):observation.start({},before_handoff=lambda:None)
    assert constructed==[failed] and cap.releases==0 and not failed.resources_released
    replacement=first.start_session('fake',str(config),step_mode=True)
    try:
        assert failed.resources_released and cap.releases==1 and failed.release_error is None
        assert len(constructed)==2 and ownership.owner is replacement
    finally:first.stop_session()


def wait_until(predicate,timeout=2.):
    deadline=time.monotonic()+timeout
    while not predicate() and time.monotonic()<deadline:time.sleep(.01)
    return predicate()


class ReleaseWrapper:
    """Native capture whose release raises while `failing`; counts confirmed releases."""
    def __init__(self,cap,failing=False):self.cap=cap;self.failing=failing;self.releases=0
    def release(self):
        if self.failing:raise OSError('native capture release failed')
        self.releases+=1;self.cap.release()
    def __getattr__(self,name):return getattr(self.cap,name)


def counted_detector(refs):
    import weakref
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
    return Detector


def assert_pending_after_release_failure(session,message):
    with pytest.raises(RuntimeError):session.stop(timeout=.05)
    status=session.lifecycle()
    assert message in status['release_error'] and status['terminal_state']=='releasing'
    assert not status['resources_released'] and not status['finalized']


def test_capture_release_failure_still_releases_other_capture_and_models_then_stop_completes(video_factory,tmp_path):
    refs=[];session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=counted_detector(refs),anfis_factory=FakeANFIS)
    first=session.pair.system;second=session.pair.acquisition
    first.cap=failing=ReleaseWrapper(first.cap,failing=True);second.cap=counted=ReleaseWrapper(second.cap)
    try:
        session.control('stop');assert wait_until(lambda:session.release_error is not None)
        assert second.resources_released and counted.releases==1 and not first.resources_released
        assert not session.detectors and session.decision is None and all(ref() is None for ref in refs)
        assert_pending_after_release_failure(session,'native capture release failed')
        assert session.lifecycle()['capture_release_pending']
        failing.failing=False;session.stop(timeout=2)
        status=session.lifecycle()
        assert status['resources_released'] and status['finalized'] and status['release_error'] is None
        assert failing.releases==1 and counted.releases==1
    finally:failing.failing=False;session.stop(timeout=2)


class LiveCapture:
    def __init__(self,failing=False):self.failing=failing;self.releases=0;self.opened=True
    def isOpened(self):return self.opened
    def set(self,*a):return True
    def read(self):time.sleep(.005);return False,None
    def release(self):
        if self.failing:raise OSError('native capture release failed')
        self.releases+=1;self.opened=False


def test_live_capture_release_failure_still_releases_other_camera_and_models(video_factory,tmp_path,monkeypatch):
    import src.perception.frame_grabber as grabber
    hooked=[];monkeypatch.setattr(threading,'excepthook',hooked.append)
    caps=[LiveCapture(failing=True),LiveCapture()];opened=iter(caps)
    monkeypatch.setattr(grabber.cv2,'VideoCapture',lambda *a:next(opened))
    data=run_payload(video_factory);data['source_mode']='live'
    refs=[];session=ObservationSession(data,tmp_path,detector_factory=counted_detector(refs),anfis_factory=FakeANFIS)
    try:
        session.control('stop');assert wait_until(lambda:session.release_error is not None)
        assert_pending_after_release_failure(session,'native capture release failed')
        assert caps[1].releases==1 and session.pair.acquisition.resources_released
        assert not session.pair.system.resources_released and session.lifecycle()['capture_release_pending']
        assert not session.detectors and session.decision is None and all(ref() is None for ref in refs)
        caps[0].failing=False;session.stop(timeout=2)
        assert session.lifecycle()['resources_released'] and session.lifecycle()['finalized']
        assert [cap.releases for cap in caps]==[1,1] and not hooked
    finally:caps[0].failing=False;session.stop(timeout=2)


def test_detector_close_failure_still_closes_other_detectors_then_stop_completes(video_factory,tmp_path):
    import weakref
    refs=[];closed=[];failing=[True]
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);self.index=len(refs);refs.append(weakref.ref(self))
        def close(self):
            if self.index==0 and failing[0]:raise OSError('detector close failed')
            closed.append(self.index)
    session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=Detector,anfis_factory=FakeANFIS)
    try:
        session.control('stop');assert wait_until(lambda:session.release_error is not None)
        assert closed==[1] and refs[1]() is None and session.decision is None
        assert refs[0]() is not None and list(session.detectors)==['system']
        assert session.pair.resources_released
        assert_pending_after_release_failure(session,'detector close failed')
        failing[0]=False;session.stop(timeout=2)
        assert closed==[1,0] and all(ref() is None for ref in refs)
        assert session.lifecycle()['resources_released'] and session.lifecycle()['finalized']
    finally:failing[0]=False;session.stop(timeout=2)


def test_blocked_prediction_delays_model_disposal_while_captures_are_cleaned_up(video_factory,tmp_path):
    entered=threading.Event();release=threading.Event()
    refs=[];session=ObservationSession(run_payload(video_factory),tmp_path,detector_factory=counted_detector(refs),anfis_factory=FakeANFIS)
    def evaluate(s,c):entered.set();assert release.wait(5);return ActuationDecision(c,35)
    session.engine.predictions.evaluate=evaluate
    first=session.pair.system;second=session.pair.acquisition
    first.cap=failing=ReleaseWrapper(first.cap,failing=True);second.cap=counted=ReleaseWrapper(second.cap)
    try:
        session.predict('blocked');assert entered.wait(1)
        session.control('stop')
        assert wait_until(lambda:second.resources_released and session.release_error is not None)
        time.sleep(.1)
        assert session.lifecycle()['prediction_release_pending'] and counted.releases==1
        assert len(session.detectors)==2 and session.decision is not None and all(ref() is not None for ref in refs)
        release.set()
        assert wait_until(lambda:not session.detectors and session.decision is None)
        assert all(ref() is None for ref in refs)
        assert_pending_after_release_failure(session,'native capture release failed')
        failing.failing=False;session.stop(timeout=2)
        assert session.lifecycle()['resources_released'] and session.lifecycle()['finalized']
        assert failing.releases==1 and counted.releases==1
    finally:release.set();failing.failing=False;session.stop(timeout=2)


def test_legacy_shutdown_cannot_report_released_while_an_admitted_start_registers(monkeypatch,ownership):
    # sys.settrace inserts a scheduling point right after the closing check; it demonstrates the
    # interleaving rather than proving an uninstrumented race.
    import inspect,sys
    import src.ui.session_manager as legacy
    from src.ui.runtime_ownership import RuntimeConflict
    entered=threading.Event();release=threading.Event();made=[];errors=[];reports=[]
    class Replay(LegacyOwner):
        def __init__(self,**kw):super().__init__(**kw);made.append(self)
    monkeypatch.setattr(legacy,'ReplaySession',Replay)
    manager=legacy.SessionManager();code=legacy.SessionManager.start_session.__code__
    lines,first=inspect.getsourcelines(legacy.SessionManager.start_session)
    target=first+next(i for i,line in enumerate(lines) if 'Legacy manager is shutting down' in line)+1
    def trace(frame,event,arg):
        if frame.f_code is code and event=='line' and frame.f_lineno==target and not entered.is_set():
            entered.set();assert release.wait(3)
        return trace
    def start():
        sys.settrace(trace)
        try:manager.start_session('unused','unused')
        except Exception as error:errors.append(error)
        finally:sys.settrace(None)
    starter=threading.Thread(target=start);starter.start();assert entered.wait(1)
    closer=threading.Thread(target=lambda:reports.append(manager.shutdown(timeout=2)));closer.start()
    try:
        closer.join(.2);early=list(reports)
        assert not early or not early[0]['resources_released'],f'shutdown reported {early} before an admitted start registered'
    finally:release.set();starter.join(3);closer.join(3)
    # Either shutdown's closing decision rejected the start, or the start won and shutdown stopped it.
    assert reports and all(isinstance(error,RuntimeConflict) for error in errors)
    if reports[0]['resources_released']:assert all(session.resources_released for session in made)
    assert manager.active_session is None or manager.active_session.resources_released
