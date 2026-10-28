"""Independent supported-entrypoint counterexamples; injected captures/models only."""
from functools import partial
import threading
import pytest
import cv2
from src.ui import observation_session as obs, session_manager as legacy
from src.ui.runtime_ownership import RuntimeConflict
from test_observation_lifecycle import ownership
from test_observation_sessions import FakeDetector, FakeANFIS
from test_lifecycle_final_corrections import Capture, inputs, legacy_inputs


def fail_named_thread(monkeypatch, prefix, failure):
    if failure=='construct':
        original=threading.Thread.__init__
        def construct(self,*a,**kw):
            if kw.get('name','').startswith(prefix):raise RuntimeError('cannot construct cleanup thread')
            return original(self,*a,**kw)
        monkeypatch.setattr(threading.Thread,'__init__',construct)
    else:
        original=threading.Thread.start
        def start(self):
            if self.name.startswith(prefix):raise RuntimeError('cannot start cleanup thread')
            return original(self)
        monkeypatch.setattr(threading.Thread,'start',start)


@pytest.mark.parametrize('failure',['construct','start'])
@pytest.mark.parametrize('entrypoint',['observation','legacy'])
def test_preview_public_retry_recovers_when_retry_thread_creation_fails(tmp_path,monkeypatch,ownership,failure,entrypoint):
    class Pending(Capture):
        repair=False
        def release(self):
            self.attempts+=1
            if not self.repair:raise OSError('original preview release failure')
            self.releases+=1;self.active=False
    cap=Pending();monkeypatch.setattr(cv2,'VideoCapture',lambda *a:cap)
    source=tmp_path/'preview.mp4';source.write_bytes(b'injected media')
    manager=obs.ObservationManager(tmp_path/'runs') if entrypoint=='observation' else legacy.SessionManager()
    with monkeypatch.context() as patch:
        fail_named_thread(patch,'Preview-release',failure)
        with pytest.raises(Exception) as caught:
            if entrypoint=='observation':manager.preview('system',str(source))
            else:manager.get_video_snapshot(str(source))
    pending=ownership.preview_owners()[0]
    try:
        assert pending.cap is cap and cap.active and ownership.leases==1
        assert str(caught.value)=='original preview release failure'
        with pytest.raises(RuntimeConflict):obs.ObservationManager(tmp_path/'blocked',preflight=lambda *a:None).start({})
        cap.repair=True
        assert not pending.thread or not pending.thread.is_alive()
        before=cap.attempts
        r1=ownership.stop_previews(timeout=.05);r2=manager.shutdown(timeout=.05)
        print('PREVIEW',entrypoint,failure,'error=',str(caught.value),'release_attempts=',before,'->',cap.attempts,'public_retry=',r1,'shutdown=',r2)
        assert r1 and r2['resources_released'] and not cap.active, 'Public retry never restarts the absent cleanup worker'
    finally:
        cap.repair=True
        if not pending.resources_released:pending.finish()  # test-only disposal after proving public retries failed


@pytest.mark.parametrize('failure',['construct','start'])
@pytest.mark.parametrize('mode',['replay','live'])
@pytest.mark.parametrize('index',[0,1])
def test_failed_partial_owner_stays_registered_when_cleanup_thread_creation_fails(tmp_path,monkeypatch,ownership,failure,mode,index):
    from src.perception.frame_grabber import FrameGrabber
    caps=[]
    class Pending(Capture):
        repair=False
        def __init__(self,i):super().__init__();self.index=i
        def isOpened(self):
            if self.index==index:raise ValueError('original acquisition error')
            return super().isOpened()
        def release(self):
            self.attempts+=1
            if self.index==index and not self.repair:raise OSError('unconfirmed partial capture release')
            self.releases+=1;self.active=False
    def capture(*a):
        cap=Pending(len(caps));caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    monkeypatch.setattr(FrameGrabber,'_build_gstreamer_pipeline',lambda self:None)
    factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    manager=obs.ObservationManager(tmp_path/'runs',session_factory=factory,shutdown_timeout=.01)
    data=inputs(tmp_path,mode)
    with monkeypatch.context() as patch:
        fail_named_thread(patch,'Observation-cleanup-',failure)
        with pytest.raises(obs.ReplacementFailed) as caught:manager.start(data,before_handoff=lambda:None)
    cap=caps[index]
    initial=caught.value.__cause__
    if not hasattr(initial,'session_owner'):initial=initial.__context__
    partial_owner=initial.resource_owner
    try:
        assert caught.value.__cause__ is initial and str(initial)=='original acquisition error'
        report=manager.shutdown(timeout=.01)
        constructed=[]
        class Replacement:
            resources_released=True;run_id='replacement'
        replacement=obs.ObservationManager(tmp_path/'replacement',preflight=lambda *a:None,session_factory=lambda *a:(constructed.append(True) or Replacement()))
        # This is a supported manager start, with a constructor sentinel, no arbitrary registry mutation.
        try:replacement.start({},before_handoff=lambda:None)
        except RuntimeConflict:pass
        print('PARTIAL',mode,failure,'error=',str(caught.value),'current=',manager.current,'capture_active=',cap.active,'shutdown=',report,'replacement_constructed=',bool(constructed))
        assert ownership.owner is manager.current and manager.current is not None and not report['resources_released'] and not constructed, 'Failed cleanup scheduling erased session-owner registration and admitted replacement'
    finally:
        cap.repair=True
        if manager.current:manager.current.stop(timeout=2)
        else:partial_owner.close()
    assert all(not item.active and item.releases==1 for item in caps)
    replacement.start({},before_handoff=lambda:None)
    assert constructed==[True]
