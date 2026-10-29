import threading
import time
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from src.ui.observation_session import ObservationSession
from critical_path_session_oracle import OriginalSessionMethods
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload,wait_applied


def settled(session):
    for role in ('system','acquisition'):assert session.jpeg(role)


def test_stepped_preview_bytes_equal_original(video_factory,tmp_path):
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        for _ in range(8):
            settled(session)
            for role in ('system','acquisition'):
                visual=OriginalSessionMethods._render(session,role,session.clean_frames[role],[])
                if visual.shape[1]>session.preview_width:visual=cv2.resize(visual,(session.preview_width,round(visual.shape[0]*session.preview_width/visual.shape[1])),interpolation=cv2.INTER_LINEAR)
                assert session.jpeg(role)==cv2.imencode('.jpg',visual,[cv2.IMWRITE_JPEG_QUALITY,82])[1].tobytes()
            seq=session.engine.processed['acquisition']
            result=session.control('step');wait_applied(session,result['control_seq'])
            deadline=time.monotonic()+2
            while session.engine.processed['acquisition']==seq and time.monotonic()<deadline:time.sleep(.001)
        assert session.preview_encoded_frames>=16
    finally:session.stop()
    assert not session.preview_thread.is_alive() and session.resources_released


def test_render_job_freezes_boxes_labels_and_wait_text(video_factory,tmp_path,monkeypatch):
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        settled(session)
        v=SimpleNamespace(bbox=[20,30,80,100],track_id=7,class_name='car',stable_id=True,epoch=2)
        with session.engine.lock:session.engine.waits.active[(2,7)]={'row':{'red_wait_sec':2.34}}
        labels=session._preview_labels('acquisition',[v])
        expected=OriginalSessionMethods._render(session,'acquisition',session.clean_frames['acquisition'],[v])
        v.bbox[:]=[0,0,1,1];v.class_name='changed';session.engine.waits.active.clear()
        # Engine lock raises if the encoder drawing path tries to read mutable wait state.
        class Forbidden:
            def __enter__(self):raise AssertionError('encoder touched engine lock')
            def __exit__(self,*a):pass
        with monkeypatch.context() as patch:
            patch.setattr(session.engine,'lock',Forbidden())
            assert np.array_equal(session._render('acquisition',session.clean_frames['acquisition'],labels),expected)
        assert labels==(((20,30,80,100),'#7 car | red wait 2.3s'),)
    finally:session.stop()


def test_preview_coalescing_error_containment_and_bounded_wait(video_factory,tmp_path,monkeypatch):
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    entered=threading.Event();release=threading.Event();original=cv2.imencode
    try:
        settled(session)
        def blocked(*args,**kw):entered.set();assert release.wait(5);return original(*args,**kw)
        with monkeypatch.context() as patch:
            patch.setattr(cv2,'imencode',blocked)
            frame=session.clean_frames['acquisition'];packet=SimpleNamespace(role='acquisition',timestamp=0.)
            session._publish_frame(packet,frame,[]);assert entered.wait(2)
            for i in range(12):session._publish_frame(packet,np.full_like(frame,i),[])
            assert len(session.preview_jobs)<=2 and session.preview_coalesced_frames>=11
            tick=time.monotonic();session.applied_state='running';assert session.jpeg('acquisition');assert time.monotonic()-tick<.1
            session.applied_state='paused';tick=time.monotonic();assert session.jpeg('acquisition');assert time.monotonic()-tick<.35
            release.set()
        settled(session)
        with monkeypatch.context() as patch:
            patch.setattr(cv2,'imencode',lambda *a,**kw:(_ for _ in ()).throw(RuntimeError('synthetic encoding failure')))
            session._publish_frame(packet,frame,[]);session.jpeg('acquisition')
            assert session.preview_errors==1 and session.error is None
        session._publish_frame(packet,frame,[]);settled(session)
        assert session.telemetry()['preview_errors']==1
    finally:release.set();session.stop()


def test_stop_during_encode_releases_native_owners_then_joins(video_factory,tmp_path,monkeypatch):
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    entered=threading.Event();release=threading.Event();original=cv2.imencode
    try:
        settled(session)
        def blocked(*a,**kw):entered.set();assert release.wait(5);return original(*a,**kw)
        monkeypatch.setattr(cv2,'imencode',blocked)
        session._publish_frame(SimpleNamespace(role='acquisition',timestamp=0.),session.clean_frames['acquisition'],[])
        assert entered.wait(2)
        with pytest.raises(RuntimeError,match='stopping'):session.stop(timeout=.03)
        deadline=time.monotonic()+2
        while (session.detectors or not session.pair.resources_released) and time.monotonic()<deadline:time.sleep(.005)
        assert not session.detectors and session.pair.resources_released
        assert not session.resources_released and session.preview_thread.is_alive()
        release.set();session.stop(timeout=3)
        assert session.finalized and session.resources_released and not session.preview_thread.is_alive()
    finally:release.set();session.stop()


def test_fifty_start_stop_cycles_leave_no_preview_threads(video_factory,tmp_path):
    payload=run_payload(video_factory);before={t.ident for t in threading.enumerate()}
    for i in range(50):
        session=ObservationSession(payload,tmp_path/str(i),detector_factory=FakeDetector,anfis_factory=FakeANFIS)
        settled(session);session.stop()
        assert session.finalized and session.resources_released and not session.preview_thread.is_alive()
    assert not [t.name for t in threading.enumerate() if t.ident not in before and t.name.startswith('Observation-')]


def test_provenance_sampling_failure_cannot_bypass_measurement_seal(video_factory,tmp_path,monkeypatch):
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    def fail():raise OSError('synthetic final provenance read error')
    monkeypatch.setattr(session,'_read_execution_provenance',fail)
    session.stop()
    assert session.engine.closed and session.resources_released
    assert session.finalized and session.release_error is None
