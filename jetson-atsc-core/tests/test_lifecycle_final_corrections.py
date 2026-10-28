"""C1–C4: real owners/constructors with injected native captures and models only."""
from functools import partial
import json
import os
import threading
import time
import weakref
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.experiment.calibration import CameraCalibration
from src.perception.frame_grabber import FrameGrabber
from src.ui import observation_api as api, observation_session as obs, session_manager as legacy
from src.ui.runtime_ownership import RuntimeConflict
from test_observation_calibration import payload
from test_observation_lifecycle import ownership
from test_observation_sessions import FakeANFIS, FakeDetector


class Capture:
    def __init__(self):
        self.active=True;self.attempts=0;self.releases=0
    def isOpened(self):return self.active
    def get(self,prop):
        return {cv2.CAP_PROP_FPS:25,cv2.CAP_PROP_FRAME_WIDTH:320,
                cv2.CAP_PROP_FRAME_HEIGHT:240,cv2.CAP_PROP_FRAME_COUNT:50}.get(prop,0)
    def set(self,*a):return True
    def read(self):return True,np.zeros((240,320,3),np.uint8)
    def release(self):
        self.attempts+=1;self.releases+=1;self.active=False


def inputs(tmp_path,mode='replay'):
    data={'source_mode':mode,'step_mode':True,'policy':{'debounce_frames':1}}
    for role in ('system','acquisition'):
        data[role]=payload(role);path=tmp_path/(role+'.mp4');path.write_bytes(b'injected media')
        data[role]['source']=str(path)
    return data


def legacy_inputs(tmp_path,monkeypatch,detector=FakeDetector,anfis=FakeANFIS):
    cfg=CameraCalibration(payload()).legacy_metrics_config()
    cfg['traffic_light_monitor']={'lamp_roi_pixel':{'xmin':10,'xmax':40,'ymin':10,'ymax':40}}
    config=tmp_path/'legacy.json';config.write_text(json.dumps(cfg))
    monkeypatch.setattr(legacy,'LensRectifier',lambda *a:SimpleNamespace(rectify=lambda frame:frame))
    monkeypatch.setattr(legacy,'DetectorTracker',detector);monkeypatch.setattr(legacy,'ANFISInferenceEngine',anfis)
    return str(tmp_path/'system.mp4'),str(config)


@pytest.mark.parametrize('spelling',['relative','absolute'])
@pytest.mark.parametrize('role',['system','acquisition'])
def test_legacy_http_preview_identity_releases_before_reopen(tmp_path,monkeypatch,ownership,spelling,role):
    caps=[]
    def capture(*a):cap=Capture();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    source,config=legacy_inputs(tmp_path,monkeypatch)
    manager=legacy.SessionManager();session=manager.start_session(source,config,step_mode=True)
    previews=obs.ObservationManager(tmp_path/'runs');monkeypatch.setattr(api,'observations',previews)
    app=FastAPI();app.include_router(api.router)
    alias=os.path.relpath(source) if spelling=='relative' else source
    try:
        with TestClient(app) as client:
            response=client.get('/api/observation/snapshot',params={'role':role,'source':alias})
            assert response.status_code==409 and len(caps)==1
            assert session.latest_jpeg and not session.resources_released
            manager.stop_session()
            response=client.get('/api/observation/snapshot',params={'role':role,'source':alias})
            assert response.status_code==200 and len(caps)==2
            assert [cap.releases for cap in caps]==[1,1]
    finally:manager.stop_session()


def test_preview_rechecks_a_legacy_owner_under_lease(tmp_path,monkeypatch,ownership):
    source,config=legacy_inputs(tmp_path,monkeypatch);caps=[]
    def capture(*a):cap=Capture();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    manager=legacy.SessionManager();previews=obs.ObservationManager(tmp_path/'runs')
    original=previews._owned_preview;calls=[]
    def lookup(*args):
        calls.append(True);result=original(*args)
        if len(calls)==1:manager.start_session(source,config,step_mode=True)
        return result
    monkeypatch.setattr(previews,'_owned_preview',lookup)
    try:
        with pytest.raises(RuntimeConflict):previews.preview('system',os.path.relpath(source))
        assert len(caps)==1 and len(calls)==2
    finally:manager.stop_session()


def assert_starts_excluded(manager,data,source,config,cap_count,caps):
    with pytest.raises(RuntimeConflict):manager.start(data,before_handoff=lambda:None)
    with pytest.raises(RuntimeConflict):legacy.SessionManager().start_session(source,config,step_mode=True)
    assert len(caps)==cap_count


@pytest.mark.parametrize('failure',['transient','persistent'])
@pytest.mark.parametrize('read_fails',[False,True])
def test_failed_preview_retains_owner_until_paced_confirmed_retry(tmp_path,monkeypatch,ownership,failure,read_fails):
    allow=threading.Event();retry=threading.Event();done=threading.Event();times=[]
    class Pending(Capture):
        def read(self):
            if read_fails:raise ValueError('preview read failed')
            return super().read()
        def release(self):
            self.attempts+=1;times.append(time.monotonic())
            if self.attempts>1:
                retry.set()
                if failure=='transient':assert allow.wait(3)
            if not allow.is_set():raise OSError('preview release unconfirmed')
            self.releases+=1;self.active=False;done.set()
    cap=Pending();caps=[]
    def capture(*a):caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    data=inputs(tmp_path);source,config=legacy_inputs(tmp_path,monkeypatch)
    manager=obs.ObservationManager(tmp_path/'runs',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
    try:
        with pytest.raises((ValueError,OSError)):manager.preview('system',source)
        assert_starts_excluded(manager,data,source,config,1,caps)
        pending=ownership.preview_owners()
        assert len(pending)==1 and pending[0].cap is cap and not pending[0].resources_released
        assert retry.wait(1)
        assert not ownership.stop_previews(timeout=.12)
        assert 2<=cap.attempts<=6 and times[1]-times[0]>=.015
        assert cap.releases==0 and cap.active
        assert_starts_excluded(manager,data,source,config,1,caps)
        allow.set()
        # Automatic retry completes without requiring a new start/stop to dispose the capture.
        assert done.wait(2)
        assert ownership.stop_previews(timeout=1)
        assert cap.releases==1 and ownership.leases==0 and not ownership.preview_owners()
        assert ownership.stop_previews(timeout=0) and cap.releases==1
        monkeypatch.setattr(cv2,'VideoCapture',lambda *a:Capture())
        replacement=manager.start(data,before_handoff=lambda:None)
        replacement.stop(timeout=2)
    finally:
        allow.set()
        if hasattr(ownership,'stop_previews'):assert ownership.stop_previews(timeout=2)
        if manager.current:manager.current.stop(timeout=2)


@pytest.mark.parametrize('blocked',['read','release'])
def test_preview_blocked_native_call_remains_owned_during_both_shutdowns(tmp_path,monkeypatch,ownership,blocked):
    entered=threading.Event();allow=threading.Event();results=[];errors=[]
    class Blocked(Capture):
        def read(self):
            if blocked=='read':entered.set();assert allow.wait(3)
            return super().read()
        def release(self):
            if blocked=='release':entered.set();assert allow.wait(3)
            super().release()
    cap=Blocked();caps=[]
    def capture(*a):caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    manager=obs.ObservationManager(tmp_path/'runs');other=legacy.SessionManager()
    def preview():
        try:results.append(manager.preview('system','0',source_mode='live'))
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=preview);thread.start()
    try:
        assert entered.wait(1)
        for report in (manager.shutdown(timeout=.01),other.shutdown(timeout=.01)):
            assert not report['resources_released'] and not report['finalized']
            assert report['preview_release_pending']
        assert cap.active and cap.releases==0
        with pytest.raises(RuntimeConflict):obs.ObservationManager(tmp_path/'other').start({})
    finally:allow.set();thread.join(2)
    assert results and not errors and not thread.is_alive() and cap.releases==1
    assert manager.shutdown(timeout=1)['resources_released']
    assert other.shutdown(timeout=1)['resources_released']


def raise_chain(native,style):
    try:native.low()
    except ValueError as inner:
        if style=='implicit':raise RuntimeError('startup wrapper')
        if style=='group':raise ExceptionGroup('startup wrapper',[inner]) from None
        outer=RuntimeError('startup wrapper')
        if style=='cycle':inner.__cause__=outer
        raise outer from inner


@pytest.mark.parametrize('style',['explicit','implicit','cycle','group'])
@pytest.mark.parametrize('path',['legacy_track','observation_track','observation_constructor'])
def test_exception_graph_disposes_models_with_retained_error(tmp_path,monkeypatch,ownership,style,path):
    refs=[];caps=[]
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
        def low(self):raise ValueError('native tracking failure')
        def track(self,*a,**kw):raise_chain(self,style)
    class ANFIS(FakeANFIS):
        def __init__(self,*a,**kw):
            refs.append(weakref.ref(self))
            if path=='observation_constructor':raise_chain(self,style)
        def low(self):raise ValueError('native model construction failure')
    def capture(*a):cap=Capture();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    if path=='legacy_track':
        source,config=legacy_inputs(tmp_path,monkeypatch,Detector,ANFIS);manager=legacy.SessionManager()
        with pytest.raises((RuntimeError,ExceptionGroup),match='startup wrapper') as caught:manager.start_session(source,config,step_mode=True)
        failed=caught.value.session_owner
        assert manager.active_session is None and ownership.owner is None
    else:
        factory=partial(obs.ObservationSession,detector_factory=FakeDetector if path=='observation_constructor' else Detector,anfis_factory=ANFIS)
        manager=obs.ObservationManager(tmp_path/'runs',session_factory=factory)
        with pytest.raises(obs.ReplacementFailed,match='startup wrapper') as caught:manager.start(inputs(tmp_path),before_handoff=lambda:None)
        failed=manager.current;failed.stop(timeout=2)
    # Keep the entire original graph alive; do not call gc.collect(). Locations remain inspectable.
    error=caught.value if path=='legacy_track' else caught.value.__cause__
    graph=[error];seen=set();locations=[]
    while graph:
        item=graph.pop()
        if id(item) in seen:continue
        seen.add(id(item));tb=item.__traceback__
        while tb:locations.append(tb.tb_frame.f_code.co_name);tb=tb.tb_next
        graph.extend(e for e in (item.__cause__,item.__context__,*getattr(item,'exceptions',())) if e is not None)
    assert 'low' in locations and 'raise_chain' in locations
    assert failed.resources_released and all(ref() is None for ref in refs)
    assert all(cap.releases==1 for cap in caps)


@pytest.mark.parametrize('mode',['replay','live'])
@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('failure',['open_raises','closed','metadata'])
def test_partial_leaf_failure_keeps_every_native_owner_until_public_stop(tmp_path,monkeypatch,ownership,mode,index,failure):
    allow=threading.Event();caps=[];attempted=threading.Event()
    class Partial(Capture):
        def __init__(self,i):super().__init__();self.index=i
        def isOpened(self):
            if self.index==index:
                if failure=='open_raises':raise ValueError('original open-check error')
                if failure=='closed':return False
            return super().isOpened()
        def get(self,prop):
            if self.index==index and failure=='metadata':raise ValueError('original metadata error')
            return super().get(prop)
        def set(self,*args):
            if self.index==index and failure=='metadata':raise ValueError('original metadata error')
            return True
        def release(self):
            self.attempts+=1
            if not allow.is_set() and self.index==index:
                attempted.set();raise OSError('cleanup error must not mask acquisition')
            self.releases+=1;self.active=False
    def capture(*a):cap=Partial(len(caps));caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    monkeypatch.setattr(FrameGrabber,'_build_gstreamer_pipeline',lambda self:None)
    data=inputs(tmp_path,mode)
    if mode=='live':
        for i,role in enumerate(('system','acquisition')):data[role]['source']=str(i+20)
    source,config=legacy_inputs(tmp_path,monkeypatch)
    manager=obs.ObservationManager(tmp_path/'runs',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS),shutdown_timeout=.01)
    try:
        with pytest.raises(obs.ReplacementFailed) as caught:manager.start(data,before_handoff=lambda:None)
        assert 'cleanup error' not in str(caught.value)
        assert ('Cannot open' if failure=='closed' else 'original') in str(caught.value)
        failed=manager.current
        assert attempted.wait(1) and len(caps)==index+1
        assert not failed.resources_released and ownership.owner is failed
        assert failed.pair is not None and len(failed.pair._sources())==len(caps)
        assert any(leaf.cap is caps[index] for leaf in failed.pair._sources())
        assert_starts_excluded(manager,data,source,config,index+1,caps)
        with pytest.raises(RuntimeError):failed.stop(timeout=.01)
        assert not manager.shutdown(timeout=.01)['resources_released']
        allow.set();failed.stop(timeout=2);failed.stop(timeout=0)
        assert failed.resources_released and not failed.lifecycle()['capture_release_pending']
        assert all(cap.releases==1 and not cap.active for cap in caps)
        assert failed.pair.health()[('system','acquisition')[index]]['state']!='release_pending'
        # A fresh manager can take the released registry owner; the old one is shut down.
        monkeypatch.setattr(cv2,'VideoCapture',lambda *a:Capture())
        replacement=obs.ObservationManager(tmp_path/'replacement',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS)).start(inputs(tmp_path),before_handoff=lambda:None)
        replacement.stop(timeout=2)
    finally:
        allow.set()
        if manager.current:manager.current.stop(timeout=2)


@pytest.mark.parametrize('mode',['replay','live'])
def test_second_leaf_failure_preserves_first_unconfirmed_cleanup(tmp_path,monkeypatch,ownership,mode):
    allow=threading.Event();caps=[]
    class Partial(Capture):
        def __init__(self,i):super().__init__();self.index=i
        def isOpened(self):return self.index!=1
        def release(self):
            self.attempts+=1
            if self.index==0 and not allow.is_set():raise OSError('system release pending')
            self.releases+=1;self.active=False
    def capture(*a):cap=Partial(len(caps));caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture);monkeypatch.setattr(FrameGrabber,'_build_gstreamer_pipeline',lambda self:None)
    manager=obs.ObservationManager(tmp_path/'runs',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
    try:
        with pytest.raises(obs.ReplacementFailed,match='Cannot open acquisition'):manager.start(inputs(tmp_path,mode),before_handoff=lambda:None)
        failed=manager.current
        assert not failed.resources_released and failed.pair is not None and len(failed.pair._sources())==2
        assert [cap.active for cap in caps]==[True,False]
        allow.set();failed.stop(timeout=2)
        assert failed.resources_released and [cap.releases for cap in caps]==[1,1]
    finally:
        allow.set()
        if manager.current:manager.current.stop(timeout=2)


@pytest.mark.parametrize('api_name',['observation','legacy'])
def test_all_preview_entrypoints_retain_pending_capture_and_exclude_same_source(tmp_path,monkeypatch,ownership,api_name):
    allow=threading.Event();caps=[]
    class Pending(Capture):
        def release(self):
            self.attempts+=1
            if not allow.is_set():raise OSError('pending preview release')
            self.releases+=1;self.active=False
    def capture(*a):cap=Pending();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    source=tmp_path/'preview.mp4';source.write_bytes(b'injected media')
    preview_manager=obs.ObservationManager(tmp_path/'runs');old=legacy.SessionManager()
    try:
        preview=(lambda:preview_manager.preview('system',str(source))) if api_name=='observation' else (lambda:old.get_video_snapshot(str(source)))
        with pytest.raises(OSError,match='pending preview release'):preview()
        # Neither public preview spelling may reopen a source with unconfirmed release.
        with pytest.raises(RuntimeConflict):preview_manager.preview('system',os.path.relpath(source))
        with pytest.raises(RuntimeConflict):old.get_video_snapshot(str(source))
        assert len(caps)==1
        for report in (preview_manager.shutdown(timeout=.01),old.shutdown(timeout=.01)):
            assert report['preview_release_pending'] and not report['resources_released']
        allow.set();assert ownership.stop_previews(timeout=2)
        assert caps[0].releases==1
    finally:
        allow.set()
        # The frozen INPUT has no pending owner API; keep its red failure visible.
        if hasattr(ownership,'stop_previews'):assert ownership.stop_previews(timeout=2)


@pytest.mark.parametrize('style',['explicit','implicit'])
def test_chained_legacy_error_keeps_failed_capture_registered_until_public_stop(tmp_path,monkeypatch,ownership,style):
    refs=[];allow=threading.Event();fail=[True];caps=[]
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
        def low(self):raise ValueError('native tracking failure')
        def track(self,*a,**kw):
            if fail[0]:raise_chain(self,style)
            return super().track(*a,**kw)
    class Pending(Capture):
        def release(self):
            self.attempts+=1
            if not allow.is_set():raise OSError('pending legacy capture')
            self.releases+=1;self.active=False
    def capture(*a):cap=Pending();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    source,config=legacy_inputs(tmp_path,monkeypatch,Detector);manager=legacy.SessionManager()
    try:
        with pytest.raises(RuntimeError,match='startup wrapper') as caught:manager.start_session(source,config,step_mode=True)
        failed=caught.value.session_owner
        assert refs and all(ref() is None for ref in refs)
        assert ownership.owner is failed and manager.active_session is failed and not failed.resources_released
        with pytest.raises(RuntimeConflict):legacy.SessionManager().start_session(source,config,step_mode=True)
        observer=obs.ObservationManager(tmp_path/'runs',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
        with pytest.raises(RuntimeConflict):observer.start(inputs(tmp_path),before_handoff=lambda:None)
        with pytest.raises(RuntimeError):manager.stop_session(timeout=.01)
        assert len(caps)==1 and caps[0].active
        allow.set();manager.stop_session();assert failed.resources_released and caps[0].releases==1
        fail[0]=False;replacement=manager.start_session(source,config,step_mode=True)
        assert not replacement.resources_released and len(caps)==2
    finally:allow.set();manager.stop_session()


@pytest.mark.parametrize('mode',['replay','live'])
@pytest.mark.parametrize('index',[0,1])
def test_native_capture_constructor_failure_preserves_partial_pair(tmp_path,monkeypatch,ownership,mode,index):
    allow=threading.Event();caps=[];calls=[]
    class Pending(Capture):
        def release(self):
            self.attempts+=1
            if not allow.is_set():raise OSError('first capture release pending')
            self.releases+=1;self.active=False
    def capture(*a):
        calls.append(True)
        if len(calls)==index+1:raise ValueError('native constructor failed')
        cap=Pending();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture);monkeypatch.setattr(FrameGrabber,'_build_gstreamer_pipeline',lambda self:None)
    manager=obs.ObservationManager(tmp_path/'runs',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS),shutdown_timeout=.01)
    try:
        with pytest.raises(obs.ReplacementFailed,match='native constructor failed'):manager.start(inputs(tmp_path,mode),before_handoff=lambda:None)
        failed=manager.current
        assert failed is not None and failed.pair is not None and len(caps)==index
        if index:
            assert not failed.resources_released and not failed.pair.resources_released
            with pytest.raises(RuntimeError):failed.stop(timeout=.01)
        allow.set();failed.stop(timeout=2)
        assert failed.resources_released and all(cap.releases==1 for cap in caps)
    finally:
        allow.set()
        if manager.current:manager.current.stop(timeout=2)


def test_legacy_snapshot_http_reports_owned_source_conflict(tmp_path,monkeypatch,ownership):
    import src.ui.app as application
    source,config=legacy_inputs(tmp_path,monkeypatch)
    from pathlib import Path
    Path(source).write_bytes(b'injected media')
    caps=[]
    def capture(*a):cap=Capture();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    manager=legacy.SessionManager();manager.start_session(source,config,step_mode=True)
    monkeypatch.setattr(application,'manager',manager)
    try:
        client=TestClient(application.app)
        response=client.get('/api/video/snapshot',params={'video_path':source})
        assert response.status_code==409 and len(caps)==1
        manager.stop_session()
        response=client.get('/api/video/snapshot',params={'video_path':source})
        assert response.status_code==200 and len(caps)==2 and caps[1].releases==1
    finally:manager.stop_session()


def test_pending_preview_allows_a_different_source_preview(tmp_path,monkeypatch,ownership):
    allow=threading.Event();caps=[]
    class Pending(Capture):
        def release(self):
            self.attempts+=1
            if not allow.is_set():raise OSError('preview still owned')
            self.releases+=1;self.active=False
    def capture(*a):cap=Pending() if not caps else Capture();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    manager=obs.ObservationManager(tmp_path/'runs')
    try:
        with pytest.raises(OSError,match='still owned'):manager.preview('system','20',source_mode='live')
        assert manager.preview('acquisition','21',source_mode='live')
        assert len(caps)==2 and caps[1].releases==1 and caps[0].active
        assert ownership.leases==1 and ownership.preview_owners()[0].cap is caps[0]
        with pytest.raises(RuntimeConflict):legacy.SessionManager().start_session('unused','unused')
    finally:
        allow.set()
        if hasattr(ownership,'stop_previews'):assert ownership.stop_previews(timeout=2)


@pytest.mark.parametrize('index',[0,1])
@pytest.mark.parametrize('failure',['gstreamer_release','reader_start'])
def test_live_backend_startup_failure_retains_leaf_owner(tmp_path,monkeypatch,ownership,index,failure):
    allow=threading.Event();caps=[]
    class Partial(Capture):
        def __init__(self,i):super().__init__();self.index=i
        def isOpened(self):
            return False if failure=='gstreamer_release' and self.index==index else super().isOpened()
        def read(self):return False,None
        def release(self):
            self.attempts+=1
            if self.index==index and not allow.is_set():raise OSError('pipeline release pending')
            self.releases+=1;self.active=False
    def capture(*a):cap=Partial(len(caps));caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    if failure=='reader_start':
        monkeypatch.setattr(FrameGrabber,'_build_gstreamer_pipeline',lambda self:None)
        start=threading.Thread.start
        def start_reader(thread):
            if thread.name==('Grabber-System' if index==0 else 'Grabber-Acquisition'):raise ValueError('reader startup error')
            return start(thread)
        monkeypatch.setattr(threading.Thread,'start',start_reader)
    data=inputs(tmp_path,'live')
    for i,role in enumerate(('system','acquisition')):data[role]['source']=str(i+20)
    manager=obs.ObservationManager(tmp_path/'runs',session_factory=partial(obs.ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS),shutdown_timeout=.01)
    try:
        with pytest.raises(obs.ReplacementFailed,match='pipeline release pending' if failure=='gstreamer_release' else 'reader startup error'):
            manager.start(data,before_handoff=lambda:None)
        failed=manager.current
        assert not failed.resources_released and failed.pair is not None
        assert len(caps)==(index+1 if failure=='gstreamer_release' else 2)
        assert len(failed.pair._sources())==len(caps)
        assert failed.pair._sources()[index].cap is caps[index]
        allow.set();failed.stop(timeout=2);failed.stop(timeout=0)
        assert failed.resources_released and all(cap.releases==1 and not cap.active for cap in caps)
    finally:
        allow.set()
        if manager.current:manager.current.stop(timeout=2)


@pytest.mark.parametrize('entrypoint',['observation','legacy'])
def test_shutdown_rejects_new_preview_allocation(tmp_path,monkeypatch,ownership,entrypoint):
    caps=[]
    def capture(*a):cap=Capture();caps.append(cap);return cap
    monkeypatch.setattr(cv2,'VideoCapture',capture)
    source=tmp_path/'shutdown.mp4';source.write_bytes(b'injected media')
    manager=obs.ObservationManager(tmp_path/'runs') if entrypoint=='observation' else legacy.SessionManager()
    assert manager.shutdown(timeout=0)['resources_released']
    with pytest.raises(RuntimeConflict,match='shutting down'):
        if entrypoint=='observation':manager.preview('system',str(source))
        else:manager.get_video_snapshot(str(source))
    assert not caps


@pytest.mark.parametrize('style',['explicit','implicit'])
def test_legacy_worker_error_graph_does_not_retain_disposed_detector(tmp_path,monkeypatch,ownership,style):
    refs=[];errors=[];reported=threading.Event();fail=[False]
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);refs.append(weakref.ref(self))
        def low(self):raise ValueError('native tracking failure')
        def track(self,*a,**kw):
            if fail[0]:raise_chain(self,style)
            return super().track(*a,**kw)
    def retain_error(args):errors.append(args.exc_value);reported.set()
    monkeypatch.setattr(threading,'excepthook',retain_error)
    monkeypatch.setattr(cv2,'VideoCapture',lambda *a:Capture())
    source,config=legacy_inputs(tmp_path,monkeypatch,Detector);manager=legacy.SessionManager()
    session=manager.start_session(source,config,step_mode=True)
    try:
        fail[0]=True;session.step();assert reported.wait(2);session._thread.join(1)
        assert errors and str(errors[0])=='startup wrapper'
        assert session.resources_released and refs and all(ref() is None for ref in refs)
    finally:manager.stop_session()
