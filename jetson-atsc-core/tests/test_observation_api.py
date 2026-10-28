import io,json,zipfile
from functools import partial
from fastapi.testclient import TestClient
import pytest
from src.ui.app import app
import src.ui.observation_api as api
from src.ui.observation_session import ObservationManager,ObservationSession
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload


@pytest.fixture
def client(tmp_path,monkeypatch):
    manager=ObservationManager(tmp_path/"runs",tmp_path/"profiles",partial(ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
    monkeypatch.setattr(api,"observations",manager)
    with TestClient(app) as client:yield client
    if manager.current:manager.current.stop()


def test_two_camera_api_prediction_conflicts_and_export(client,video_factory):
    data=run_payload(video_factory)
    start=client.post("/api/observation/start",json=data)
    assert start.status_code==200,start.text
    run_id=start.json()["run_id"]
    a=client.get("/stream/observation/system?limit=1")
    b=client.get("/stream/observation/acquisition?limit=1")
    assert a.status_code==200 and b.status_code==200 and a.content!=b.content
    capture={"run_id":run_id,"request_id":"click-1"}
    first=client.post("/api/observation/predict",json=capture)
    assert first.status_code==200,first.text
    assert client.post("/api/observation/predict",json=capture).status_code==200
    assert client.post("/api/observation/predict",json={**capture,"request_id":"another"}).status_code==409
    assert client.post("/api/observation/predict",json={**capture,"run_id":"wrong"}).status_code==422
    export=client.get("/api/observation/export")
    assert export.status_code==200
    with zipfile.ZipFile(io.BytesIO(export.content)) as bundle:
        assert json.loads(bundle.read("manifest.json"))["operation"]=="observation"
    stop=client.post("/api/observation/control",json={"run_id":run_id,"action":"stop"})
    assert stop.status_code==200


def test_calibration_save_and_snapshot_are_role_specific(client,video_factory):
    data=run_payload(video_factory)
    a=client.post("/api/observation/calibration",json=data["system"])
    b=client.post("/api/observation/calibration",json=data["acquisition"])
    assert a.status_code==200 and b.status_code==200
    profiles=client.get("/api/observation/profiles").json()["profiles"]
    assert {p["calibration"]["role"] for p in profiles}=={"system","acquisition"}
    snapshot=client.get("/api/observation/snapshot",params={"role":"acquisition","source":data["acquisition"]["source"]})
    assert snapshot.status_code==200 and snapshot.headers["content-type"]=="image/jpeg"
    invalid={**data["acquisition"],"signal_roi":{"xmin":0,"ymin":0,"xmax":1000,"ymax":1000}}
    assert client.post("/api/observation/calibration",json=invalid).status_code==422


def test_unsupported_operation_and_invalid_source_rejected(client,video_factory):
    data=run_payload(video_factory)
    data["policy"]["operation"]="control"
    assert client.post("/api/observation/start",json=data).status_code==422
    data["policy"]["operation"]="observation";data["acquisition"]["source"]="missing.mp4"
    assert client.post("/api/observation/start",json=data).status_code==422


def test_static_replacement_failure_preserves_observation_and_legacy(client,video_factory,monkeypatch):
    data=run_payload(video_factory)
    stopped=[];monkeypatch.setattr(api.legacy_manager,'stop_session',lambda:stopped.append(True))
    assert client.post('/api/observation/start',json=data).status_code==200
    old=api.observations.current;stopped.clear()
    invalid={**data,'acquisition':{**data['acquisition'],'source':'missing.mp4'}}
    assert client.post('/api/observation/start',json=invalid).status_code==422
    assert api.observations.current is old and old.state=='paused'
    assert not stopped


def test_manager_reserves_before_preflight_and_serializes_handoff(tmp_path,video_factory):
    import threading
    from src.ui.observation_session import ObservationConflict
    entered=threading.Event();release=threading.Event();events=[]
    class Session:
        def __init__(self,payload,root):self.run_id='fake';self.resources_released=False
        def stop(self,**kw):events.append('old_stopped');self.resources_released=True
    def preflight(payload,root):entered.set();assert release.wait(5);events.append('preflight')
    manager=ObservationManager(tmp_path,session_factory=Session,preflight=preflight)
    old=Session({},tmp_path);manager.current=old
    errors=[]
    def start():
        try:manager.start({},before_handoff=lambda:events.append('legacy_stopped'))
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=start);thread.start()
    try:
        assert entered.wait(1)
        with pytest.raises(ObservationConflict):manager.start({})
        assert not events and manager.require() is old
    finally:release.set();thread.join(2)
    assert not errors and events==['preflight','old_stopped','legacy_stopped']
    manager.current.stop()


def test_stop_timeout_retains_owner_and_posthandoff_failure_is_explicit(tmp_path):
    from src.ui.observation_session import ObservationConflict,ReplacementFailed
    class Owner:
        run_id='old';resources_released=False
        def stop(self,**kw):raise RuntimeError('blocked')
    built=[]
    def factory(*args):built.append(True);raise RuntimeError('GPU acquisition failed')
    manager=ObservationManager(tmp_path,session_factory=factory,preflight=lambda *a:None)
    old=Owner();manager.current=old
    with pytest.raises(ObservationConflict,match='blocked'):manager.start({})
    assert manager.current is old and not built
    old.resources_released=True;old.stop=lambda **kw:None
    with pytest.raises(ReplacementFailed,match='old.*ended.*replacement failed'):manager.start({})
    assert manager.current is old and built==[True]
    assert manager.last_start['previous_run_id']=='old' and manager.last_start['previous_ended']
    assert manager.last_start['replacement_state']=='failed'


def test_static_preflight_never_opens_capture_or_model(video_factory,tmp_path,monkeypatch):
    import src.ui.observation_session as module
    data=run_payload(video_factory)
    def forbidden(*a,**kw):raise AssertionError('resource allocation during static preflight')
    monkeypatch.setattr(module.cv2,'VideoCapture',forbidden)
    monkeypatch.setattr(module,'DetectorTracker',forbidden)
    monkeypatch.setattr(module,'ANFISInferenceEngine',forbidden)
    module.ObservationSession.preflight(data,tmp_path)
    for mutate in (lambda p:p['policy'].update(max_gap_sec=-1),
                   lambda p:p['acquisition'].update(signal_roi={'xmin':0,'ymin':0,'xmax':999,'ymax':999}),
                   lambda p:p['acquisition'].update(source=p['system']['source'])):
        from copy import deepcopy
        invalid=deepcopy(data);mutate(invalid)
        with pytest.raises(ValueError):module.ObservationSession.preflight(invalid,tmp_path)


@pytest.mark.parametrize('failure',['models','storage'])
def test_static_resource_failure_preserves_existing_owner(video_factory,tmp_path,monkeypatch,failure):
    import src.ui.observation_session as module
    data=run_payload(video_factory);stopped=[]
    class Owner:
        run_id='healthy';resources_released=False
        def stop(self,**kw):stopped.append(True)
    manager=ObservationManager(tmp_path/'runs');old=Owner();manager.current=old
    if failure=='models':monkeypatch.setattr(module,'ROOT',tmp_path/'missing-model-root')
    else:
        def unavailable(*a,**kw):raise OSError('storage unavailable')
        monkeypatch.setattr(module.tempfile,'TemporaryFile',unavailable)
    with pytest.raises((ValueError,OSError)):manager.start(data,before_handoff=lambda:stopped.append('legacy'))
    assert manager.current is old and not stopped


def test_concurrent_http_starts_return_conflict_before_second_factory(client,video_factory,monkeypatch):
    import threading
    entered=threading.Event();release=threading.Event();responses=[]
    manager=api.observations;original=manager._preflight
    def preflight(data):entered.set();assert release.wait(5);return original(data)
    monkeypatch.setattr(manager,'_preflight',preflight)
    data=run_payload(video_factory)
    thread=threading.Thread(target=lambda:responses.append(client.post('/api/observation/start',json=data)))
    thread.start()
    try:
        assert entered.wait(1)
        conflict=client.post('/api/observation/start',json=data)
        assert conflict.status_code==409
        assert manager.current is None
    finally:release.set();thread.join(3)
    assert len(responses)==1 and responses[0].status_code==200


def test_distinct_managers_cannot_allocate_a_second_observation_owner(tmp_path):
    from src.ui.observation_session import ObservationConflict
    made=[]
    class Session:
        def __init__(self,*args):self.run_id='owner';self.resources_released=False;made.append(self)
        def stop(self,**kw):self.resources_released=True
    first=ObservationManager(tmp_path/'a',session_factory=Session,preflight=lambda *a:None)
    second=ObservationManager(tmp_path/'b',session_factory=Session,preflight=lambda *a:None)
    owner=first.start({})
    try:
        with pytest.raises(ObservationConflict,match='owner'):second.start({})
        assert len(made)==1 and first.current is owner
    finally:owner.stop()


# Phase B corrections: previews participate in the process-wide owner/reservation.
import threading
import time
from types import SimpleNamespace
import numpy as np
import src.ui.observation_session as observation_module
from src.ui.runtime_ownership import RuntimeConflict,RuntimeOwnership


@pytest.fixture
def ownership(monkeypatch):
    import src.ui.session_manager as legacy
    shared=RuntimeOwnership()
    monkeypatch.setattr(observation_module,'runtime_ownership',shared);monkeypatch.setattr(legacy,'runtime_ownership',shared)
    return shared


class PreviewOwner:
    def __init__(self,*a,**kw):
        self.run_id='owner';self.resources_released=False;self.state='paused'
        self.calibrations={r:SimpleNamespace(payload={'source':s}) for r,s in (('system','0'),('acquisition','1'))}
    def stop(self,**kw):self.resources_released=True
    def jpeg(self,role,clean=False):return b'cached-'+role.encode()


class PreviewCapture:
    def __init__(self,opened=True,frame=True,error=None,gate=None):
        self.opened=opened;self.frame=frame;self.error=error;self.gate=gate;self.releases=0;self.entered=threading.Event()
    def isOpened(self):return self.opened
    def read(self):
        self.entered.set()
        if self.gate is not None:assert self.gate.wait(3)
        if self.error:raise self.error
        return (True,np.zeros((24,32,3),np.uint8)) if self.frame else (False,None)
    def release(self):self.releases+=1


def capture_spy(monkeypatch,make=PreviewCapture):
    opened=[]
    def capture(source):opened.append((source,make()));return opened[-1][1]
    monkeypatch.setattr(observation_module.cv2,'VideoCapture',capture)
    return opened


@pytest.mark.parametrize('case',['reserved_start','other_manager'])
def test_preview_never_opens_a_source_reserved_or_owned_elsewhere(tmp_path,monkeypatch,ownership,case):
    opened=capture_spy(monkeypatch);entered=threading.Event();release=threading.Event();errors=[]
    manager=ObservationManager(tmp_path,session_factory=PreviewOwner,preflight=lambda *a:None)
    starter=None
    if case=='reserved_start':
        def factory(*a):entered.set();assert release.wait(3);return PreviewOwner()
        manager.session_factory=factory
        def start():
            try:manager.start({},before_handoff=lambda:None)
            except Exception as error:errors.append(error)
        starter=threading.Thread(target=start);starter.start();assert entered.wait(1)
    else:ownership.claim('observation',PreviewOwner())
    try:
        if case=='reserved_start':
            for source in ('0','5'):
                with pytest.raises(RuntimeConflict):manager.preview('system',source,source_mode='live')
        else:
            assert manager.preview('system','0',source_mode='live')==b'cached-system'
            with pytest.raises(RuntimeConflict):manager.preview('acquisition','0',source_mode='live')
        assert opened==[]
    finally:
        release.set()
        if starter:starter.join(2)
        if manager.current:manager.current.stop()
    assert not errors


def test_blocked_preview_capture_excludes_observation_and_legacy_starts(tmp_path,monkeypatch,ownership):
    import src.ui.session_manager as legacy
    gate=threading.Event();opened=capture_spy(monkeypatch,lambda:PreviewCapture(gate=gate));made=[];results=[]
    class Replay(PreviewOwner):
        def __init__(self,**kw):super().__init__();made.append('legacy')
    monkeypatch.setattr(legacy,'ReplaySession',Replay)
    manager=ObservationManager(tmp_path,session_factory=lambda *a:made.append('observation') or PreviewOwner(),preflight=lambda *a:None)
    thread=threading.Thread(target=lambda:results.append(manager.preview('system','2',source_mode='live')));thread.start()
    try:
        deadline=time.monotonic()+1
        while not opened and time.monotonic()<deadline:time.sleep(.005)
        cap=opened[0][1];assert cap.entered.wait(1)
        with pytest.raises(RuntimeConflict):manager.start({},before_handoff=lambda:None)
        with pytest.raises(RuntimeConflict):legacy.SessionManager().start_session('unused','unused')
        assert not made and cap.releases==0
    finally:gate.set();thread.join(2)
    assert results and cap.releases==1
    session=manager.start({},before_handoff=lambda:None)
    assert made==['observation'];session.stop()


def test_concurrent_previews_of_different_sources_both_proceed(tmp_path,monkeypatch,ownership):
    gate=threading.Event();opened=capture_spy(monkeypatch,lambda:PreviewCapture(gate=gate));results={}
    manager=ObservationManager(tmp_path,session_factory=PreviewOwner,preflight=lambda *a:None)
    threads=[threading.Thread(target=lambda r=r,s=s:results.update({r:manager.preview(r,s,source_mode='live')}))
             for r,s in (('system','2'),('acquisition','3'))]
    for thread in threads:thread.start()
    try:
        deadline=time.monotonic()+1
        while len(opened)<2 and time.monotonic()<deadline:time.sleep(.005)
        assert len(opened)==2 and all(cap.entered.wait(1) for _,cap in opened)
    finally:
        gate.set()
        for thread in threads:thread.join(2)
    assert set(results)=={'system','acquisition'} and all(results.values())
    assert [cap.releases for _,cap in opened]==[1,1]


@pytest.mark.parametrize('failure',['open_raises','closed','no_frame','read_raises'])
def test_failed_preview_releases_capture_once_and_frees_its_lease(tmp_path,monkeypatch,ownership,failure):
    caps=[]
    def capture(source):
        if failure=='open_raises':raise OSError('capture backend failed')
        caps.append(PreviewCapture(opened=failure!='closed',frame=failure!='no_frame',
                                   error=OSError('read failed') if failure=='read_raises' else None));return caps[-1]
    monkeypatch.setattr(observation_module.cv2,'VideoCapture',capture)
    manager=ObservationManager(tmp_path,session_factory=PreviewOwner,preflight=lambda *a:None)
    with pytest.raises((ValueError,OSError)):manager.preview('system','2',source_mode='live')
    assert [cap.releases for cap in caps]==([] if failure=='open_raises' else [1])
    session=manager.start({},before_handoff=lambda:None)
    assert ownership.owner is session;session.stop()


def test_existing_owner_cached_preview_is_served_even_during_replacement(tmp_path,monkeypatch,ownership):
    opened=capture_spy(monkeypatch);entered=threading.Event();release=threading.Event();errors=[]
    def preflight(*a):entered.set();assert release.wait(3)
    manager=ObservationManager(tmp_path,session_factory=PreviewOwner,preflight=preflight)
    old=PreviewOwner();manager.current=old;ownership.claim('observation',old)
    assert manager.preview('acquisition','1',source_mode='live')==b'cached-acquisition'
    def start():
        try:manager.start({},before_handoff=lambda:None)
        except Exception as error:errors.append(error)
    thread=threading.Thread(target=start);thread.start()
    try:
        assert entered.wait(1)
        assert manager.preview('system','0',source_mode='live')==b'cached-system'
        with pytest.raises(RuntimeConflict):manager.preview('system','5',source_mode='live')
        assert opened==[]
    finally:release.set();thread.join(2)
    assert not errors and old.resources_released;manager.current.stop()


def metrics_client_manager(tmp_path,monkeypatch,made):
    class Detector(FakeDetector):
        def __init__(self,*a,**kw):super().__init__(*a,**kw);made.append(self)
    manager=ObservationManager(tmp_path/'runs',tmp_path/'profiles',partial(ObservationSession,detector_factory=Detector,anfis_factory=FakeANFIS))
    monkeypatch.setattr(api,'observations',manager)
    return manager


@pytest.mark.parametrize('lane_metrics',[{'min_stopped_observations':'invalid-integer'},{'stopped_speed_threshold_mps':'not-a-speed'}])
def test_static_metrics_configuration_error_is_422_before_any_handoff(tmp_path,video_factory,monkeypatch,ownership,lane_metrics):
    from copy import deepcopy
    import src.experiment.sources as sources
    made=[];captures=[];stopped=[];manager=metrics_client_manager(tmp_path,monkeypatch,made)
    monkeypatch.setattr(api.legacy_manager,'stop_session',lambda:stopped.append(True))
    data=run_payload(video_factory);invalid=deepcopy(data);invalid['system']['lane_metrics']=lane_metrics
    try:
        with TestClient(app) as client:
            assert client.post('/api/observation/start',json=data).status_code==200
            healthy=manager.current;run_id=healthy.run_id;made.clear();stopped.clear()
            original=sources.cv2.VideoCapture
            monkeypatch.setattr(sources.cv2,'VideoCapture',lambda *a:captures.append(a) or original(*a))
            response=client.post('/api/observation/start',json=invalid)
            monkeypatch.setattr(sources.cv2,'VideoCapture',original)
            assert response.status_code==422,response.text
            assert manager.current is healthy and healthy.run_id==run_id and healthy.state=='paused'
            assert not healthy.lifecycle()['stop_requested'] and ownership.owner is healthy
            assert not stopped and not made and not captures
    finally:
        if manager.current:manager.current.stop()


def test_valid_legacy_lane_metrics_still_replace_a_healthy_run(tmp_path,video_factory,monkeypatch,ownership):
    from copy import deepcopy
    made=[];stopped=[];manager=metrics_client_manager(tmp_path,monkeypatch,made)
    monkeypatch.setattr(api.legacy_manager,'stop_session',lambda:stopped.append(True))
    data=run_payload(video_factory);valid=deepcopy(data)
    valid['system']['lane_metrics']={'min_stopped_observations':3,'stopped_speed_threshold_mps':.8,
        'stop_line_tripwire_y_pixel':200,'pcu_weights':{'motorcycle':.4,'car':1.,'truck':1.6,'bus':1.6}}
    try:
        with TestClient(app) as client:
            assert client.post('/api/observation/start',json=data).status_code==200
            old=manager.current;stopped.clear()
            response=client.post('/api/observation/start',json=valid)
            assert response.status_code==200,response.text
            assert old.resources_released and manager.current is not old and stopped==[True]
            metrics=manager.current.metrics
            assert metrics.min_stopped_observations==3 and metrics.stopped_speed_threshold==.8
    finally:
        if manager.current:manager.current.stop()


@pytest.mark.parametrize('method',['measured_dimensions','ruler'])
def test_typed_scale_binding_profile_roundtrip(client,video_factory,method):
    data=run_payload(video_factory)['system']
    data.update(scale_evidence={'method':method,'reference':'synthetic survey, not field evidence'},binding={'reviewed':True})
    if method=='ruler':data['ruler']={'p1':[30,180],'p2':[280,180],'length_m':6}
    response=client.post('/api/observation/calibration',json=data)
    assert response.status_code==200,response.text
    saved=response.json()
    assert saved['scale']['state']=='verified'
    profiles=client.get('/api/observation/profiles').json()['profiles']
    loaded=next(p['calibration'] for p in profiles if p['profile_id']==saved['profile_id'])
    again=client.post('/api/observation/calibration',json=loaded)
    assert again.status_code==200,again.text
    assert again.json()['revision']==saved['revision']
    assert loaded['scale_evidence']==data['scale_evidence']
    assert loaded['binding']['geometry_digest']


def test_saving_old_profile_does_not_claim_measured_scale(client,video_factory):
    data=run_payload(video_factory)['system']
    response=client.post('/api/observation/calibration',json=data)
    assert response.status_code==200,response.text
    assert response.json()['scale']['state']=='review_required'
    assert 'scale_evidence' not in response.json()['calibration']


def test_measured_dimensions_must_be_explicit_in_api_input(client,video_factory):
    data=run_payload(video_factory)['system']
    data.pop('road_length_m')
    data.update(scale_evidence={'method':'measured_dimensions','reference':'synthetic fixture'},binding={'reviewed':True})
    response=client.post('/api/observation/calibration',json=data)
    assert response.status_code==422,response.text


def test_reviewed_binding_survives_typed_numeric_normalization(client,video_factory):
    from src.experiment.calibration import CameraCalibration
    data=run_payload(video_factory)['system']
    data.update(scale_evidence={'method':'measured_dimensions','reference':'synthetic dimensions'},binding={'reviewed':True})
    saved=CameraCalibration(data).payload  # integer coordinates from a local profile
    response=client.post('/api/observation/calibration',json=saved)
    assert response.status_code==200,response.text
    assert response.json()['scale']['state']=='verified'


@pytest.mark.parametrize('measured',[False,True])
def test_saved_profiles_start_through_typed_api_without_scale_promotion(client,video_factory,measured):
    data=run_payload(video_factory)
    for role in ('system','acquisition'):
        if measured:data[role].update(scale_evidence={'method':'measured_dimensions','reference':'synthetic profile roundtrip'},binding={'reviewed':True})
        response=client.post('/api/observation/calibration',json=data[role])
        assert response.status_code==200,response.text
        data[role]=response.json()['calibration']
    started=client.post('/api/observation/start',json=data)
    assert started.status_code==200,started.text
    telemetry=client.get('/api/observation/telemetry').json()
    assert telemetry['scale_trust']==('verified' if measured else 'unverified')
    assert telemetry['qualification_policy']=='motion-phase-scale-v2'
    capture=client.post('/api/observation/predict',json={'run_id':started.json()['run_id'],'request_id':'saved-profile'})
    assert capture.status_code==200,capture.text
    for evidence in capture.json()['scale_evidence'].values():
        assert evidence['state']==('verified' if measured else 'review_required')
