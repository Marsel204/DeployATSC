import pytest
from fastapi.testclient import TestClient
import src.ui.app as application
import src.ui.observation_api as api
from scripts import launch_ui


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(application.app.state,'operation','observation',raising=False)
    with TestClient(application.app) as client:yield client


@pytest.mark.parametrize('method,path,payload',[
    ('post','/api/session/start',{'video_path':'unused','config_path':'unused'}),
    ('post','/api/session/control',{'action':'play'}),
    ('get','/api/session/telemetry',None),('get','/stream/video_feed?limit=1',None)])
def test_observation_mode_disables_legacy_runtime_only(client,method,path,payload,monkeypatch):
    def forbidden(*a,**kw):raise AssertionError('Legacy runtime entered in observation mode')
    monkeypatch.setattr(application.manager,'start_session',forbidden)
    if path.startswith('/stream/'):
        class Session:
            state='running';resources_released=True
            def get_latest_frame(self):return b'fake jpeg'
            def stop(self,**kw):pass
        monkeypatch.setattr(application.manager,'active_session',Session())
    response=client.request(method,path,json=payload)
    assert response.status_code==409
    assert client.get('/api/sample_videos').status_code==200
    assert client.get('/api/configs').status_code==200
    assert client.get('/api/observation/bootstrap').status_code==200


def test_host_and_origin_are_checked_for_reads_and_mutations(client):
    for path in ('/','/api/observation/telemetry','/stream/observation/system?limit=1'):
        assert client.get(path,headers={'Host':'untrusted.example'}).status_code==400
        assert client.get(path,headers={'Origin':'https://untrusted.example'}).status_code==403
    assert client.post('/api/observation/control',json={'run_id':'x','action':'stop'},
                       headers={'Origin':'null'}).status_code==403
    preflight=client.options('/api/observation/start',headers={'Origin':'https://untrusted.example',
                            'Access-Control-Request-Method':'POST'})
    assert preflight.status_code in (400,403)
    assert preflight.headers.get('access-control-allow-origin')!='*'
    assert client.get('/',headers={'Host':'127.0.0.1:8000','Origin':'http://127.0.0.1:8000'}).status_code==200


def test_actual_launcher_selects_observation_mode_and_refuses_lan(monkeypatch):
    monkeypatch.setattr(launch_ui.uvicorn,'run',lambda *a,**kw:None)
    monkeypatch.setattr(application.app.state,'operation','legacy',raising=False)
    launch_ui.serve_observation()
    assert application.app.state.operation=='observation'
    with pytest.raises(ValueError,match='loopback'):launch_ui.serve_observation(host='0.0.0.0')


def test_explicit_legacy_mode_keeps_routes_compatible(monkeypatch):
    monkeypatch.setattr(application.app.state,'operation','legacy',raising=False)
    monkeypatch.setattr(application.manager,'active_session',None)
    with TestClient(application.app) as client:
        assert client.post('/api/session/control',json={'action':'pause'}).json()=={'status':'no_active_session'}
        assert client.get('/api/session/telemetry').json()['status']=='idle'
        assert client.get('/api/sample_videos').status_code==200


def test_shared_reservation_rejects_legacy_start_while_observation_owns_native_work(tmp_path):
    from src.ui.runtime_ownership import runtime_ownership,RuntimeConflict
    from src.ui.session_manager import SessionManager
    previous=(runtime_ownership.kind,runtime_ownership.owner)
    class Owner:resources_released=False
    try:
        runtime_ownership.claim('observation',Owner())
        with pytest.raises(RuntimeConflict):SessionManager().start_session('unused','unused')
    finally:runtime_ownership.claim(*previous)


@pytest.mark.parametrize('host',['evil@localhost','localhost:invalid','localhost/ignored','localhost#ignored'])
def test_malformed_host_is_not_an_allowlisted_host(client,host):
    assert client.get('/',headers={'Host':host}).status_code==400
