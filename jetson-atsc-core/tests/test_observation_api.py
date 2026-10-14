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
