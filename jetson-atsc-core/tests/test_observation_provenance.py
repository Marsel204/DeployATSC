"""Hardware-free regressions through real calibration/session/API/storage paths."""
import hashlib
import io
import json
import zipfile
from copy import deepcopy
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.ui.app import app
import src.ui.observation_api as api
from src.ui.observation_session import ObservationManager,ObservationSession
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
from test_observation_calibration import payload

SECRET_SOURCE='rtsp://camera-user:private-pass@camera.local/private-stream?token=private-token#private-fragment'


@pytest.mark.parametrize('source',[SECRET_SOURCE,'rtspsrc location="'+SECRET_SOURCE+'" password=pipeline-secret ! decodebin ! appsink'])
def test_profile_storage_and_preview_use_public_handles(tmp_path,monkeypatch,source):
    manager=ObservationManager(tmp_path/'runs',tmp_path/'profiles')
    monkeypatch.setattr(api,'observations',manager)
    seen=[]
    monkeypatch.setattr(manager,'preview',lambda role,source,*args:seen.append(source) or b'jpeg')
    with TestClient(app) as client:
        data=payload();data['source']=source
        saved=client.post('/api/observation/calibration',json=data)
        assert saved.status_code==200,saved.text
        public=saved.json()['calibration']['source']
        assert public.startswith('source_') and public not in source
        roundtrip=client.post('/api/observation/calibration',json=saved.json()['calibration'])
        assert roundtrip.status_code==200,roundtrip.text
        assert roundtrip.json()['revision']==saved.json()['revision']
        listed=client.get('/api/observation/profiles').text
        preview=client.get('/api/observation/snapshot',params={'role':'system','source':public})
        assert preview.status_code==200 and seen==[public]
        serialized=saved.text+listed+''.join(p.read_text() for p in (tmp_path/'profiles').glob('*'))+str(preview.request.url)
        for secret in ('private-pass','private-token','private-stream','pipeline-secret'):
            assert secret not in serialized
        assert hashlib.sha256(source.encode()).hexdigest() not in serialized


def test_api_errors_and_bootstrap_do_not_echo_credentials(tmp_path,monkeypatch):
    manager=ObservationManager(tmp_path/'runs',tmp_path/'profiles')
    monkeypatch.setattr(api,'observations',manager)
    monkeypatch.setattr(api,'launch_settings',{'source_sys':SECRET_SOURCE})
    def broken(*args):raise RuntimeError('backend rejected '+SECRET_SOURCE)
    monkeypatch.setattr(manager,'preview',broken)
    with TestClient(app) as client:
        outputs=[client.get('/api/observation/bootstrap').text,
                 client.post('/api/observation/calibration',json={**payload(),'source':SECRET_SOURCE,'scale_evidence':{'method':'invalid','reference':SECRET_SOURCE}}).text,
                 client.post('/api/observation/snapshot',json={'role':'system','source':SECRET_SOURCE}).text]
        for output in outputs:
            assert 'private-pass' not in output and 'private-token' not in output
        assert client.post('/api/observation/snapshot',json={'role':'system','source':SECRET_SOURCE}).status_code==500


class ExecutionDetector(FakeDetector):
    def __init__(self,*args,**kwargs):
        from types import SimpleNamespace
        super().__init__(*args,**kwargs)
        self.device='requested:cuda'
        self.conf_threshold=kwargs['conf_threshold'];self.imgsz=kwargs['imgsz'];self.iou_threshold=.5
        self.model=SimpleNamespace(predictor=SimpleNamespace(device='synthetic:cpu',args=SimpleNamespace(conf=.28,iou=.5,imgsz=640,half=False,tracker='bytetrack.yaml')))


def test_real_session_persists_execution_media_clock_and_capture_diagnostics(tmp_path,video_factory):
    from test_decision_engine import synthetic_anfis
    data=run_payload(video_factory);data['policy']['offset_sec']=.04
    for role in ('system','acquisition'):
        data[role].update(scale_evidence={'method':'measured_dimensions','reference':'synthetic dimensions'},binding={'reviewed':True})
    session=ObservationSession(data,tmp_path/'runs',detector_factory=ExecutionDetector,anfis_factory=lambda *a,**kw:synthetic_anfis(10.004),sysfs_root=tmp_path/"fake-sysfs")
    try:
        session.predict('actual-call');session.engine.predictions.await_idle()
        telemetry=session.telemetry()
        assert telemetry['predictions'][0]['inference_diagnostics']['raw_green_sec']==10.004
        assert telemetry['scale_trust']=='verified'
        execution=telemetry['execution']
        assert execution['detectors']['system']['effective_device']=='synthetic:cpu'
        assert execution['detectors']['system']['requested_device']=='requested:cuda'
        assert execution['anfis']['providers']==['SyntheticExecutionProvider']
        assert execution['detectors']['system']['effective_settings']['conf']==.28
        clocks=telemetry['clock_provenance']
        assert clocks['acquisition']['offset_sec']==.04
        assert clocks['system']['last_processed']['mapped_timestamp']==telemetry['metrics']['timestamp']
        assert clocks['system']['timestamp_origin'] in ('presentation_time','validated_fps_fallback')
        session.stop()
        with zipfile.ZipFile(io.BytesIO(session.export())) as bundle:
            manifest=json.loads(bundle.read('manifest.json'));evidence=json.loads(bundle.read('evidence.json'))
        assert evidence['execution']==execution
        assert evidence['clock_provenance']['acquisition']['offset_sec']==.04
        media=manifest['media_identity']['system']
        assert media['sha256']==hashlib.sha256(Path(data['system']['source']).read_bytes()).hexdigest()
        assert media['bytes']==Path(data['system']['source']).stat().st_size
        assert manifest['execution']['anfis']['providers']==['SyntheticExecutionProvider']
    finally:session.stop()


def test_file_clock_fallback_is_per_packet_and_persisted_in_health(tmp_path,monkeypatch):
    import cv2
    from src.experiment.sources import FileSource
    path=tmp_path/'synthetic.mp4';path.write_bytes(b'not decoded by fake')
    class Capture:
        times=iter([0.,0.,200.])
        def isOpened(self):return True
        def get(self,key):
            return next(self.times) if key==cv2.CAP_PROP_POS_MSEC else {cv2.CAP_PROP_FPS:10,cv2.CAP_PROP_FRAME_WIDTH:20,cv2.CAP_PROP_FRAME_HEIGHT:20,cv2.CAP_PROP_FRAME_COUNT:3}[key]
        def read(self):return True,np.zeros((20,20,3),np.uint8)
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',lambda *a:Capture())
    source=FileSource(str(path),'acquisition',offset=.05)
    try:
        packets=[source.read() for _ in range(3)]
        assert [p.timestamp_origin for p in packets]==['presentation_time','validated_fps_fallback','presentation_time']
        assert source.health()['timestamp_fallback_count']==1
        assert source.health()['offset_sec']==.05
        assert packets[1].timestamp==pytest.approx(.05)
    finally:source.close()


def test_live_credentials_never_reach_stored_or_public_provenance(tmp_path,monkeypatch):
    import time
    import src.ui.observation_session as module
    from src.experiment.sources import FramePacket
    seen=[]
    class Pair:
        def __init__(self,system,acquisition,*args,**kwargs):
            seen.extend([system,acquisition]);self.origin=time.monotonic();self.first=True;self.ended=False;self.has_backlog=False;self.resources_released=False
        def step(self):
            if not self.first:return []
            self.first=False
            return [FramePacket(role,1,self.origin,0,np.full((240,320,3),color,np.uint8),'host_monotonic_receive') for role,color in [('system',40),('acquisition',(0,0,255))]]
        def health(self):return {r:{'state':'ready','age_sec':0,'image_size':[320,240],'timestamp_origin':'host_monotonic_receive','error':'backend '+SECRET_SOURCE} for r in ('system','acquisition')}
        def close(self):self.ended=True;self.resources_released=True
    monkeypatch.setattr(module,'LivePair',Pair)
    # Catch any attempt to hash a credential-bearing source or representation.
    original=hashlib.sha256
    def guarded(data=b'',*args,**kwargs):
        assert b'private-pass' not in data and b'private-token' not in data
        return original(data,*args,**kwargs)
    monkeypatch.setattr(hashlib,'sha256',guarded)
    data={'system':payload(),'acquisition':payload('acquisition'),'source_mode':'live','step_mode':True,'policy':{'debounce_frames':1}}
    data['system']['source']=SECRET_SOURCE;data['acquisition']['source']='/dev/video1'
    session=ObservationSession(data,tmp_path/'runs',detector_factory=ExecutionDetector,anfis_factory=FakeANFIS)
    try:
        session.predict('secret-safe');session.engine.predictions.await_idle()
        row=session.engine.predictions.rows()[0]
        assert row['provenance']['execution']['detectors']['system']['effective_device']=='synthetic:cpu'
        assert row['provenance']['clock_provenance']['system']['timestamp_origin']=='host_monotonic_receive'
        telemetry=json.dumps(session.telemetry())
        session.stop()
        stored=''.join(p.read_text() for p in (tmp_path/'runs'/session.run_id).glob('*.json*'))
        with zipfile.ZipFile(io.BytesIO(session.export())) as bundle:export=''.join(bundle.read(n).decode() for n in bundle.namelist())
        assert seen==[SECRET_SOURCE,'/dev/video1']
        for output in (telemetry,stored,export):
            for secret in ('private-pass','private-token','private-stream'):
                assert secret not in output
    finally:session.stop()


def test_public_handle_expires_without_private_credentials_on_restart(tmp_path):
    from src.common.source_identity import resolve_source
    from src.experiment.calibration import CameraCalibration
    data=payload();data['source']='source_'+'f'*32
    with pytest.raises(ValueError,match='re-enter the source locally'):CameraCalibration(data)
    assert resolve_source('/dev/video0')=='/dev/video0'


def test_short_url_values_do_not_corrupt_profile_hashes_or_geometry():
    from src.experiment.calibration import CameraCalibration
    data=payload();data['source']='rtsp://x:y@camera.local/feed?token=1'
    data.update(scale_evidence={'method':'measured_dimensions','reference':'synthetic fixture'},binding={'reviewed':True})
    calibration=CameraCalibration(data)
    assert CameraCalibration(calibration.payload).revision==calibration.revision
    assert calibration.scale_status['state']=='verified'


def test_legacy_export_redacts_public_copy_without_rewriting_history(tmp_path):
    from src.experiment.export import ExperimentStore
    root=tmp_path/'old';root.mkdir()
    manifest={'schema_version':1,'run_id':'old','sources':{'system':SECRET_SOURCE}}
    evidence={'schema_version':1,'predictions':[{'request_id':'old','status':'matched','error':'source '+SECRET_SOURCE}], 'vehicles':[],'comparisons':[]}
    for name,data in [('manifest.json',manifest),('evidence.json',evidence)]:
        (root/name).write_text(json.dumps(data))
    before={p.name:p.read_bytes() for p in root.iterdir()}
    with zipfile.ZipFile(io.BytesIO(ExperimentStore.open(root).export())) as bundle:
        output=''.join(bundle.read(n).decode() for n in bundle.namelist())
        assert 'private-pass' not in output and 'private-token' not in output
        assert json.loads(bundle.read('evidence.json'))['predictions'][0].get('qualification') is None
    assert {p.name:p.read_bytes() for p in root.iterdir()}==before


def test_profile_filename_uses_redacted_name(tmp_path):
    manager=ObservationManager(tmp_path/'runs',tmp_path/'profiles')
    data=payload();data['source']=SECRET_SOURCE;data['name']=SECRET_SOURCE
    result=manager.save_calibration(data)
    assert 'private-pass' not in result['profile_id'] and 'private-token' not in result['profile_id']
    assert 'private-pass' not in str(list((tmp_path/'profiles').iterdir()))


def test_credentials_matching_domain_labels_do_not_rewrite_app_semantics():
    from src.experiment.calibration import CameraCalibration
    data=payload();data['source']='rtsp://system:RED@camera.local/feed?token=unknown'
    calibration=CameraCalibration(data)
    assert calibration.role=='system'
    assert calibration.scale_status['state']=='review_required'
    from src.common.source_identity import public_data
    safe=public_data({'phase':'RED','qualification':{'state':'unknown'},'error':'backend password RED'})
    assert safe['phase']=='RED' and safe['qualification']['state']=='unknown'
    assert 'RED' not in safe['error']
