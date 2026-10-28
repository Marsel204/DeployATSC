"""Independent Phase C expected-behavior probes; synthetic sources/backends only."""
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import threading
import zipfile

import numpy as np
import pytest

from src.experiment.calibration import CameraCalibration
from src.experiment.export import ExperimentStore
from src.ui.observation_api import CameraSettings
from src.ui.observation_session import ObservationSession
from test_observation_calibration import payload
from test_observation_sessions import FakeANFIS, FakeDetector


def test_switching_from_ruler_to_measured_dimensions_uses_measured_dimensions():
    """An existing ruler draft must not silently override the newly chosen method."""
    script = Path('src/ui/static/observation.js').read_text()
    handler = script[script.index('  async function saveCalibration()'):script.index('  async function loadProfile(')]
    data = payload()
    data.pop('rectification_path')
    data['rulerPoints'] = [[30, 180], [280, 180]]
    harness = r'''
const elements=new Map();
const $=id=>{if(!elements.has(id))elements.set(id,{value:'',checked:false,textContent:''});return elements.get(id);};
const copy=structuredClone;
let draft=DATA,modalRole='system',sent;
const camera={system:{}};
const closeCalibration=()=>{},loadProfiles=async()=>{},updateControls=()=>{},show=()=>{};
const json=async(url,payload)=>{sent=payload;return {profile_id:'new',calibration:payload};};
$('calibrationName').value='dimension-review';$('groundWidth').value='8';$('groundLength').value='40';$('rulerLength').value='6';
$('scaleMethod').value='measured_dimensions';$('scaleReference').value='synthetic measured 8 by 40 m';$('bindingReviewed').checked=true;
HANDLER
saveCalibration().then(()=>process.stdout.write(JSON.stringify(sent))).catch(e=>{console.error(e);process.exitCode=1;});
'''.replace('DATA', json.dumps(data)).replace('HANDLER', handler)
    result = subprocess.run(['node', '-'], input=harness, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr
    sent = json.loads(result.stdout)
    typed = CameraSettings.model_validate(sent).model_dump(mode='json', exclude_none=True)
    camera = CameraCalibration(typed)
    print(json.dumps({'method': camera.scale_status['method'], 'scale': camera.scale_status['state'],
                      'sent_ruler': sent.get('ruler'), 'declared_width': sent['road_width_m'],
                      'actual_width': camera.homography.ground_width,
                      'declared_length': sent['road_length_m'], 'actual_length': camera.homography.ground_length}))
    assert camera.scale_status['state'] == 'verified'
    assert camera.homography.ground_width == pytest.approx(8)
    assert camera.homography.ground_length == pytest.approx(40)


def test_api_calibration_dimensions_ignore_inactive_ruler_and_roundtrip():
    data = payload()
    data.update(scale_evidence={'method': 'measured_dimensions', 'reference': 'synthetic survey'},
                binding={'reviewed': True},
                ruler={'p1': [30, 180], 'p2': [280, 180], 'length_m': 6})
    camera = CameraCalibration(CameraSettings.model_validate(data).model_dump(mode='json', exclude_none=True))
    assert 'ruler' not in camera.payload
    assert camera.scale_status['state'] == 'verified'
    assert camera.homography.ground_width == pytest.approx(8)
    assert camera.homography.ground_length == pytest.approx(40)
    assert CameraCalibration(camera.payload).revision == camera.revision
    # Legacy unqualified rulers retain their diagnostic interpretation.
    data.pop('scale_evidence')
    data.pop('binding')
    legacy = CameraCalibration(data)
    assert legacy.scale_status['state'] == 'review_required'
    assert legacy.homography.measure_pixel_distance([30, 180], [280, 180]) == pytest.approx(6)


@pytest.mark.parametrize('observed_role', ['system', 'acquisition'])
def test_capture_clock_is_bound_to_frozen_system_input(tmp_path, monkeypatch, observed_role):
    """Capture after metrics admission, before image rendering, using real worker/capture/store."""
    from types import SimpleNamespace
    import src.ui.observation_session as module
    from src.experiment.sources import FramePacket
    entered, release = threading.Event(), threading.Event()

    class Pair:
        def __init__(self, *args, **kwargs):
            self.system = SimpleNamespace(ended=False, fps=25)
            self.acquisition = SimpleNamespace(ended=False, fps=25)
            self.batch = 0
            self.ended = False
            self.has_backlog = False
            self.resources_released = False
        def step(self):
            self.batch += 1
            timestamp = (self.batch - 1) / 25
            origin = 'presentation_time' if self.batch == 1 else 'validated_fps_fallback'
            return [FramePacket(role, self.batch, timestamp, timestamp,
                       np.full((240, 320, 3), color, np.uint8), origin)
                    for role, color in [('system', 40), ('acquisition', (0, 0, 255))]]
        def health(self):
            return {role: {'state': 'ready', 'image_size': [320, 240],
                          'timestamp_origin': 'presentation_time' if self.batch == 1 else 'validated_fps_fallback',
                          'timestamp_fallback_count': max(0, self.batch - 1)}
                    for role in ('system', 'acquisition')}
        def close(self):
            self.ended = self.resources_released = True

    monkeypatch.setattr(module, 'ReplayPair', Pair)
    render = ObservationSession._render
    def barrier(self, role, frame, tracks):
        # Camera 1 input is now the admitted preview frame (YOLO runs on the click).
        if role == observed_role and (self.camera1_latest or {}).get('timestamp') == .04:
            entered.set()
            assert release.wait(5), 'probe release timeout'
        return render(self, role, frame, tracks)
    monkeypatch.setattr(ObservationSession, '_render', barrier)
    settings = {'source_mode': 'replay', 'step_mode': True, 'policy': {'debounce_frames': 1}}
    for role in ('system', 'acquisition'):
        path = tmp_path / (role + '.mp4')
        path.write_bytes(b'fake pair never decodes these fixture bytes')
        settings[role] = payload(role)
        settings[role]['source'] = str(path)
    session = ObservationSession(settings, tmp_path / 'runs', detector_factory=FakeDetector, anfis_factory=FakeANFIS)
    try:
        session.control('step')
        assert entered.wait(3), 'worker did not reach packet admission barrier'
        session.predict('frozen-clock')
        session.engine.predictions.await_idle()
        row = session.engine.predictions.rows()[0]
        frozen = deepcopy(row['provenance']['clock_provenance'][observed_role]['last_processed'])
        print(json.dumps({'frozen_input_timestamp': row['snapshot']['timestamp'], 'frozen_clock': frozen,
                          'stored_clock': session.store.lookup('frozen-clock')['provenance']['clock_provenance']['system']['last_processed']}))
    finally:
        release.set()
        session.stop()
    assert row['snapshot']['timestamp'] == .04
    assert frozen['mapped_timestamp'] == row['snapshot']['timestamp']
    assert frozen['origin'] == 'validated_fps_fallback'
    assert frozen['sequence'] == 2
    if observed_role == 'acquisition':
        assert row['phase_evidence']['last_observed_at'] == frozen['mapped_timestamp']
    assert session.store.lookup('frozen-clock')['provenance']['clock_provenance'][observed_role]['last_processed'] == frozen
    assert session.predict('frozen-clock')['provenance']['clock_provenance'][observed_role]['last_processed'] == frozen


def test_legacy_export_redacts_fragments_on_first_access_after_restart(tmp_path):
    """A fresh process has no fragment registry; source is only in the manifest."""
    program = r'''
import hashlib,io,json,pathlib,sys,zipfile
from src.experiment.export import ExperimentStore
root=pathlib.Path(sys.argv[1]);root.mkdir()
source='rtsp://review-user:review-password@camera.local/feed?token=review-token'
manifest={'schema_version':1,'run_id':'old','sources':{'system':source}}
evidence={'schema_version':1,'error':'Backend rejected review-password and review-token','predictions':[],'vehicles':[],'comparisons':[]}
for name,data in [('manifest.json',manifest),('evidence.json',evidence)]: (root/name).write_text(json.dumps(data))
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
with zipfile.ZipFile(io.BytesIO(ExperimentStore.open(root).export())) as z:
    text=''.join(z.read(name).decode() for name in z.namelist())
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
print(json.dumps({'password_leaked':'review-password' in text,'token_leaked':'review-token' in text,'old_files_unchanged':before==after}))
assert 'review-password' not in text and 'review-token' not in text
'''
    result = subprocess.run([str(Path('.venv/bin/python').resolve()), '-c', program, str(tmp_path / 'legacy')],
                            text=True, capture_output=True, timeout=20)
    print(result.stdout)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('endpoint', ['evidence', 'history/predictions', 'predictions/p'])
@pytest.mark.parametrize('source_location', ['manifest', 'evidence'])
def test_first_historical_api_read_redacts_fragments_without_changing_old_files(tmp_path, endpoint, source_location):
    program = r'''
import hashlib,json,pathlib,sys
from fastapi.testclient import TestClient
from src.ui.app import app
import src.ui.observation_api as api
from src.ui.observation_session import ObservationManager
base=pathlib.Path(sys.argv[1]);root=base/'old';root.mkdir(parents=True)
source='rtsp://first-user:first-password@camera.local/feed?token=first-token'
manifest={'schema_version':1,'run_id':'old'}
row={'request_id':'p','status':'failed','error':'Backend rejected first-password and first-token'}
evidence={'schema_version':1,'error':row['error'],'predictions':[row],'vehicles':[],'comparisons':[]}
if sys.argv[3]=='manifest':manifest['sources']={'system':source}
else:evidence['camera_calibrations']={'system':{'source':source}}
for name,data in [('manifest.json',manifest),('evidence.json',evidence)]: (root/name).write_text(json.dumps(data))
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
api.observations=ObservationManager(base)
response=TestClient(app).get('/api/observation/'+sys.argv[2],params={'run_id':'old'})
assert response.status_code==200,response.text
assert 'first-password' not in response.text and 'first-token' not in response.text,response.text
after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()}
assert before==after
'''
    result = subprocess.run([str(Path('.venv/bin/python').absolute()), '-c', program,
                             str(tmp_path / 'runs'), endpoint, source_location],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
