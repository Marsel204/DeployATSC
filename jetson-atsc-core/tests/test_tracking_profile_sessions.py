"""T44/T45: acquisition-only selection, truthful metadata and old-run compatibility."""
import hashlib
import io
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import zipfile

from fastapi.testclient import TestClient
import pytest

from src.perception.detector_tracker import DetectorTracker
from src.perception.tracking_profiles import apply_tracking_profile, tracking_profile_provenance
from src.ui.app import app
import src.ui.observation_api as api
from src.ui.observation_session import ObservationSession, ObservationManager
from src.experiment.export import ExperimentStore
from test_observation_sessions import FakeANFIS, FakeDetector, run_payload, wait_applied

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def native_without_model(monkeypatch):
    """Real capability/kwargs/session path with no hardware allocation."""
    def load(self):
        self.model = SimpleNamespace(track=lambda **kw: [SimpleNamespace(boxes=None)], predictor=None)
    monkeypatch.setattr(DetectorTracker, '_load_model', load)
    return partial(DetectorTracker, device='cpu')


@pytest.mark.parametrize('profile', [None, 'baseline', 'low_score'])
def test_t44_t45_native_session_roles_manifest_and_provenance(tmp_path, video_factory, native_without_model, profile):
    data = run_payload(video_factory)
    if profile is not None:data['tracking_profile'] = profile
    expected = profile or 'baseline'
    session = ObservationSession(data, tmp_path/'runs', detector_factory=native_without_model, anfis_factory=FakeANFIS)
    try:
        acquisition, system = session.detectors['acquisition'], session.detectors['system']
        assert acquisition.tracking_profile.name == expected
        assert system.tracking_profile.name == 'baseline'
        path = session.store.directory/'manifest.json'
        before = path.read_bytes()
        manifest = json.loads(before)
        binding = manifest['tracking_profile']
        assert binding['name'] == binding['requested'] == expected
        assert binding['applied_roles'] == ['acquisition']
        assert binding['status'] == 'applied'
        assert manifest['execution']['tracking_profile'] == binding
        record = manifest['execution']['detectors']['acquisition']['tracking_profile']
        assert record == acquisition.tracking_profile_provenance()
        assert binding['tracker_yaml'] == record['tracker_yaml']
        assert binding['effective_thresholds'] == record['effective_thresholds']
        assert record['effective_thresholds']['conf'] == (.1 if expected == 'low_score' else .28)
        assert record['effective_thresholds']['new_track_thresh'] == .25
        assert record['effective_thresholds']['track_buffer'] == 30
        assert manifest['execution']['detectors']['system']['tracking_profile']['effective_thresholds']['conf'] == .28
        for filename, digest in manifest['source_hashes'].items():
            assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest() == digest
        if expected == 'low_score':
            tracker = ROOT/'configs/trackers/low_score.yaml'
            assert record['tracker_yaml'] == dict(path='configs/trackers/low_score.yaml',
                                                 sha256=hashlib.sha256(tracker.read_bytes()).hexdigest())
        acquisition.reset_tracking()
        result = session.control('step');wait_applied(session, result['control_seq'])
        session._read_execution_provenance()
        session.persist()
        session.stop()
        assert path.read_bytes() == before
        assert session.execution_provenance()['detectors']['acquisition']['tracking_profile'] == record
        assert session.execution_provenance()['tracking_profile'] == binding
        with zipfile.ZipFile(io.BytesIO(session.export())) as bundle:
            exported = json.loads(bundle.read('manifest.json'))
            assert exported['tracking_profile'] == binding
            assert exported['source_hashes'] == manifest['source_hashes']
            assert json.loads(bundle.read('evidence.json'))['execution']['detectors']['acquisition']['tracking_profile'] == record
    finally:session.stop()


@pytest.mark.parametrize('profile', ['baseline', 'low_score'])
def test_t44_injected_factory_without_capability_keeps_signature_and_truthful_metadata(tmp_path, video_factory, profile):
    class StrictDetector(FakeDetector):
        def __init__(self, path, conf_threshold, imgsz):
            super().__init__()
    data = run_payload(video_factory);data['tracking_profile'] = profile
    session = ObservationSession(data, tmp_path/'runs', detector_factory=StrictDetector, anfis_factory=FakeANFIS)
    try:
        binding = session.store.manifest['tracking_profile']
        assert binding['requested'] == profile and binding['name'] == 'baseline'
        assert binding['status'] == 'injected_component' and binding['applied_roles'] == []
        assert binding['tracker_yaml'] == {'path': None, 'sha256': None}
        assert binding['effective_thresholds'] is None
        assert not hasattr(session.detectors['acquisition'], 'tracking_profile')
        assert session.execution_provenance()['detectors']['acquisition']['tracking_profile']['status'] == 'injected_component'
    finally:session.stop()


def test_capability_helper_noops_and_rejects_unknown_without_mutation():
    component = object()
    binding = apply_tracking_profile(component, 'low_score')
    assert binding['status'] == 'injected_component'
    assert tracking_profile_provenance(component)['name'] == 'baseline'
    with pytest.raises(ValueError, match='Unknown tracking profile'):
        apply_tracking_profile(component, 'custom')


def test_t44_unknown_profile_returns_422_before_manager_start(tmp_path, video_factory, monkeypatch):
    manager = ObservationManager(tmp_path/'runs', tmp_path/'profiles')
    monkeypatch.setattr(api, 'observations', manager)
    called = []
    monkeypatch.setattr(manager, 'start', lambda *a, **kw: called.append(True))
    data = run_payload(video_factory);data['tracking_profile'] = 'custom'
    with TestClient(app) as client:
        response = client.post('/api/observation/start', json=data)
    assert response.status_code == 422
    assert 'Unknown tracking profile' in response.text and not called


def test_api_default_and_low_score_payload_are_accepted(video_factory):
    data = run_payload(video_factory)
    assert api.StartObservation(**data).model_dump()['tracking_profile'] == 'baseline'
    assert api.StartObservation(**{**data, 'tracking_profile': 'low_score'}).tracking_profile == 'low_score'


def test_direct_session_preflight_rejects_unknown_before_resources(tmp_path, monkeypatch):
    from unittest.mock import Mock
    import src.ui.observation_session as module
    forbidden = Mock(side_effect=AssertionError('resource allocation'))
    monkeypatch.setattr(module, 'CameraCalibration', forbidden)
    with pytest.raises(ValueError, match='Unknown tracking profile'):
        ObservationSession.preflight({'tracking_profile': 'unknown'}, tmp_path)
    forbidden.assert_not_called()


def test_t45_older_saved_run_loads_and_exports_without_inventing_profile(tmp_path):
    folder = tmp_path/'old';folder.mkdir()
    manifest = {'schema_version': 1, 'run_id': 'old', 'operation': 'observation'}
    evidence = {'run_id': 'old', 'vehicles': [], 'predictions': [], 'comparisons': []}
    for name, value in [('manifest.json', manifest), ('evidence.json', evidence)]:
        (folder/name).write_text(json.dumps(value))
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    reader = ExperimentStore.open(folder)
    assert reader.read()['run_id'] == 'old'
    with zipfile.ZipFile(io.BytesIO(reader.export())) as bundle:
        assert 'tracking_profile' not in json.loads(bundle.read('manifest.json'))
    assert {p.name: p.read_bytes() for p in folder.iterdir()} == before
