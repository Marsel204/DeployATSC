"""T41-T43/T46: opt-in tracking floor; legacy still/fallback calls stay exact."""
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml

from src.perception.detector_tracker import DetectorTracker
from src.perception.tracking_profiles import TRACKING_PROFILES, get_tracking_profile

ROOT = Path(__file__).resolve().parents[1]
LOW_YAML = ROOT / 'configs/trackers/low_score.yaml'
SETTINGS = dict(tracker_type='bytetrack', track_high_thresh=.25, track_low_thresh=.1,
                new_track_thresh=.25, track_buffer=30, match_thresh=.8, fuse_score=True)


def detector(monkeypatch, **kwargs):
    monkeypatch.setattr(DetectorTracker, '_load_model', lambda self: None)
    value = DetectorTracker('unused.pt', conf_threshold=.28, **kwargs)
    value.model = SimpleNamespace(track=Mock(return_value=[]), predict=Mock(return_value=[]), predictor=None)
    return value


def test_registry_is_small_and_immutable():
    assert set(TRACKING_PROFILES) == {'baseline', 'low_score'}
    assert get_tracking_profile('baseline') is TRACKING_PROFILES['baseline']
    with pytest.raises(FrozenInstanceError):
        get_tracking_profile('low_score').name = 'changed'
    with pytest.raises(TypeError):
        TRACKING_PROFILES['another'] = get_tracking_profile('baseline')


@pytest.mark.parametrize('name', ['missing', '', '../trackers/custom.yaml', None, 10])
def test_unknown_profile_rejected_before_model_allocation(monkeypatch, name):
    load = Mock()
    monkeypatch.setattr(DetectorTracker, '_load_model', load)
    with pytest.raises(ValueError, match='Unknown tracking profile'):
        DetectorTracker('unused.pt', tracking_profile=name)
    load.assert_not_called()


@pytest.mark.parametrize('profile', [None, 'baseline', 'low_score'])
def test_t41_exact_track_kwargs(monkeypatch, profile):
    options = {} if profile is None else {'tracking_profile': profile}
    value = detector(monkeypatch, device='test:device', **options)
    frame = np.zeros((20, 20, 3), np.uint8)
    frames = [frame]
    value.track(frames, timestamp=1.)
    value.model.track.assert_called_once_with(source=frames, conf=.1 if profile == 'low_score' else .28,
        iou=.5, imgsz=640, device='test:device', persist=True,
        tracker=str(LOW_YAML) if profile == 'low_score' else 'bytetrack.yaml', verbose=False)
    assert value.conf_threshold == .28 and value.iou_threshold == .5


def test_constructor_default_and_custom_confidence_remain_backward_compatible(monkeypatch):
    monkeypatch.setattr(DetectorTracker, '_load_model', lambda self: None)
    for kwargs, expected in [({}, .30), ({'conf_threshold': .61}, .61)]:
        value = DetectorTracker('unused.pt', **kwargs)
        value.model = SimpleNamespace(track=Mock(return_value=[]))
        value.track([np.zeros((20, 20, 3), np.uint8)])
        assert value.model.track.call_args.kwargs['conf'] == expected


@pytest.mark.parametrize('profile', ['baseline', 'low_score'])
def test_t42_predict_fallback_and_detect_once_unchanged(monkeypatch, profile):
    value = detector(monkeypatch, tracking_profile=profile)
    value.model.track.side_effect = RuntimeError('forced tracker failure')
    frame = np.zeros((20, 20, 3), np.uint8)
    value.track([frame], timestamp=1.)
    value.model.predict.assert_called_once_with(source=[frame], conf=.28, iou=.5,
                                               imgsz=640, device='cuda:0', verbose=False)
    still_predictor = object()
    monkeypatch.setattr('src.perception.detector_tracker._still_predictor_type', lambda: still_predictor)
    value.model.predict.reset_mock()
    value.detect_once(frame, timestamp=2.)
    value.model.predict.assert_called_once_with(source=frame, predictor=still_predictor, conf=.28,
                                               iou=.5, imgsz=640, device='cuda:0', verbose=False)


def test_t43_yaml_has_only_frozen_bytetrack_keys():
    # A dict equality assertion rejects extra keys as well as altered values.
    assert yaml.safe_load(LOW_YAML.read_text()) == SETTINGS


@pytest.mark.parametrize('profile', ['baseline', 'low_score'])
def test_t46_profile_survives_epoch_reset(monkeypatch, profile):
    value = detector(monkeypatch, tracking_profile=profile, max_gap_sec=.5)
    frames = [np.zeros((20, 20, 3), np.uint8)]
    value.track(frames, timestamp=1.)
    first = value.model.track.call_args.kwargs.copy()
    value.track_trajectories[1] = [(1., 0., 0.)]
    value.track_obs_counts[1] = 1
    value.reset_tracking()
    value.track(frames, timestamp=2.)
    assert value.epoch == 1 and not value.track_trajectories and not value.track_obs_counts
    assert value.max_gap_sec == .5 and value.model.predictor is None
    assert value.model.track.call_args.kwargs == first


def test_profile_provenance_hashes_actual_yaml_and_effective_settings(monkeypatch):
    value = detector(monkeypatch, tracking_profile='low_score')
    record = value.tracking_profile_provenance()
    assert record['name'] == 'low_score' and record['status'] == 'applied'
    assert record['tracker_yaml'] == dict(path='configs/trackers/low_score.yaml',
                                         sha256=hashlib.sha256(LOW_YAML.read_bytes()).hexdigest())
    assert record['effective_thresholds'] == dict(SETTINGS, conf=.1, predict_conf=.28,
                                                 iou=.5, agnostic_nms=False)


def test_baseline_provenance_resolves_installed_yaml_without_changing_call(monkeypatch):
    from importlib.util import find_spec
    value = detector(monkeypatch)
    record = value.tracking_profile_provenance()
    actual = Path(find_spec('ultralytics').origin).parent/'cfg/trackers/bytetrack.yaml'
    assert record['name'] == 'baseline'
    assert record['tracker_yaml'] == dict(path='ultralytics/cfg/trackers/bytetrack.yaml',
                                         sha256=hashlib.sha256(actual.read_bytes()).hexdigest())
    assert record['effective_thresholds']['conf'] == .28
