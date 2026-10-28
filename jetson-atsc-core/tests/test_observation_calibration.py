from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pytest

from src.experiment.calibration import CameraCalibration, legacy_system_payload


def payload(role="system", size=(320, 240)):
    w, h = size
    return {"role": role, "source": "test.mp4", "image_size": [w, h],
            "anchors": [[20, 30], [w-20, 30], [w-10, h-20], [10, h-20]],
            "polygon": [[20, 30], [w-20, 30], [w-10, h-20], [10, h-20]],
            "road_width_m": 8, "road_length_m": 40, "rectification_path": None,
            "signal_roi": {"xmin": 10, "ymin": 10, "xmax": 40, "ymax": 40},
            "stop_line": [[10, h-30], [w-10, h-30]], "crossing_direction": 1}


def test_independent_camera_sizes_and_transforms():
    a = CameraCalibration(payload(size=(1920, 1080)))
    b = CameraCalibration(payload("acquisition", (1280, 720)))
    assert a.revision != b.revision
    assert a.transform(np.zeros((1080, 1920, 3), np.uint8)).shape == (1080, 1920, 3)
    with pytest.raises(ValueError, match="dimensions"):
        b.transform(np.zeros((1080, 1920, 3), np.uint8))
    for camera in [a, b]:
        x, y = camera.homography.ground_to_pixel(4, 20)
        gx, gy = camera.homography.pixel_to_ground(x, y)
        assert gx == pytest.approx(4, abs=.05)
        assert gy == pytest.approx(20, abs=.2)


@pytest.mark.parametrize("change", [
    {"anchors": [[1, 1]] * 4}, {"road_width_m": -1},
    {"signal_roi": {"xmin": 0, "ymin": 0, "xmax": 400, "ymax": 10}},
    {"stop_line": [[1, 1], [1, 1]]}, {"image_size": [float("nan"), 240]},
])
def test_invalid_acquisition_geometry_rejected(change):
    data = payload("acquisition"); data.update(change)
    with pytest.raises(ValueError):
        CameraCalibration(data)


def test_legacy_profile_migration_is_readonly_and_role_specific(tmp_path):
    source = Path(__file__).resolve().parents[1] / "configs/custom_simpang5_jamika.json"
    original = source.read_bytes()
    data = legacy_system_payload(json.loads(original), "sample.mp4", [1920, 1080])
    assert data["role"] == "system"
    assert data["anchors"] == json.loads(original)["homography"]["src_points_pixel"]
    assert data.get("signal_roi") is None
    assert source.read_bytes() == original


def test_rectification_and_source_binding_changes_revision():
    data = payload()
    a = CameraCalibration(data)
    data = deepcopy(data); data["source"] = "another.mp4"
    assert CameraCalibration(data).revision != a.revision


def test_new_road_profile_uses_its_own_ground_area():
    from src.analytics.traffic_metrics import TrafficMetricsExtractor
    camera=CameraCalibration(payload())
    metrics=TrafficMetricsExtractor.from_config(camera.legacy_metrics_config(),camera.homography)
    assert metrics.physical_road_area==pytest.approx(320,abs=.01)


def test_self_intersecting_waiting_polygon_rejected():
    data=payload("acquisition")
    data["polygon"]=[[20,30],[290,200],[20,200],[250,30]]
    with pytest.raises(ValueError,match="intersect"):
        CameraCalibration(data)


def test_saved_lens_digest_rejects_same_size_content_replacement(tmp_path):
    lens=tmp_path/'lens.npz'
    def write(value):
        np.savez(lens,K=np.array([[200.,0,160],[0,200.,120],[0,0,1]]),D=np.array([value,0.,0.,0.]),dim=[320,240])
    write(.01)
    data=payload();data['rectification_path']=str(lens)
    saved=CameraCalibration(data).payload
    write(.02)
    with pytest.raises(ValueError,match='digest|content'):
        CameraCalibration(saved)


@pytest.mark.parametrize('method',['measured_dimensions','ruler'])
def test_explicit_scale_and_reviewed_binding_roundtrip(method):
    data=payload()
    data['scale_evidence']={'method':method,'reference':'synthetic measured fixture'}
    data['binding']={'reviewed':True}
    if method=='ruler':data['ruler']={'p1':[30,180],'p2':[280,180],'length_m':6}
    calibrated=CameraCalibration(data)
    assert calibrated.scale_status['state']=='verified'
    assert calibrated.payload['binding']['geometry_digest']
    assert CameraCalibration(calibrated.payload).revision==calibrated.revision
    changed=deepcopy(calibrated.payload);changed['road_width_m']=9
    with pytest.raises(ValueError,match='binding'):
        CameraCalibration(changed)


def test_default_dimensions_and_legacy_save_remain_review_required():
    data=payload()
    first=CameraCalibration(data)
    second=CameraCalibration(first.payload)
    assert first.scale_status['state']==second.scale_status['state']=='review_required'
    assert 'scale_evidence_missing' in second.scale_status['reasons']


def test_ruler_scale_requires_actual_ruler():
    data=payload();data.update(scale_evidence={'method':'ruler','reference':'fixture'},binding={'reviewed':True})
    with pytest.raises(ValueError,match='ruler'):CameraCalibration(data)


def test_run_preflight_checks_saved_lens_content_again(video_factory,tmp_path):
    from src.ui.observation_session import ObservationSession
    from test_observation_sessions import run_payload,FakeDetector,FakeANFIS
    lens=tmp_path/'lens.npz'
    def write(value):np.savez(lens,K=np.array([[200.,0,160],[0,200.,120],[0,0,1]]),D=np.array([value,0.,0.,0.]),dim=[320,240])
    write(.01)
    data=run_payload(video_factory);data['system']['rectification_path']=str(lens)
    data['system']=CameraCalibration(data['system']).payload
    write(.02)
    with pytest.raises(ValueError,match='digest'):
        ObservationSession.preflight(data,tmp_path/'runs',FakeDetector,FakeANFIS)
