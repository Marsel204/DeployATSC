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
