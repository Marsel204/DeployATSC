"""Test-Driven Scaffold for Interactive ROI Calibration and Persistence API."""

import json
import os
from pathlib import Path
import sys
import pytest
from fastapi.testclient import TestClient

CORE_ROOT = Path(__file__).resolve().parent.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.ui.app import app

client = TestClient(app)

JAMIKA_VIDEO = str(CORE_ROOT / "data/sample_videos/simpang5_jamika_h264.mp4")


def test_get_video_snapshot(video_factory):
    """Verify endpoint extracts clean unannotated raw first frame JPEG."""
    video = video_factory()
    resp = client.get(f"/api/video/snapshot?video_path={video}")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert len(resp.content) > 1000  # Valid JPEG binary content


def test_save_valid_custom_roi_config(tmp_path, isolated_config_output):
    """Verify custom config creation with user-drawn lamp box and corridor quad."""
    config_payload = {
        "name": "test_jamika_custom",
        "lamp_roi_pixel": {
            "ymin": 470,
            "xmin": 340,
            "ymax": 620,
            "xmax": 430,
        },
        "src_points_pixel": [
            [200.0, 700.0],
            [1080.0, 700.0],
            [900.0, 400.0],
            [380.0, 400.0],
        ],
        "ground_dimensions_m": {
            "width": 8.0,
            "length": 40.0,
        },
        "description": "Simpang 5 Jamika Custom User-Drawn ROI",
    }

    resp = client.post("/api/configs/save_custom", json=config_payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "saved"
    saved_path = data["config_path"]
    assert os.path.exists(saved_path)

    # Verify saved JSON schema conforms to pipeline requirements
    with open(saved_path, "r", encoding="utf-8") as f:
        saved_cfg = json.load(f)

    assert saved_cfg["traffic_light_monitor"]["lamp_roi_pixel"]["ymin"] == 470
    assert saved_cfg["homography"]["src_points_pixel"][0] == [200.0, 700.0]
    assert "dst_points_ground" in saved_cfg["homography"]

    # Cleanup temporary test config
    if os.path.exists(saved_path):
        os.remove(saved_path)


def test_reject_degenerate_corridor_quad():
    """Verify backend rejects degenerate / non-convex 4-point corridor quads."""
    degenerate_payload = {
        "name": "test_degenerate",
        "lamp_roi_pixel": {
            "ymin": 100,
            "xmin": 100,
            "ymax": 200,
            "xmax": 200,
        },
        # Collinear points that cannot form a perspective projection
        "src_points_pixel": [
            [100.0, 100.0],
            [200.0, 200.0],
            [300.0, 300.0],
            [400.0, 400.0],
        ],
        "ground_dimensions_m": {
            "width": 8.0,
            "length": 40.0,
        },
    }

    resp = client.post("/api/configs/save_custom", json=degenerate_payload)
    assert resp.status_code in (400, 422)
