"""
Unit and integration tests for arbitrary N-point polygon road area
and metric sample distance ruler calibration (TDD Mandate).
"""

import json
import os
from pathlib import Path
import sys
import tempfile
import numpy as np
import pytest
from fastapi.testclient import TestClient

CORE_ROOT = Path(__file__).resolve().parent.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from src.analytics.homography_engine import HomographyEngine
from src.ui.app import app
from src.ui.session_manager import SessionManager


@pytest.fixture
def base_4pt_homography_data():
    """Rectangular ground 10m x 40m projected to trapezoidal image pixel coordinates."""
    src = [
        [500.0, 300.0],  # Top-left (far)
        [800.0, 300.0],  # Top-right (far)
        [1100.0, 900.0], # Bottom-right (near)
        [200.0, 900.0],  # Bottom-left (near)
    ]
    dst = [
        [0.0, 40.0],
        [10.0, 40.0],
        [10.0, 0.0],
        [0.0, 0.0],
    ]
    return src, dst


def test_homography_engine_arbitrary_multipoint_polygon(base_4pt_homography_data):
    """
    Test that HomographyEngine accepts an arbitrary N-point polygon (N=8)
    for road corridor containment, distinct from the 4-point perspective anchor points.
    """
    src, dst = base_4pt_homography_data

    # Define an 8-point polygon with a concave cut-out / curb notch
    # e.g., following a curved sidewalk
    polygon_8pt = [
        [500.0, 300.0],
        [800.0, 300.0],
        [1100.0, 900.0],
        [800.0, 900.0],
        [750.0, 700.0],  # Notch indent
        [400.0, 700.0],  # Notch indent
        [350.0, 900.0],
        [200.0, 900.0],
    ]

    engine = HomographyEngine(
        src_points_pixel=src,
        dst_points_ground=dst,
        ground_width_m=10.0,
        ground_length_m=40.0,
        corridor_polygon_pixel=polygon_8pt,
    )

    # Point clearly inside polygon
    assert engine.is_in_approach_corridor(650.0, 500.0) is True

    # Point inside the 4-point trapezoid, but in the notched cutout (outside 8-pt polygon)
    # y=800, x=550 is inside the trapezoid [200,900]-[1100,900], but under the notch y>700
    assert engine.is_in_approach_corridor(550.0, 800.0) is False

    # Point outside everything
    assert engine.is_in_approach_corridor(50.0, 100.0) is False


def test_measure_pixel_distance_and_sample_ruler_calibration(base_4pt_homography_data):
    """
    Test pixel distance measurement and sample ruler calibration.
    """
    src, dst = base_4pt_homography_data
    engine = HomographyEngine(
        src_points_pixel=src,
        dst_points_ground=dst,
        ground_width_m=10.0,
        ground_length_m=40.0,
    )

    # Pick two points on the ground plane along the longitudinal direction
    # Say p1 is at ground (5.0, 10.0), p2 is at ground (5.0, 12.0) -> ground distance is 2.0m
    u1, v1 = engine.ground_to_pixel(5.0, 10.0)
    u2, v2 = engine.ground_to_pixel(5.0, 12.0)

    # Before recalibration, measured distance should be close to 2.0m
    d_initial = engine.measure_pixel_distance([u1, v1], [u2, v2])
    assert abs(d_initial - 2.0) < 0.05

    # Now simulate a ruler calibration with a known sample:
    # Say the user tells us the real distance between these two pixels is actually 3.0m
    scale = engine.calibrate_from_sample_ruler([u1, v1], [u2, v2], real_distance_m=3.0)
    assert scale > 0

    # Measured distance after calibration MUST equal 3.0m exactly (within tolerance)
    d_calibrated = engine.measure_pixel_distance([u1, v1], [u2, v2])
    assert abs(d_calibrated - 3.0) < 0.01

    # Invertibility check
    gx, gy = engine.pixel_to_ground(u1, v1)
    u_rec, v_rec = engine.ground_to_pixel(gx, gy)
    assert abs(u_rec - u1) <= 1
    assert abs(v_rec - v1) <= 1


def test_session_manager_validate_and_save_multipoint_polygon():
    """
    Test that SessionManager accepts polygon_points_pixel and sample_ruler in custom config.
    """
    sm = SessionManager()

    payload = {
        "name": "test_multipoint_calib",
        "description": "N-point polygon and 2m sample ruler test",
        "lamp_roi_pixel": {"ymin": 100, "xmin": 200, "ymax": 150, "xmax": 230},
        "src_points_pixel": [
            [500, 300],
            [800, 300],
            [1100, 900],
            [200, 900],
        ],
        "polygon_points_pixel": [
            [500, 300],
            [650, 280],
            [800, 300],
            [950, 600],
            [1100, 900],
            [650, 920],
            [200, 900],
            [350, 600],
        ],
        "sample_ruler": {
            "p1": [645, 600],
            "p2": [655, 650],
            "length_m": 2.0,
        },
        "ground_dimensions_m": {"width": 8.0, "length": 40.0},
    }

    result = sm.validate_and_save_custom_config(payload)
    assert result["status"] == "saved"

    # Check saved JSON config file
    config_file = result["config_path"]
    with open(config_file, "r") as f:
        cfg = json.load(f)

    assert "sample_ruler" in cfg["homography"]
    assert cfg["homography"]["sample_ruler"]["length_m"] == 2.0
    assert "corridor_polygon_pixel" in cfg["homography"]
    assert len(cfg["homography"]["corridor_polygon_pixel"]) == 8
    assert len(cfg["lane_metrics"]["corridor_polygon"]) == 8

    # Load into HomographyEngine from config
    engine = HomographyEngine.from_config(cfg)
    assert len(engine.corridor_poly) == 8

    # Clean up test config
    if os.path.exists(config_file):
        os.remove(config_file)


def test_api_save_custom_config_multipoint():
    """
    Test FastAPI /api/configs/save_custom endpoint with multi-point polygon and sample ruler.
    """
    client = TestClient(app)

    payload = {
        "name": "api_test_poly",
        "description": "API test for polygon & ruler",
        "lamp_roi_pixel": {"ymin": 100, "xmin": 200, "ymax": 150, "xmax": 230},
        "src_points_pixel": [
            [500, 300],
            [800, 300],
            [1100, 900],
            [200, 900],
        ],
        "polygon_points_pixel": [
            [500, 300],
            [700, 300],
            [800, 300],
            [1100, 900],
            [200, 900],
        ],
        "sample_ruler": {
            "p1": [600, 600],
            "p2": [610, 650],
            "length_m": 2.0,
        },
        "ground_dimensions_m": {"width": 8.0, "length": 40.0},
    }

    response = client.post("/api/configs/save_custom", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "saved"

    config_path = data["config_path"]
    if os.path.exists(config_path):
        os.remove(config_path)
