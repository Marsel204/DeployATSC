"""Tests for ATSC Web Dashboard REST API and Streaming Endpoints."""

import json
import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parent.parent
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

import pytest
from fastapi.testclient import TestClient

# Import app from src.ui.app (will fail until src/ui/app.py is implemented)
from src.ui.app import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def isolated_legacy_replay_media(tmp_path_factory):
    """Exercise API behavior with local generated media, not absent field datasets."""
    import shutil
    import cv2
    import numpy as np
    import src.ui.session_manager as sessions
    global CORE_ROOT
    original = CORE_ROOT
    temporary = tmp_path_factory.mktemp("legacy-ui")
    (temporary / "configs").mkdir()
    for source in (original / "configs").iterdir():
        if source.is_file():shutil.copy2(source, temporary / "configs" / source.name)
    (temporary / "models").symlink_to(original / "models", target_is_directory=True)
    directory = temporary / "data/sample_videos/multi_angle_test"
    directory.mkdir(parents=True)
    writer = cv2.VideoWriter(str(directory / "video_bandung_pasteur_h264.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (1920, 1080))
    assert writer.isOpened()
    for _ in range(30):writer.write(np.full((1080,1920,3),40,np.uint8))
    writer.release()
    patch = pytest.MonkeyPatch()
    patch.setattr(sessions, "CORE_ROOT", temporary)
    CORE_ROOT = temporary
    try:yield
    finally:
        sessions.manager.stop_session()
        CORE_ROOT = original
        patch.undo()


def test_index_html():
    """Verify / serves the HTML dashboard."""
    response = client.get("/")
    assert response.status_code == 200
    assert "ATSC JETSON EDGE CONTROLLER" in response.text
    assert "<title>ATSC Edge Control & Replay Dashboard</title>" in response.text


def test_get_configs():
    """Verify /api/configs returns available calibration configurations."""
    response = client.get("/api/configs")
    assert response.status_code == 200
    data = response.json()
    assert "configs" in data
    assert isinstance(data["configs"], list)
    config_names = [c["name"] for c in data["configs"]]
    assert "stress_indonesia_bandung_pasteur.json" in config_names
    assert "stress_indonesia_bekasi.json" in config_names


def test_get_sample_videos():
    """Verify /api/sample_videos returns available sample videos."""
    response = client.get("/api/sample_videos")
    assert response.status_code == 200
    data = response.json()
    assert "videos" in data
    assert isinstance(data["videos"], list)
    assert len(data["videos"]) > 0


def test_session_lifecycle():
    """Verify starting, stepping, pausing, and stopping a replay session."""
    # Find a real sample video
    sample_video = str(CORE_ROOT / "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4")
    sample_config = str(CORE_ROOT / "configs/stress_indonesia_bandung_pasteur.json")

    start_payload = {
        "video_path": sample_video,
        "config_path": sample_config,
        "trigger_interval_sec": 15.0,
        "step_mode": True,
    }

    # Start session
    start_resp = client.post("/api/session/start", json=start_payload)
    assert start_resp.status_code == 200
    start_data = start_resp.json()
    assert start_data["status"] == "initialized"
    assert "session_id" in start_data

    # Check telemetry
    telem_resp = client.get("/api/session/telemetry")
    assert telem_resp.status_code == 200
    telem_data = telem_resp.json()
    assert "frame_idx" in telem_data
    assert "metrics" in telem_data
    assert "V_w" in telem_data["metrics"]
    assert "Q" in telem_data["metrics"]
    assert "L" in telem_data["metrics"]

    # Step frame
    step_resp = client.post("/api/session/control", json={"action": "step"})
    assert step_resp.status_code == 200

    # Stop session
    stop_resp = client.post("/api/session/control", json={"action": "stop"})
    assert stop_resp.status_code == 200
    assert stop_resp.json()["status"] == "stopped"


def test_video_feed_header():
    """Verify /stream/video_feed initiates multipart/x-mixed-replace stream."""
    sample_video = str(CORE_ROOT / "data/sample_videos/multi_angle_test/video_bandung_pasteur_h264.mp4")
    sample_config = str(CORE_ROOT / "configs/stress_indonesia_bandung_pasteur.json")

    client.post("/api/session/start", json={
        "video_path": sample_video,
        "config_path": sample_config,
        "step_mode": True,
    })

    with client.stream("GET", "/stream/video_feed?limit=1") as stream_resp:
        assert stream_resp.status_code == 200
        assert "multipart/x-mixed-replace" in stream_resp.headers.get("content-type", "")
        # Read the first chunk
        chunk = next(stream_resp.iter_bytes())
        assert len(chunk) > 0

    client.post("/api/session/control", json={"action": "stop"})


def test_calibration_canvas_dom_order():
    """Verify that calibrationModal is placed before script to prevent null element evaluation errors."""
    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    modal_pos = html.find('id="calibrationModal"')
    script_pos = html.find('<script>')
    canvas_pos = html.find('id="calibrationCanvas"')

    assert modal_pos != -1, "calibrationModal must be present in index.html"
    assert canvas_pos != -1, "calibrationCanvas must be present in index.html"
    assert script_pos != -1, "<script> tag must be present in index.html"
    assert modal_pos < script_pos, "calibrationModal must precede <script> tag in DOM hierarchy"
    assert "setupCalibrationCanvas" in html, "setupCalibrationCanvas handler must be present"
    assert "#1 Far-Left" in html, "Corridor helper text must align with dst_points Far-to-Near ordering"


def test_index_html_js_syntax_and_tab_switching():
    """Verify that embedded JavaScript in index.html is syntactically valid and has no duplicate declarations."""
    import re
    import subprocess

    response = client.get("/")
    assert response.status_code == 200
    html = response.text

    script_match = re.search(r"<script>([\s\S]*?)</script>", html)
    assert script_match is not None, "Script block must exist in index.html"
    js_code = script_match.group(1)

    # Verify switchSourceTab accepts tab and button, and tabs have data-tab attributes
    assert 'data-tab="samples"' in html
    assert 'data-tab="youtube"' in html
    assert 'data-tab="upload"' in html
    assert "switchSourceTab(tab, btn)" in js_code

    # If node is available, execute syntax validation (validates scope, syntax, and declarations)
    try:
        proc = subprocess.run(
            ["node", "-e", f"new Function({json.dumps(js_code)});"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert proc.returncode == 0
    except FileNotFoundError:
        pass


