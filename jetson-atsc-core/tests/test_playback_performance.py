import threading

import cv2
import numpy as np
import pytest

from src.ui.runtime_metrics import AcquisitionRates
from src.ui.observation_session import ObservationSession


def test_recent_rate_reports_playback_and_excludes_cold_inference():
    stats = AcquisitionRates()
    stats.record(0., 0., 5000.)
    for i in range(1, 101):
        stats.record(i/10, i/10, 60.)
    result = stats.snapshot(10., True)
    assert result['processed_fps'] == 10.
    assert result['realtime_factor'] == 1.
    assert result['steady_inference_ms'] == 60.
    assert result['cold_start_ms'] == 5000.
    assert stats.snapshot(10., False)['processed_fps'] == 0.
    assert stats.snapshot(20., True)['processed_fps'] == 0.


def test_pause_resume_and_step_do_not_pollute_active_rate():
    stats = AcquisitionRates()
    for i in range(10):stats.record(i/10, i/10, 60.)
    stats.reset_playback()
    stats.record(200., 1., 60.)
    assert stats.snapshot(200., True)['processed_fps'] == 0.
    stats.record(200.1, 1.1, 60.)
    assert stats.snapshot(200.1, True)['processed_fps'] == 10.
    assert stats.snapshot(200.1, True)['realtime_factor'] == 1.
    # Slow processing must report slower source playback, rather than always claiming 1×.
    stats.reset_playback()
    stats.record(201., 1.2, 60.)
    stats.record(201.2, 1.3, 60.)
    assert stats.snapshot(201.2, True)['realtime_factor'] == .5


def test_performance_history_is_bounded():
    stats = AcquisitionRates()
    for i in range(10000):stats.record(i/10, i/10, 60.)
    assert len(stats.samples) <= 512
    assert len(stats.inference) <= 60


def test_clean_preview_is_encoded_on_demand_and_tracks_latest_frame(monkeypatch):
    session = ObservationSession.__new__(ObservationSession)
    session.lock = threading.RLock()
    session.jpeg_frames = {}; session.clean_frames = {}; session.clean_jpeg_cache = {}
    session.frame_times = {}; session.storage_suspended = True
    session._render = lambda role, frame, tracks: np.full_like(frame, 200)
    packet = type('Packet', (), {'role': 'acquisition', 'timestamp': 0.})()
    calls = []; encode = cv2.imencode
    def counted(*args, **kwargs):
        calls.append(args[1].copy())
        return encode(*args, **kwargs)
    monkeypatch.setattr(cv2, 'imencode', counted)
    session._publish_frame(packet, np.full((40, 40, 3), 20, np.uint8), [])
    assert len(calls) == 1
    clean = session.jpeg('acquisition', clean=True)
    decoded = cv2.imdecode(np.frombuffer(clean, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.mean() == pytest.approx(20, abs=1)
    assert session.jpeg('acquisition', clean=True) == clean
    assert len(calls) == 2
    session._publish_frame(packet, np.full((40, 40, 3), 90, np.uint8), [])
    fresh = session.jpeg('acquisition', clean=True)
    assert cv2.imdecode(np.frombuffer(fresh, np.uint8), cv2.IMREAD_COLOR).mean() == pytest.approx(90, abs=1)
    assert len(calls) == 4
