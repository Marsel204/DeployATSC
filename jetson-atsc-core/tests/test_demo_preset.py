"""The demo exercises the real observation pipeline with labelled synthetic inputs."""
import cv2
import pytest


def test_demo_has_distinct_feeds_and_scripted_stop_and_crossing(tmp_path):
    from scripts.launch_demo import DemoDetector, make_preset, phase_at
    payload = make_preset(tmp_path, duration=23)
    assert payload['system']['source'] != payload['acquisition']['source']
    assert 'DEMO' in payload['policy']['movement_id']
    for role in ('system', 'acquisition'):
        capture = cv2.VideoCapture(payload[role]['source'])
        ok, frame = capture.read()
        capture.release()
        assert ok and list(frame.shape[1::-1]) == payload[role]['image_size']
    detector = DemoDetector('unused')
    stopped = detector.track([frame], timestamp=3)[0]
    assert len(stopped) == 3 and all(v.velocity_valid and v.velocity_mps == 0 for v in stopped)
    moving = detector.track([frame], timestamp=13)[0]
    assert moving[0].bottom_center[1] > 245
    assert phase_at(3) == 'RED' and phase_at(13) == 'GREEN' and phase_at(21) == 'YELLOW'


def test_demo_capture_generates_cycle_waits_and_restartable_export(tmp_path):
    from scripts.launch_demo import make_preset, make_manager, step_to
    from src.experiment.export import ExperimentStore
    payload = make_preset(tmp_path, duration=23)
    manager = make_manager(tmp_path)
    session = manager.start(payload)
    try:
        step_to(session, 3)
        session.predict('demo-click')
        session.engine.predictions.await_idle()
        # Camera 1 analyses the clicked still frame after admission (on-demand contract).
        row = session.engine.predictions.records['demo-click']
        assert row['camera_analysis']['inference_calls'] == 1
        assert row['snapshot']['quality']['Q']['state'] == 'single_frame_spatial_proxy'
        step_to(session, 22)
        session.stop()
        evidence = session.store.read()
        assert row['snapshot']['queue_meters'] > 0
        assert evidence['comparisons'][0]['actual_green_sec'] == pytest.approx(8, abs=.11)
        assert evidence['comparisons'][0]['prediction']['request_id'] == 'demo-click'
        assert evidence['comparisons'][0]['eligible']
        assert any(v['red_wait_sec'] > 0 and v['crossed_stop_line'] for v in evidence['vehicles'])
        assert evidence['predictions'][0]['inference_diagnostics']['provider'] == 'CPUExecutionProvider'
        assert session.finalized and session.resources_released
        reopened = ExperimentStore.open(session.store.directory)
        assert reopened.read()['run_id'] == session.run_id
        assert reopened.export().startswith(b'PK')
    finally:
        if not session.resources_released:
            session.stop()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_real_preset_uses_native_detector_not_scripted_tracks(tmp_path, device):
    from scripts.launch_demo import make_manager
    from src.perception.detector_tracker import DetectorTracker
    manager = make_manager(tmp_path, scripted=False, device=device)
    detector = manager.session_factory.keywords['detector_factory']
    assert detector.func is DetectorTracker
    assert detector.keywords['device'] == device
