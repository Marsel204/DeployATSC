"""Camera 1 runs YOLO only on an explicit capture: exactly one still frame per accepted request.

Camera 2 keeps continuous detection/tracking. Single-frame queue values are a labelled
spatial proxy, never measured motion or a qualified stopped queue.
"""
from functools import partial
import threading
import time

import numpy as np
import pytest

from src.experiment.predictions import PredictionConflict
from src.experiment.sources import FramePacket
from src.perception.detector_tracker import TrackedVehicle
from src.ui.observation_session import ObservationManager, ObservationSession
from test_observation_calibration import payload
from test_observation_sessions import FakeANFIS, run_payload


def car(track, x, bottom, t=0.):
    return TrackedVehicle(track, 1, "car", [x-12, bottom-20, x+12, bottom], .9, t, stable_id=False)


class CountingDetector:
    """Records every native call per role instance; Camera 1 still images may be gated."""
    def __init__(self, gate=None, detections=None):
        self.epoch = 0; self.track_calls = []; self.detect_calls = []; self.resets = 0
        self.closed = False; self.gate = gate; self.entered = threading.Event()
        self.detections = detections if detections is not None else [car(1, 160, 150), car(2, 120, 100)]
        self.closed_during_call = False; self.in_call = False

    def track(self, frames, timestamp=None):
        self.track_calls.append(timestamp)
        return [[] for _ in frames]

    def detect_once(self, frame, timestamp=None):
        self.in_call = True
        try:
            self.detect_calls.append({"timestamp":timestamp, "mean":float(frame.mean()),
                                      "thread":threading.current_thread().name})
            self.entered.set()
            if self.gate is not None:assert self.gate.wait(10)
            return [car(v.track_id, (v.bbox[0]+v.bbox[2])/2, v.bbox[3], timestamp) for v in self.detections]
        finally:self.in_call = False

    def update_velocities(self, *args, **kwargs):pass

    def reset_tracking(self):
        assert not self.in_call, "Camera 1 model reset beneath its one-shot call"
        self.resets += 1; self.epoch += 1

    def close(self):
        self.closed_during_call = self.closed_during_call or self.in_call
        self.closed = True


def factory(made, **kwargs):
    def make(*args, **ignored):
        detector = CountingDetector(**kwargs) if not made else CountingDetector()
        made.append(detector)
        return detector
    return make


def scripted_pair(n=60, green_from=99., dt=.1):
    class Pair:
        def __init__(self, *args, **kwargs):
            self.k = 0; self.ended = False; self.has_backlog = False; self.closed = False
            self.system = type("S", (), {"fps":200})()
            self.acquisition = type("A", (), {"ended":False})()

        @property
        def resources_released(self):return self.closed

        def step(self):
            if self.k >= n:
                self.ended = True; self.acquisition.ended = True
                return []
            t = round(self.k*dt, 6); self.k += 1
            color = (0, 255, 0) if t >= green_from else (0, 0, 255)
            return [FramePacket("system", self.k, t, t, np.full((240, 320, 3), self.k, np.uint8), "test_clock"),
                    FramePacket("acquisition", self.k, t, t, np.full((240, 320, 3), color, np.uint8), "test_clock")]

        def close(self):self.closed = True
        def health(self):return {r:{"state":"ready", "age_sec":0, "image_size":[320, 240]} for r in ("system", "acquisition")}
        def media_identity(self):return {"state":"scripted"}
    return Pair


def step(session, count=1):
    for _ in range(count):
        status = session.control("step")
        deadline = time.monotonic()+5
        while True:
            with session.lock:
                done = session.applied_control_seq >= status["control_seq"] and not session.steps and not session.packets
            if done:break
            assert time.monotonic() < deadline and not session.error, session.error
            time.sleep(.002)


def scripted_session(monkeypatch, tmp_path, video_factory, made, *, n=60, green_from=99., gate=None, detections=None):
    import src.ui.observation_session as module
    monkeypatch.setattr(module, "ReplayPair", scripted_pair(n, green_from))
    return ObservationSession(run_payload(video_factory), tmp_path,
                              detector_factory=factory(made, gate=gate, detections=detections), anfis_factory=FakeANFIS)


def test_zero_camera1_inference_across_startup_replay_controls_and_preview(video_factory, tmp_path):
    made = []
    manager = ObservationManager(tmp_path/"runs", tmp_path/"profiles",
        partial(ObservationSession, detector_factory=factory(made), anfis_factory=FakeANFIS))
    data = run_payload(video_factory)
    session = manager.start(data)
    try:
        system, acquisition = session.detectors["system"], session.detectors["acquisition"]
        assert system is made[0] and acquisition is made[1]
        assert not system.track_calls and not system.detect_calls  # startup
        step(session, 5)
        session.control("play")
        deadline = time.monotonic()+5
        while session.engine.processed["acquisition"] < 30 and time.monotonic() < deadline:time.sleep(.01)
        session.control("pause"); session.control("play"); session.control("pause")
        time.sleep(.05)
        assert manager.preview("system", data["system"]["source"]) == session.jpeg("system", True)
        step(session, 3)
        telemetry = session.telemetry()
        assert not system.track_calls and not system.detect_calls and system.resets == 0
        assert len(acquisition.track_calls) == session.engine.processed["acquisition"] >= 30
        assert session.engine.processed["system"] >= 30  # decoded preview frames, not inference
        assert telemetry["metrics"] is None
        assert telemetry["runtime"]["system"]["frames"] == 0
        assert telemetry["runtime"]["system"]["inference_mode"] == "on_demand_single_frame"
        assert telemetry["runtime"]["system"]["preview_frames"] >= 30
        assert telemetry["runtime"]["acquisition"]["inference_mode"] == "continuous"
        camera1 = telemetry["camera1"]
        assert camera1["inference_mode"] == "on_demand_single_frame"
        assert camera1["frame_available"] and camera1["frame_fresh"] and camera1["last_analysis"] is None
        assert session.jpeg("system") and session.jpeg("system", True)
    finally:
        manager.shutdown()


def test_one_accepted_click_runs_exactly_one_still_frame_and_retries_never_reinfer(monkeypatch, video_factory, tmp_path):
    made = []; gate = threading.Event()
    session = scripted_session(monkeypatch, tmp_path, video_factory, made, gate=gate)
    try:
        step(session, 4)
        system = session.detectors["system"]
        before = session.engine.processed["system"]
        record = session.predict("click-1")
        assert record["status"] == "predicting" and record["snapshot"] is None
        assert record["camera_analysis"]["state"] == "pending"
        frame = record["camera1_frame"]
        assert frame["sequence"] == before and frame["timestamp"] == pytest.approx(record["clicked_at"])
        assert frame["timestamp_origin"] == "test_clock" and frame["source_timestamp"] == frame["timestamp"]
        assert system.entered.wait(2)
        # Concurrent same-ID retries during the blocked call, then a new overlapping click.
        results = []
        threads = [threading.Thread(target=lambda:results.append(session.predict("click-1"))) for _ in range(6)]
        for thread in threads:thread.start()
        for thread in threads:thread.join(2)
        assert len(results) == 6 and all(r["request_id"] == "click-1" for r in results)
        with pytest.raises(PredictionConflict):session.predict("click-2")
        gate.set(); session.engine.predictions.await_idle()
        for _ in range(3):session.predict("click-1")
        assert len(system.detect_calls) == 1 and not system.track_calls
        assert system.detect_calls[0]["mean"] == pytest.approx(before)  # the frozen click frame
        assert system.detect_calls[0]["thread"].startswith("ANFIS-observation")
        done = session.engine.predictions.records["click-1"]
        analysis = done["camera_analysis"]
        assert analysis["state"] == "complete" and analysis["inference_calls"] == 1
        assert analysis["method"] == "single_frame_still_image" and analysis["detections"] == 2
        assert analysis["completed_at"] is not None and analysis["latency_ms"] >= 0
        assert done["recommended_green_sec"] == 35 and done["status"] == "awaiting_green"
        telemetry = session.telemetry()
        assert telemetry["runtime"]["system"]["frames"] == 1
        assert telemetry["metrics"]["v_w_pcu"] == 2 and telemetry["camera1"]["last_analysis"]["request_id"] == "click-1"
        assert telemetry["camera1"]["last_analysis"]["frame_sequence"] == before
        assert telemetry["camera1"]["last_analysis"]["age_sec"] >= 0
        # Further playback never re-runs Camera 1.
        step(session, 3)
        assert len(system.detect_calls) == 1 and not system.track_calls
        assert session.telemetry()["camera1"]["last_analysis"]["age_sec"] == pytest.approx(.3, abs=.01)
    finally:
        gate.set(); session.stop()


def test_single_frame_queue_is_a_labelled_diagnostic_proxy_with_real_counts(monkeypatch, video_factory, tmp_path):
    made = []
    session = scripted_session(monkeypatch, tmp_path, video_factory, made)
    try:
        step(session, 2)
        session.predict("proxy"); session.engine.predictions.await_idle()
        record = session.engine.predictions.records["proxy"]
        snapshot = record["snapshot"]
        q = snapshot["quality"]["Q"]
        assert q["state"] == "single_frame_spatial_proxy" and q["method"] == "single_frame_spatial_extent"
        assert q["available"] is True and q["complete"] is False and q["measured"] is False
        assert "single_frame_no_motion_evidence" in q["reasons"] and "stopped_state_unknown" in q["reasons"]
        assert q["motion_valid_vehicle_count"] == 0 and q["observed_vehicle_count"] == 2
        assert snapshot["stopped_vehicles"] is None  # unknown, never a fabricated zero
        assert snapshot["queue_meters"] > 0 and snapshot["n_lv"] == 2 and snapshot["occupancy_pct"] > 0
        assert snapshot["quality"]["V_w"]["complete"] and snapshot["quality"]["L"]["state"] == "footprint_estimate"
        assert record["metric_method"] == "camera1_single_frame_spatial_proxy"
        qualification = record["qualification"]
        assert qualification["state"] == "diagnostic"
        assert "Q_measurement_single_frame_spatial_proxy" in qualification["reasons"]
        assert not record["qualified"]
    finally:
        session.stop()


def test_blocked_camera1_call_never_stops_camera2_and_freezes_click_evidence(monkeypatch, video_factory, tmp_path):
    made = []; gate = threading.Event()
    session = scripted_session(monkeypatch, tmp_path, video_factory, made, green_from=1.0, gate=gate)
    try:
        step(session, 5)
        record = session.predict("blocked")
        assert record["measurement_phase_at_capture"] == "RED"
        clicked, frame = record["clicked_at"], dict(record["camera1_frame"])
        system, acquisition = session.detectors["system"], session.detectors["acquisition"]
        assert system.entered.wait(2)
        processed = session.engine.processed["acquisition"]
        # No measurement/control lock is held by the blocked native call.
        for lock in (session.lock, session.engine.lock, session.engine.predictions.lock, session.persist_lock):
            assert lock.acquire(timeout=.5); lock.release()
        step(session, 10)
        assert session.engine.processed["acquisition"] == processed+10
        assert len(acquisition.track_calls) == session.engine.processed["acquisition"]
        assert session.engine.phase.phase == "GREEN"
        assert session.telemetry()["prediction_busy"] is True
        pending = session.engine.predictions.records["blocked"]
        assert pending["clicked_at"] == clicked and pending["camera1_frame"] == frame
        assert pending["measurement_phase_at_capture"] == "RED" and pending["green_start"] == pytest.approx(1.0)
        with pytest.raises(PredictionConflict):session.predict("during-green-in-flight")
        gate.set(); session.engine.predictions.await_idle()
        done = session.engine.predictions.records["blocked"]
        assert done["status"] == "late" and not done["eligible"]
        assert done["camera1_frame"] == frame and done["clicked_at"] == clicked
        assert done["completed_lead_sec"] < 0 and done["camera_analysis"]["lead_sec"] < 0
        assert done["click_lead_sec"] == pytest.approx(1.0-clicked)
        assert len(system.detect_calls) == 1
    finally:
        gate.set(); session.stop()


def test_timely_single_frame_capture_matches_with_total_completion_lead(monkeypatch, video_factory, tmp_path):
    made = []
    session = scripted_session(monkeypatch, tmp_path, video_factory, made, green_from=1.0)
    try:
        step(session, 5)
        session.predict("timely"); session.engine.predictions.await_idle()
        step(session, 10)
        record = session.engine.predictions.records["timely"]
        assert record["status"] == "matched" and record["eligible"] and not record["qualified"]
        analysis = record["camera_analysis"]
        assert analysis["completed_at"] <= record["completed_at"]
        assert record["completed_lead_sec"] == pytest.approx(record["green_start"]-record["completed_at"])
        assert analysis["lead_sec"] == pytest.approx(record["green_start"]-analysis["completed_at"])
        assert record["completed_lead_sec"] > 0
        comparison = session.engine.snapshot()["comparisons"][0]
        assert comparison["prediction"]["metric_method"] == "camera1_single_frame_spatial_proxy"
        # The observed green is still running, so the cycle comparison is not yet complete.
        assert comparison["prediction"]["eligible"] and not comparison["eligible"] and not comparison["qualified"]
    finally:
        session.stop()


def test_unavailable_and_stale_camera1_frames_are_rejected_without_inference(monkeypatch, video_factory, tmp_path):
    made = []
    session = scripted_session(monkeypatch, tmp_path, video_factory, made)
    try:
        step(session, 2)
        with session.lock:
            latest = session.camera1_latest; session.camera1_latest = None
        with pytest.raises(ValueError, match="Camera 1 frame is unavailable"):session.predict("none")
        with session.lock:
            session.camera1_latest = dict(latest, timestamp=latest["timestamp"]-5)
        with pytest.raises(ValueError, match="Camera 1 frame is stale"):session.predict("stale")
        with session.lock:
            session.camera1_latest = dict(latest, generation=latest["generation"]-1)
        with pytest.raises(ValueError, match="measurement discontinuity"):session.predict("old-generation")
        assert not session.detectors["system"].detect_calls
        assert not session.engine.predictions.records
    finally:
        session.stop()


def test_cancel_during_camera1_call_discards_late_result_and_keeps_model(monkeypatch, video_factory, tmp_path):
    made = []; gate = threading.Event()
    session = scripted_session(monkeypatch, tmp_path, video_factory, made, gate=gate)
    try:
        step(session, 3)
        session.predict("cancelled"); assert session.detectors["system"].entered.wait(2)
        assert session.cancel("cancelled")["status"] == "cancelled"
        step(session, 2)
        gate.set(); session.engine.predictions.await_idle()
        record = session.engine.predictions.records["cancelled"]
        assert record["status"] == "cancelled" and record["recommended_green_sec"] is None
        assert record["snapshot"] is None and record["camera_analysis"]["state"] == "discarded_after_cancelled"
        assert session.telemetry()["camera1"]["last_analysis"] is None and session.telemetry()["metrics"] is None
        assert len(session.detectors["system"].detect_calls) == 1
        session.predict("after-cancel"); session.engine.predictions.await_idle()
        assert len(session.detectors["system"].detect_calls) == 2
    finally:
        gate.set(); session.stop()


def test_stop_during_camera1_call_retains_model_until_native_return_and_seals(monkeypatch, video_factory, tmp_path):
    made = []; gate = threading.Event()
    session = scripted_session(monkeypatch, tmp_path, video_factory, made, gate=gate)
    try:
        step(session, 3)
        session.predict("sealed"); system = session.detectors["system"]
        assert system.entered.wait(2)
        with pytest.raises(RuntimeError, match="stopping"):session.stop(timeout=.2)
        lifecycle = session.lifecycle()
        assert lifecycle["measurement_sealed"] and lifecycle["prediction_release_pending"]
        assert not session.resources_released and not system.closed
        assert "system" in session.detectors
        record = session.engine.predictions.records["sealed"]
        assert record["status"] == "unmatched" and record["camera_analysis"]["state"] == "sealed_before_completion"
        gate.set(); session.stop(timeout=5)
        assert session.resources_released and session.finalized
        assert system.closed and not system.closed_during_call
        evidence = session.store.read()["predictions"][0]
        assert evidence["status"] == "unmatched" and evidence["snapshot"] is None
        assert evidence["recommended_green_sec"] is None
        assert evidence["camera_analysis"]["state"] == "sealed_before_completion"
    finally:
        gate.set()
        if not session.resources_released:session.stop(timeout=5)


def test_storage_failure_during_camera1_call_fences_late_publication(monkeypatch, video_factory, tmp_path):
    made = []; gate = threading.Event()
    session = scripted_session(monkeypatch, tmp_path, video_factory, made, gate=gate)
    try:
        step(session, 3)
        session.predict("fenced"); assert session.detectors["system"].entered.wait(2)
        original = session.store.persist
        monkeypatch.setattr(session.store, "persist", lambda state:(_ for _ in ()).throw(OSError("disk full")))
        with session.engine.lock:session.engine.revision += 1
        assert session.persist() is False and session.storage_suspended and session.engine.closed
        with pytest.raises(ValueError, match="suspended"):session.predict("rejected")
        gate.set(); session.engine.predictions.await_idle()
        record = session.engine.predictions.records["fenced"]
        assert record["snapshot"] is None and record["recommended_green_sec"] is None
        monkeypatch.setattr(session.store, "persist", original)
        assert session.retry_storage()
        session.stop(timeout=5)
        assert session.resources_released and session.detectors == {}
        assert len(made[0].detect_calls) == 1
    finally:
        gate.set()
        if not session.resources_released:session.stop(timeout=5)


def test_replacement_waits_for_blocked_camera1_call(monkeypatch, video_factory, tmp_path):
    import src.ui.observation_session as module
    from src.ui.observation_session import ObservationConflict
    monkeypatch.setattr(module, "ReplayPair", scripted_pair())
    made = []; gate = threading.Event()
    manager = ObservationManager(tmp_path/"runs", tmp_path/"profiles",
        partial(ObservationSession, detector_factory=factory(made, gate=gate), anfis_factory=FakeANFIS), shutdown_timeout=.2)
    data = run_payload(video_factory)
    first = manager.start(data, before_handoff=lambda:None)
    try:
        step(first, 2)
        first.predict("replace"); assert made[0].entered.wait(2)
        with pytest.raises(ObservationConflict):manager.start(data, before_handoff=lambda:None)
        assert manager.current is first and not made[0].closed
        gate.set()
        second = manager.start(data, before_handoff=lambda:None)
        assert first.resources_released and made[0].closed and not made[0].closed_during_call
        assert second.detectors["system"].detect_calls == []
    finally:
        gate.set(); manager.shutdown()


def test_evicted_and_restarted_retries_never_reinfer(monkeypatch, video_factory, tmp_path):
    from fastapi.testclient import TestClient
    import src.ui.observation_session as module
    import src.ui.observation_api as api
    from src.ui.app import app
    monkeypatch.setattr(module, "ReplayPair", scripted_pair())
    made = []
    manager = ObservationManager(tmp_path/"runs", tmp_path/"profiles",
        partial(ObservationSession, detector_factory=factory(made), anfis_factory=FakeANFIS))
    monkeypatch.setattr(api, "observations", manager)
    data = run_payload(video_factory); data["retention_budget"] = 1
    session = manager.start(data, before_handoff=lambda:None)
    try:
        client = TestClient(app)
        step(session, 2)
        accepted = client.post("/api/observation/predict", json={"run_id":session.run_id, "request_id":"evict-me"})
        assert accepted.status_code == 200 and accepted.json()["status"] == "predicting"
        session.engine.predictions.await_idle()
        assert session.cancel("evict-me")["status"] == "cancelled"
        # Budget 1 retains only the newest terminal capture; a second one evicts the first.
        session.predict("newer"); session.engine.predictions.await_idle(); session.cancel("newer")
        step(session, 2)
        assert session.persist()
        assert session.engine.retention.requested("evict-me")
        assert "evict-me" not in session.engine.predictions.records
        retry = client.post("/api/observation/predict", json={"run_id":session.run_id, "request_id":"evict-me"})
        assert retry.status_code == 200 and retry.json()["request_id"] == "evict-me"
        assert len(session.detectors["system"].detect_calls) == 2
        old = session.run_id
        replacement = manager.start(data, before_handoff=lambda:None)
        stale = client.post("/api/observation/predict", json={"run_id":old, "request_id":"evict-me"})
        # The superseded run answers from its durable archive; nothing re-runs.
        assert stale.status_code == 200 and stale.json()["request_id"] == "evict-me"
        assert stale.json()["camera_analysis"]["inference_calls"] == 1
        assert replacement.detectors["system"].detect_calls == [] and len(made[0].detect_calls) == 2
    finally:
        manager.shutdown()


def test_detect_once_is_a_single_native_still_image_predict():
    from src.perception.detector_tracker import DetectorTracker

    class Tensor:
        def __init__(self, value):self.value = np.asarray(value)
        def cpu(self):return self
        def numpy(self):return self.value

    class Boxes:
        def __init__(self):
            self.xyxy = Tensor([[10, 20, 50, 80]]); self.conf = Tensor([.9]); self.cls = Tensor([1]); self.id = None
        def __len__(self):return 1

    class Model:
        names = {0:"motorcycle", 1:"car", 2:"bus", 3:"truck"}
        def __init__(self):self.calls = []
        def predict(self, **kwargs):
            self.calls.append(("predict", kwargs))
            return [type("R", (), {"boxes":Boxes()})()]
        def track(self, **kwargs):raise AssertionError("tracker reused for a still image")

    detector = DetectorTracker.__new__(DetectorTracker)
    detector.model = Model(); detector.conf_threshold = .28; detector.iou_threshold = .5; detector.imgsz = 640
    detector.epoch = 3; detector.track_obs_counts = {}
    frame = np.zeros((240, 320, 3), np.uint8)
    vehicles = detector.detect_once(frame, timestamp=4.5)
    assert len(detector.model.calls) == 1
    name, kwargs = detector.model.calls[0]
    assert kwargs["source"] is frame and kwargs["imgsz"] == 640 and kwargs["conf"] == .28
    assert len(vehicles) == 1 and vehicles[0].class_name == "car" and vehicles[0].timestamp == 4.5
    assert not vehicles[0].stable_id and not vehicles[0].velocity_valid and vehicles[0].obs_count == 0


def test_single_frame_extractor_uses_visible_extent_without_motion():
    from src.analytics.traffic_metrics import TrafficMetricsExtractor
    from src.experiment.calibration import CameraCalibration
    calibration = CameraCalibration(payload())
    extractor = TrafficMetricsExtractor.from_config(calibration.legacy_metrics_config(), calibration.homography, strict_motion=True)
    near, far = car(1, 160, 200), car(2, 160, 80)
    raw = extractor.extract_single_frame_metrics([near, far, car(3, 2, 239)])
    far_y = calibration.homography.pixel_to_ground(*far.bottom_center)[1]
    assert raw["Q"] == pytest.approx(round(far_y, 2))
    assert raw["total_corridor_vehicles"] == 2 and raw["stopped_vehicles_count"] is None
    assert raw["quality"]["Q"]["state"] == "single_frame_spatial_proxy"
    assert raw["quality"]["Q"]["complete"] is False
    empty = extractor.extract_single_frame_metrics([])
    assert empty["Q"] == 0 and empty["quality"]["Q"]["state"] == "single_frame_spatial_proxy"
    assert "no_visible_corridor_vehicles" in empty["quality"]["Q"]["reasons"]


def test_unclean_recovery_never_leaves_a_camera1_analysis_pending(monkeypatch, video_factory, tmp_path):
    import shutil
    from src.experiment.export import ExperimentStore
    made = []; gate = threading.Event()
    session = scripted_session(monkeypatch, tmp_path/"live", video_factory, made, gate=gate)
    try:
        step(session, 2)
        session.predict("interrupted"); assert made[0].entered.wait(2)
        assert session.persist()
        # A copy of the durable, unfinalized archive models a process that died mid-analysis.
        copy = tmp_path/"crashed"/session.run_id
        shutil.copytree(session.store.directory, copy)
    finally:
        gate.set(); session.stop(timeout=5)
    recovered = ExperimentStore.open(copy)
    recovered.recover()
    row = recovered.read()["predictions"][0]
    assert row["status"] == "unmatched" and row["invalidation_reason"] == "unclean_recovery"
    assert row["snapshot"] is None and row["recommended_green_sec"] is None
    assert row["camera_analysis"]["state"] == "interrupted_unclean_recovery"


def test_failed_camera1_call_is_recorded_once_and_fails_the_capture(monkeypatch, video_factory, tmp_path):
    made = []
    session = scripted_session(monkeypatch, tmp_path, video_factory, made)
    try:
        step(session, 2)
        def broken(frame, timestamp=None):
            made[0].detect_calls.append(timestamp); raise RuntimeError("CUDA error: device lost")
        monkeypatch.setattr(made[0], "detect_once", broken)
        session.predict("broken"); session.engine.predictions.await_idle()
        record = session.engine.predictions.records["broken"]
        assert record["status"] == "failed" and record["recommended_green_sec"] is None
        assert record["snapshot"] is None and record["camera_analysis"]["state"] == "failed"
        assert record["camera_analysis"]["inference_calls"] == 1
        assert "device lost" in record["camera_analysis"]["error"]
        assert session.telemetry()["metrics"] is None and len(made[0].detect_calls) == 1
        session.predict("broken"); assert len(made[0].detect_calls) == 1  # retry is a read
        session.predict("after-failure"); session.engine.predictions.await_idle()
        assert len(made[0].detect_calls) == 2
    finally:
        session.stop()
