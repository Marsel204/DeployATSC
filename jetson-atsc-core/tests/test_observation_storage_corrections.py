"""Regressions for the Phase B storage review blockers F1-F4 (and coupled F6/F7).

Temporary directories, generated clips and fake detector/ANFIS only.
"""
import json
import random
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.common.types import TrafficSnapshot
from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.predictions import PredictionConflict
from src.perception.detector_tracker import TrackedVehicle
from test_observation_sessions import FakeANFIS


# ---------------------------------------------------------------- session fixture
def _car(track, y, speed, t, x=20):
    vehicle = TrackedVehicle(track, 1, "car", [x-5, y-10, x+5, y], 1, t)
    vehicle.velocity_mps = speed
    vehicle.velocity_valid = True
    return vehicle


class ScriptedDetector:
    """Stopped queue, LOST/crossed native-ID reacquisition and sparse IDs."""
    def __init__(self, *args, **kwargs):
        self.epoch = 0

    def track(self, frames, timestamp=None):
        t = timestamp; out = []
        if .5 <= t < 5:out.append(_car(7, 20 if t < 4.5 else 35, 0 if 1 <= t < 4.3 else 2, t))
        if 7 <= t < 7.6:out.append(_car(7, 38, 2, t))  # crossed identity reacquired after occlusion
        if 1 <= t < 2 or 3.6 <= t < 6:out.append(_car(1003, 20 if t < 5.5 else 35, 0 if t < 5 else 2, t, 30))
        if 9.2 <= t < 12.5:out.append(_car(42, 20 if t < 12 else 47, 0 if 9.6 <= t < 11.8 else 2, t, 40))
        if 10 <= t < 13:out.append(_car(5, 15, 2, t, 50))
        return [out]

    def update_velocities(self, *args, **kwargs):pass

    def reset_tracking(self):self.epoch += 1


def _clips(tmp_path, seconds=14, fps=10):
    paths = {}
    for role in ("system", "acquisition"):
        path = tmp_path/f"{role}.mp4"
        if not path.exists():
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
            for index in range(fps*seconds):
                t = index/fps
                color = ((0, 0, 255) if t < 4 or t >= 9 else (0, 255, 0) if t < 8 else (0, 255, 255)) if role == "acquisition" else (40, 40, 40)
                writer.write(np.full((48, 64, 3), color, np.uint8))
            writer.release()
        paths[role] = path
    return paths


def scripted_payload(tmp_path, budget=1):
    def calibration(role, source):
        return {'role':role,'source':str(source),'image_size':[64,48],
                'anchors':[[3,3],[60,3],[60,44],[3,44]],'polygon':[[3,3],[60,3],[60,44],[3,44]],
                'road_width_m':8,'road_length_m':40,'signal_roi':{'xmin':0,'ymin':0,'xmax':64,'ymax':48},
                'stop_line':[[3,30],[60,30]],'crossing_direction':1}
    clips = _clips(tmp_path)
    return {'system':calibration('system', clips['system']), 'acquisition':calibration('acquisition', clips['acquisition']),
            'step_mode':True, 'policy':{'debounce_frames':2}, 'retention_budget':budget}


@pytest.fixture
def tapped(monkeypatch):
    """Mirror every real engine admission into an unretained full-history oracle."""
    import src.ui.observation_session as module

    class Tapped(ObservationEngine):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.oracle = ObservationEngine(*args, **kwargs)

        def update_system(self, snapshot, *, generation=None):
            super().update_system(snapshot, generation=generation)
            self.oracle.update_system(snapshot, generation=generation)

        def observe_system_frame(self, timestamp, *, generation=None):
            # Camera 1 preview admission (on-demand YOLO contract) is mirrored too.
            self.oracle.observe_system_frame(timestamp, generation=generation)
            return super().observe_system_frame(timestamp, generation=generation)

        def observe_acquisition(self, frame, timestamp, sequence, observations, *, generation=None):
            super().observe_acquisition(frame, timestamp, sequence, observations, generation=generation)
            self.oracle.observe_acquisition(frame, timestamp, sequence, observations, generation=generation)

        def invalidate(self, reason, timestamp=None):
            super().invalidate(reason, timestamp); self.oracle.invalidate(reason, timestamp)

        def close(self, reason="recording_end"):
            super().close(reason); self.oracle.close(reason)

    monkeypatch.setattr(module, "ObservationEngine", Tapped)
    return Tapped


def step(session, timeout=3):
    try:control = session.control("step")
    except ValueError:return True
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        with session.lock:
            done = session.applied_control_seq >= control["control_seq"] and not session.steps and not session.packets
        if done or session.pair.ended:return True
        time.sleep(.001)
    return False


KEYS = ('vehicles', 'comparisons', 'summary', 'phase_events', 'observation_gaps', 'processed_frames')


def durable_vs_oracle(session):
    actual = session.store.read()
    oracle = json.loads(json.dumps(session.engine.oracle.snapshot(0, 10**6, True)))
    return ({k:actual[k] for k in KEYS}, {k:oracle[k] for k in KEYS})


def neutral(evidence, run_id):
    return json.loads(json.dumps(evidence).replace(run_id, "RUN"))


# ------------------------------------------------------------------------- F1
@pytest.mark.parametrize("mode", ["blocked_http_fsync", "normal_disk_hammer"])
def test_f1_http_commits_never_discard_worker_frames_and_match_uninterrupted_oracle(tmp_path, tapped, mode):
    from src.ui.observation_session import ObservationSession
    runs = {}
    for name in ("uninterrupted", mode):
        session = ObservationSession(scripted_payload(tmp_path), tmp_path/name, detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
        release = threading.Event(); entered = threading.Event(); stop_hammer = threading.Event(); outcomes = []
        writer = None
        try:
            steps = 0
            while not session.pair.ended:
                if name == mode and steps == 40:
                    if mode == "blocked_http_fsync":
                        original = session.store._stage
                        def block(stage, operation):
                            if stage == "archive_fsync" and threading.current_thread() is writer and not release.is_set():
                                entered.set(); release.wait(10)
                            return original(stage, operation)
                        session.store._stage = block
                        # A capture/cancel-style revision: a new generation without new measurements.
                        with session.engine.predictions.lock:session.engine.predictions.revision += 1
                        writer = threading.Thread(target=lambda:outcomes.append(session.persist()))
                        writer.start()
                        assert entered.wait(3), "HTTP-thread commit must reach its archive fsync"
                        r0 = session.runtime["acquisition"]["frames"]; e0 = session.engine.processed["acquisition"]
                        for _ in range(15):
                            assert step(session), "worker must keep measuring while an HTTP commit is blocked"
                        assert writer.is_alive()
                        r1 = session.runtime["acquisition"]["frames"]; e1 = session.engine.processed["acquisition"]
                        assert r1-r0 >= 15 and e1-e0 == r1-r0, ("packets consumed vs admitted", r1-r0, e1-e0)
                        release.set(); writer.join(5); assert outcomes == [True]
                        session.store._stage = original
                    else:
                        def hammer():
                            while not stop_hammer.is_set():
                                outcomes.append(session.persist()); time.sleep(.002)
                        writer = threading.Thread(target=hammer); writer.start()
                assert step(session), "worker step did not drain"
                steps += 1
            stop_hammer.set()
            if writer is not None:writer.join(5)
            assert all(outcomes)
            session.stop()
            assert session.finalized
            actual, oracle = durable_vs_oracle(session)
            assert actual == oracle
            runs[name] = neutral(actual, session.run_id)
        finally:
            release.set(); stop_hammer.set(); session.stop()
    assert runs["uninterrupted"] == runs[mode]
    flags = [f for row in runs[mode]["vehicles"] for f in row["quality_flags"]]
    assert "observation_gap" not in flags


def test_f1_capture_accepted_during_worker_fsync_with_acceptance_time(tmp_path):
    from src.ui.observation_session import ObservationSession
    session = ObservationSession(scripted_payload(tmp_path), tmp_path/"runs", detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
    entered = threading.Event(); release = threading.Event(); result = []
    try:
        while session.engine.time < 2:assert step(session)
        original = session.store._stage
        def block(stage, operation):
            if stage == "archive_fsync" and threading.current_thread() is session.thread and not release.is_set():
                entered.set(); release.wait(10)
            return original(stage, operation)
        session.store._stage = block
        session.engine.revision += 0
        deadline = time.monotonic()+5
        while not entered.is_set() and time.monotonic() < deadline:step(session, .2)
        assert entered.is_set(), "worker commit must be blocked in archive fsync"
        accepted = session.engine.time
        def click():
            try:result.append(session.predict("during-worker-fsync"))
            except Exception as error:result.append(error)
        thread = threading.Thread(target=click); thread.start()
        deadline = time.monotonic()+2
        while "during-worker-fsync" not in session.engine.predictions.records and time.monotonic() < deadline and not result:
            time.sleep(.005)
        assert "during-worker-fsync" in session.engine.predictions.records, ("capture rejected during worker commit", result)
        assert entered.is_set() and not release.is_set()
        assert session.engine.predictions.records["during-worker-fsync"]["clicked_at"] == accepted
        release.set(); thread.join(5)
        assert isinstance(result[0], dict) and result[0]["clicked_at"] == accepted, result
        session.store._stage = original
        assert session.store.lookup("during-worker-fsync")["clicked_at"] == accepted
    finally:
        release.set(); session.stop()


def test_f1_frames_admitted_between_prepare_and_ack_match_oracle(tmp_path):
    """Acknowledgment applies only the prepared immutable prefix while later evidence continues."""
    engine = make_engine(); oracle = make_engine(); engine.enable_retention(2); store = ExperimentStore(tmp_path, "run", {})
    t = 0.; prepared = None
    try:
        for sequence in range(160):
            t += .9 if sequence in (23, 77) else .1
            color = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (0, 0, 0)][(sequence//9) % 4]
            phase = sequence % 10
            rows = [VehicleObservation(1, 'car', 0 if sequence % 13 < 7 else 1, (-1, 0), velocity_valid=sequence % 17 != 0),
                    VehicleObservation(10+sequence//10, 'car', 0 if phase < 5 else 1, (-1 if phase < 7 else 1, 0), velocity_valid=True)]
            for e in (engine, oracle):e.observe_acquisition(np.full((2, 2, 3), color, np.uint8), t, sequence, rows)
            if prepared is None and sequence % 4 == 0:prepared = engine.prepare_commit()
            elif prepared is not None and sequence % 4 == 3:
                payload, token = prepared; store.persist(payload); engine.acknowledge(token); prepared = None
        for e in (engine, oracle):e.close()
        if prepared:store.persist(prepared[0]); engine.acknowledge(prepared[1])
        engine.revision += 1
        payload, token = engine.prepare_commit(); store.persist(payload); engine.acknowledge(token)
        actual = store.read(); expected = json.loads(json.dumps(oracle.snapshot(0, 10**6, True)))
        for key in ('vehicles', 'comparisons', 'summary', 'phase_events'):assert actual[key] == expected[key], key
    finally:engine.close(); oracle.close()


# ------------------------------------------------------------------------- F2/F7
def make_engine(**policy):
    settings = dict(debounce_frames=1, max_gap_sec=2, stop_persistence_sec=.3, association_gap_sec=1.0); settings.update(policy)
    return ObservationEngine('run', RunPolicy(**settings), {'xmin':0, 'ymin':0, 'xmax':2, 'ymax':2}, lambda p:p[0],
                             lambda *_:SimpleNamespace(green_seconds=25, failsafe_active=False))


def drive(steps, retain, tmp_path, name):
    engine = make_engine(); store = ExperimentStore(tmp_path, name, {})
    if retain:engine.enable_retention(1)
    for i, (t, obs, phase, cycle) in enumerate(steps):
        # As in the real engine, the cycle being measured is the observer's current one.
        if cycle is not None:engine.phase.cycle_id = max(engine.phase.cycle_id, cycle-1)
        engine.waits.observe(obs, phase, t, i, cycle); engine.revision += 1; engine.time = t
        if retain:payload, token = engine.prepare_commit(); store.persist(payload); engine.acknowledge(token)
    engine.close()
    payload, token = engine.prepare_commit(); store.persist(payload); engine.acknowledge(token)
    return store.read(), engine


def assert_matches_oracle(steps, tmp_path):
    retained, engine = drive(steps, True, tmp_path, 'retained'); oracle, _ = drive(steps, False, tmp_path, 'oracle')
    for key in ('vehicles', 'summary', 'comparisons'):assert retained[key] == oracle[key], key
    return retained, engine


def moving(identity, x, epoch=0, region=True):
    return VehicleObservation(identity, 'car', 2.0, (x, 0), velocity_valid=True, in_region=region, epoch=epoch)


@pytest.mark.parametrize("gap", [0.6, 2.0], ids=["inside_horizon", "after_horizon"])
def test_f2_exited_identity_reacquired_is_left_censored_segment(tmp_path, gap):
    steps = [(-.2, [], 'RED', 1)]+[(i*.2, [moving(5, -3+i*.1)], 'RED', 1) for i in range(3)]
    steps.append((.6, [moving(5, -2.7, region=False)], 'RED', 1))      # leaves corridor: EXITED
    steps.append((.6+gap, [moving(5, -2.5)], 'RED', 1))               # same native identity returns
    steps += [(.8+gap, [moving(5, -2.4)], 'RED', 1), (1.0+gap, [moving(5, -2.3, region=False)], 'RED', 1)]
    evidence, _ = assert_matches_oracle(steps, tmp_path)
    assert [(r['vehicle_id'], r['left_censored']) for r in evidence['vehicles']] == [
        ('run:acquisition:0:5', False), ('run:acquisition:0:5:segment-1', True)]


def test_f2_lost_identity_reacquired_after_horizon_matches_oracle(tmp_path):
    steps = [(i*.2, [moving(5, -3+i*.1)], 'RED', 1) for i in range(3)]
    steps += [(2.0, [], 'RED', 1), (2.2, [moving(5, -2.5)], 'RED', 1), (2.4, [moving(5, -2.4)], 'RED', 1),
              (2.6, [moving(5, -2.3, region=False)], 'RED', 1)]
    evidence, _ = assert_matches_oracle(steps, tmp_path)
    assert evidence['summary']['complete_vehicle_count'] == 0


@pytest.mark.parametrize("gap", [0.4, 1.4], ids=["inside_horizon", "after_horizon"])
def test_f2_crossed_identity_reacquired_is_not_a_second_vehicle(tmp_path, gap):
    steps = [(0., [], 'GREEN', 1), (.2, [moving(9, -1.)], 'GREEN', 1), (.4, [moving(9, .5)], 'GREEN', 1), (.6, [moving(9, 1.)], 'GREEN', 1)]
    steps += [(.6+gap, [], 'GREEN', 1)]
    steps += [(.8+gap+i*.2, [moving(9, 2.+i*.5)], 'GREEN', 1) for i in range(3)]
    evidence, _ = assert_matches_oracle(steps, tmp_path)
    assert evidence['summary']['observed_vehicle_count'] == 1


def test_f2_epoch_change_is_a_new_native_identity(tmp_path):
    steps = [(0., [moving(3, -2.)], 'RED', 1), (.2, [moving(3, -2., region=False)], 'RED', 1), (2., [], 'RED', 1),
             (2.2, [moving(3, -2., epoch=1)], 'RED', 1), (2.4, [moving(3, -1.9, epoch=1)], 'RED', 1)]
    evidence, _ = assert_matches_oracle(steps, tmp_path)
    assert [r['vehicle_id'] for r in evidence['vehicles']] == ['run:acquisition:0:3', 'run:acquisition:1:3']


def test_f7_first_appearance_uses_base_identity(tmp_path):
    engine = make_engine(); engine.enable_retention(1)
    engine.waits.observe([moving(i, -1) for i in (1, 2, 3)], 'RED', 0, 0, 1)
    assert sorted(engine.waits.records) == ['run:acquisition:0:1', 'run:acquisition:0:2', 'run:acquisition:0:3']
    engine.close()


def test_f2_long_sparse_out_of_order_identities_match_oracle_with_bounded_memory(tmp_path):
    rng = random.Random(7)
    ids = rng.sample(range(10**7), 160)
    plan = {}
    for index, identity in enumerate(ids):
        start = rng.randrange(0, 1300); epoch = 0 if start < 650 else 1
        for _ in range(rng.choice((1, 2, 3))):
            length = rng.randrange(4, 16); kind = rng.choice(("cross", "exit", "lost"))
            for k in range(length):
                x = -3+.3*k if kind == "cross" else -3+.1*k
                speed = 0. if 3 <= k < 6 else 2.
                region = not (kind == "exit" and k == length-1)
                plan.setdefault(start+k, []).append(VehicleObservation(identity, 'car', speed, (x, 0), velocity_valid=True,
                                                                         in_region=region, epoch=epoch))
            start += length+rng.randrange(1, 40)
    steps = []
    for i in range(max(plan)+3):
        phase = 'RED' if (i//30) % 2 == 0 else 'GREEN'
        rows = sorted(plan.get(i, []), key=lambda o:rng.random())
        seen = set(); rows = [o for o in rows if not ((o.epoch, o.local_id) in seen or seen.add((o.epoch, o.local_id)))]
        steps.append((i*.2, rows, phase, i//60+1))
    hits = []
    import src.experiment.retention as retention_module
    original = retention_module.Retention.identity
    def counted(self, base):
        found = original(self, base)
        if found is not None:hits.append(base)
        return found
    retention_module.Retention.identity = counted
    try:
        retained, engine = drive(steps, True, tmp_path, 'retained')
    finally:retention_module.Retention.identity = original
    oracle, _ = drive(steps, False, tmp_path, 'oracle')
    for key in ('vehicles', 'summary', 'comparisons'):assert retained[key] == oracle[key], key
    assert hits, "scenario must exercise identities forgotten by memory"
    assert len(oracle['vehicles']) > 200
    assert len(engine.waits.active) <= 20 and len(engine.waits.records) <= 20
    rebuilt = engine.retention.rebuilt(ExperimentStore.open(tmp_path/'retained').records())
    for identity in {(o.epoch, o.local_id) for row in steps for o in row[1]}:
        base = f"run:acquisition:{identity[0]}:{identity[1]}"
        assert engine.retention.identity(base) == rebuilt.identity(base)


def test_f2_session_reacquisition_matches_unretained_oracle(tmp_path, tapped):
    from src.ui.observation_session import ObservationSession
    session = ObservationSession(scripted_payload(tmp_path), tmp_path/"runs", detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
    try:
        while not session.pair.ended:assert step(session)
        session.stop()
        actual, oracle = durable_vs_oracle(session)
        assert actual == oracle
        ids = [r['vehicle_id'].split(':', 1)[1] for r in actual['vehicles']]
        assert 'acquisition:0:7' in ids and not any(i.startswith('acquisition:0:7:') for i in ids)
        assert 'acquisition:0:1003:segment-1' in ids
    finally:session.stop()


# ------------------------------------------------------------------------- F3
def test_f3_new_current_run_capture_never_reads_archive_history(tmp_path, monkeypatch):
    from src.ui.observation_session import ObservationSession
    session = ObservationSession(scripted_payload(tmp_path), tmp_path/"runs", detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
    try:
        while session.engine.time < 2:assert step(session)
        with monkeypatch.context() as patch:
            for name in ("records", "lookup", "read"):
                patch.setattr(session.store, name, lambda *a, **k:(_ for _ in ()).throw(AssertionError("history scan before capture")))
            record = session.predict("fresh-click")
            assert record["request_id"] == "fresh-click"
            session.engine.predictions.await_idle()
            for _ in range(5):assert step(session)
            assert session.predict("fresh-click")["request_id"] == "fresh-click"
        assert session.store.lookup("fresh-click")["request_id"] == "fresh-click"
    finally:session.stop()


def test_f3_archive_index_is_exact_and_python_memory_does_not_grow(tmp_path):
    import tracemalloc
    from src.experiment.archive_index import ArchiveIndex
    index = ArchiveIndex(tmp_path)
    for i in range(500):index.put(f"warm-{i}".encode(), i)
    tracemalloc.start(); before = tracemalloc.take_snapshot()
    for i in range(20000):index.put(f"key-{i*7919 % 100003}".encode(), i)
    after = tracemalloc.take_snapshot(); tracemalloc.stop()
    growth = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
    assert growth < 256*1024, growth
    assert index.get(b"key-7919") == 1 and index.get(b"key-missing") is None
    index.put(b"key-7919", 99); assert index.get(b"key-7919") == 99


# ------------------------------------------------------------------------- F4
def test_f4_run_is_discoverable_only_after_coherent_generation_zero(tmp_path, monkeypatch):
    from src.ui.observation_session import ObservationManager
    original = ExperimentStore._stage
    calls = []
    monkeypatch.setattr(ExperimentStore, "_stage", lambda self, name, op:(calls.append(name), original(self, name, op))[1])
    ExperimentStore(tmp_path/"probe", "run_probe", {})
    stages = list(calls); assert stages

    class Fake:
        run_id = "run_new"; resources_released = True; state = "running"
        def __init__(self, *args, **kwargs):pass
    for index in range(len(stages)):
        for after in (False, True):
            root = tmp_path/f"{index}-{after}"; counter = [0]; error = OSError("injected "+stages[index])
            def fault(self, name, operation, index=index, after=after, counter=counter, error=error):
                counter[0] += 1
                if counter[0]-1 != index:return original(self, name, operation)
                if after:operation()
                raise error
            monkeypatch.setattr(ExperimentStore, "_stage", fault)
            with pytest.raises(OSError) as caught:ExperimentStore(root, "run_partial", {"operation":"observation"})
            assert caught.value is error
            monkeypatch.setattr(ExperimentStore, "_stage", original)   # disk repaired
            manager = ObservationManager(storage_root=root, session_factory=Fake, preflight=lambda *a:None)
            assert all(r["status"] != "unavailable" for r in manager.runs()), (stages[index], after, manager.runs())
            assert manager.start({}, before_handoff=lambda:None).run_id == "run_new"
            ExperimentStore(root, "run_next", {})
            assert ExperimentStore.open(root/"run_next").recovery_status()["generation"] == 0


def test_f4_failed_session_store_creation_does_not_block_next_start(tmp_path, monkeypatch):
    from functools import partial
    from src.ui.observation_session import ObservationManager, ObservationSession, ReplacementFailed
    import src.experiment.export as export
    original = export.ExperimentStore._write
    def full(self, name, data):
        if name == "evidence.json" and not (self.directory/"evidence.json").exists():raise OSError("ENOSPC")
        return original(self, name, data)
    manager = ObservationManager(tmp_path/"runs", session_factory=partial(ObservationSession, detector_factory=ScriptedDetector, anfis_factory=FakeANFIS))
    with monkeypatch.context() as patch:
        patch.setattr(export.ExperimentStore, "_write", full)
        with pytest.raises(ReplacementFailed):manager.start(scripted_payload(tmp_path), before_handoff=lambda:None)
    failed = manager.current
    if failed is not None:
        deadline = time.monotonic()+5
        while not failed.resources_released and time.monotonic() < deadline:time.sleep(.01)
    session = manager.start(scripted_payload(tmp_path), before_handoff=lambda:None)
    try:assert all(r["status"] != "unavailable" for r in manager.runs())
    finally:session.stop()


# ------------------------------------------------------------------------- F6
def test_f6_terminal_capture_cannot_be_cancelled_across_acknowledgment(tmp_path):
    engine = make_engine(); engine.enable_retention(1); store = ExperimentStore(tmp_path, 'run', {})
    frame = np.full((2, 2, 3), (0, 0, 255), np.uint8)
    try:
        for s in range(2):engine.observe_acquisition(frame, s*.2, s, [])
        for i in range(2):
            engine.update_system(TrafficSnapshot(timestamp=engine.time, v_w_pcu=1, queue_meters=None, occupancy_pct=1))
            engine.predict(f'r{i}'); engine.predictions.await_idle()
        payload, token = engine.prepare_commit()
        with pytest.raises(PredictionConflict):engine.predictions.cancel('r0')
        store.persist(payload); engine.acknowledge(token)
        assert store.lookup('r0')['status'] == 'failed'
    finally:engine.close()


def test_f6_api_cancel_rejects_terminal_and_commits_eligible_cancel(tmp_path, monkeypatch):
    from functools import partial
    from fastapi.testclient import TestClient
    from src.ui.app import app
    import src.ui.observation_api as api
    from src.ui.observation_session import ObservationManager, ObservationSession
    manager = ObservationManager(tmp_path/"runs", session_factory=partial(ObservationSession, detector_factory=ScriptedDetector, anfis_factory=FakeANFIS))
    monkeypatch.setattr(api, "observations", manager)
    with TestClient(app) as client:
        assert client.post("/api/observation/start", json=scripted_payload(tmp_path)).status_code == 200
        run = manager.current
        try:
            while run.engine.time < 2:assert step(run)
            run.engine.predictions.evaluate = lambda *_:(_ for _ in ()).throw(RuntimeError("backend failed"))
            assert client.post("/api/observation/predict", json={"run_id":run.run_id, "request_id":"failed"}).status_code == 200
            run.engine.predictions.await_idle()
            response = client.post("/api/observation/predictions/failed/cancel", params={"run_id":run.run_id})
            assert response.status_code == 409
            assert run.engine.predictions.records["failed"]["status"] == "failed"
            run.engine.predictions.evaluate = lambda *_:SimpleNamespace(green_seconds=30, failsafe_active=False)
            assert client.post("/api/observation/predict", json={"run_id":run.run_id, "request_id":"live"}).status_code == 200
            run.engine.predictions.await_idle()
            response = client.post("/api/observation/predictions/live/cancel", params={"run_id":run.run_id})
            assert response.status_code == 200 and response.json()["status"] == "cancelled"
            assert run.store.lookup("live")["status"] == "cancelled"
        finally:run.stop()


# ------------------------------------------------------------- repeated experiments
def test_repeated_runs_stay_separate_and_remain_accessible_after_restart(tmp_path, monkeypatch):
    import io, zipfile
    from functools import partial
    from fastapi.testclient import TestClient
    from src.ui.app import app
    import src.ui.observation_api as api
    from src.ui.observation_session import ObservationManager, ObservationSession
    factory = partial(ObservationSession, detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
    manager = ObservationManager(tmp_path/"runs", session_factory=factory)
    monkeypatch.setattr(api, "observations", manager)
    records = {}
    with TestClient(app) as client:
        for _ in range(2):
            assert client.post("/api/observation/start", json=scripted_payload(tmp_path)).status_code == 200
            run = manager.current
            while run.engine.time < 2:assert step(run)
            response = client.post("/api/observation/predict", json={"run_id":run.run_id, "request_id":"click-1"})
            assert response.status_code == 200
            run.engine.predictions.await_idle()
            while not run.pair.ended:assert step(run)
            run.stop(); assert run.finalized
            records[run.run_id] = client.post("/api/observation/predict", json={"run_id":run.run_id, "request_id":"click-1"}).json()
        first, second = records
        assert first != second and records[first]["run_id"] == first and records[second]["run_id"] == second
        # Process restart: a fresh manager with no current session.
        monkeypatch.setattr(api, "observations", ObservationManager(tmp_path/"runs", session_factory=factory))
        listed = {r["run_id"]:r for r in client.get("/api/observation/runs").json()["runs"]}
        assert set(listed) == {first, second} and all(r["status"] == "finalized" for r in listed.values())
        for run_id in (first, second):
            assert client.post("/api/observation/predict", json={"run_id":run_id, "request_id":"click-1"}).json() == records[run_id]
            evidence = client.get("/api/observation/evidence", params={"run_id":run_id}).json()
            assert evidence["vehicles"] and all(v["vehicle_id"].startswith(run_id+":") for v in evidence["vehicles"])
            assert [p["request_id"] for p in evidence["predictions"]] == ["click-1"]
            page = client.get("/api/observation/history/vehicles", params={"run_id":run_id, "limit":2}).json()
            assert page["total"] == len(evidence["vehicles"])
            bundle = client.get("/api/observation/export", params={"run_id":run_id})
            with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
                assert json.loads(archive.read("manifest.json"))["run_id"] == run_id
        assert neutral(client.get("/api/observation/evidence", params={"run_id":first}).json()["vehicles"], first) == \
            neutral(client.get("/api/observation/evidence", params={"run_id":second}).json()["vehicles"], second)


def test_earlier_vehicle_cycle_fragment_shape_remains_readable(tmp_path):
    store = ExperimentStore(tmp_path, "run", {})
    row = {"vehicle_id":"v", "first_seen":0, "cycles":[3], "wait_by_cycle":{"3":{"red_wait_sec":1., "total_stopped_sec":1.}}, "intervals":[]}
    store.persist({"snapshot_revision":1, "vehicles":[row], "_archive":[
        {"kind":"vehicle_cycles", "id":"v", "value":{"1":{"red_wait_sec":2., "total_stopped_sec":2.}}},
        {"kind":"vehicle_cycles", "id":"v", "value":{"cycles":[2], "wait_by_cycle":{}}}]})
    vehicle = store.read()["vehicles"][0]
    assert vehicle["cycles"] == [1, 2, 3]
    assert vehicle["wait_by_cycle"] == {"1":{"red_wait_sec":2., "total_stopped_sec":2.}, "3":{"red_wait_sec":1., "total_stopped_sec":1.}}


def test_f5_history_pages_reuse_one_rebuild_per_pinned_generation(tmp_path, monkeypatch):
    store = ExperimentStore(tmp_path, "run", {})
    vehicles = [{"vehicle_id":f"v{i}", "first_seen":i, "cycles":[i % 50+1], "wait_by_cycle":{}, "intervals":[]} for i in range(500)]
    cycles = [{"cycle_id":c, "green_start":c, "green_end":c+.5} for c in range(1, 51)]
    store.persist({"snapshot_revision":1, "time":100, "_archive":[{"kind":"vehicles", "id":v["vehicle_id"], "value":v} for v in vehicles]+
                   [{"kind":"comparisons", "id":c["cycle_id"], "value":c} for c in cycles]})
    calls = []
    original = ExperimentStore.read
    monkeypatch.setattr(ExperimentStore, "read", lambda self, *a, **k:(calls.append(1), original(self, *a, **k))[1])
    offset = 0; seen = []
    while offset is not None:
        page = ExperimentStore.open(store.directory).page("vehicles", offset, 100)   # a new reader per HTTP request
        seen += [r["vehicle_id"] for r in page["rows"]]; offset = page["next_offset"]
    assert len(seen) == 500 and len(calls) == 1
    assert [c["observed_vehicle_count"] for c in ExperimentStore.open(store.directory).page("comparisons", 0, 60)["rows"]] == [10]*50
    store.persist({"snapshot_revision":2, "time":100})
    assert ExperimentStore.open(store.directory).page("vehicles")["generation"] == 2 and len(calls) == 2


def test_run_list_labels_legacy_neutrally_and_keeps_recovered_origin(tmp_path):
    from src.ui.observation_session import ObservationManager
    legacy = tmp_path/"run_legacy"; legacy.mkdir()
    (legacy/"manifest.json").write_text(json.dumps({"schema_version":2, "run_id":"run_legacy"}))
    (legacy/"evidence.json").write_text(json.dumps({"closed":True, "vehicles":[]}))
    open_run = ExperimentStore(tmp_path, "run_open", {}); open_run.persist({"snapshot_revision":1, "closed":False})
    sealed = ExperimentStore(tmp_path, "run_sealed", {}); sealed.persist({"snapshot_revision":1, "closed":True})
    manager = ObservationManager(tmp_path, preflight=lambda *a:None,
        session_factory=lambda *a:SimpleNamespace(run_id="new", resources_released=True, state="running"))
    before = {r["run_id"]:r["status"] for r in manager.runs()}
    assert before == {"run_legacy":"legacy", "run_open":"unclean", "run_sealed":"abandoned"}
    manager.start({}, before_handoff=lambda:None)
    after = {r["run_id"]:r for r in manager.runs()}
    assert after["run_legacy"]["status"] == "legacy"
    assert (after["run_open"]["status"], after["run_open"]["recovered_from"]) == ("finalized", "unclean")
    assert (after["run_sealed"]["status"], after["run_sealed"]["recovered_from"]) == ("finalized", "abandoned")


def test_f9_dashboard_lists_saved_runs_and_exports_the_selected_run():
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    html = (root/"src/ui/static/index.html").read_text()
    for identity in ("savedRuns", "refreshRuns", "exportSaved", "savedRunStatus"):assert f'id="{identity}"' in html
    script = (root/"src/ui/static/observation.js").read_text().replace(
        "document.addEventListener('DOMContentLoaded',initialize);",
        "window.renderRuns=renderRuns;window.savedRunExport=savedRunExport;")
    harness = r'''
const vm=require('vm'),assert=require('assert');
class Element{constructor(){this.textContent='';this.children=[];this.value='';this.disabled=false;}
 append(...i){this.children.push(...i);} replaceChildren(...i){this.children=i;}}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element());return elements.get(id);};
const context={document:{getElementById:get,querySelectorAll:()=>[],createElement:()=>new Element(),createTextNode:t=>t},window:{}};
vm.createContext(context);vm.runInContext(SCRIPT,context);
context.window.renderRuns([{run_id:'run_a',status:'finalized',durable_revision:9},
  {run_id:'run_b',status:'finalized',recovered_from:'unclean'},{run_id:'run_old',status:'legacy'}]);
const options=get('savedRuns').children;
assert.deepStrictEqual(options.map(o=>o.value),['run_a','run_b','run_old']);
assert.match(options[1].textContent,/recovered.*unclean/);assert.match(options[2].textContent,/legacy/);
assert.strictEqual(get('exportSaved').disabled,false);
get('savedRuns').value='run_b';
assert.strictEqual(context.window.savedRunExport(),'/api/observation/export?run_id=run_b');
context.window.renderRuns([]);assert.strictEqual(get('exportSaved').disabled,true);
assert.match(get('savedRunStatus').textContent,/No saved runs/);
'''.replace("SCRIPT", json.dumps(script))
    result = subprocess.run(["node", "-"], input=harness, text=True, capture_output=True, timeout=10)
    assert result.returncode == 0, result.stdout+result.stderr
