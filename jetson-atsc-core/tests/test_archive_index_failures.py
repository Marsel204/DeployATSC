"""A failed derived-index resize must preserve the acknowledged lookup."""
import os

import pytest

from src.experiment.archive_index import ArchiveIndex


@pytest.mark.parametrize("written", [0, 8, 24])
def test_short_rehash_keeps_original_table_and_can_retry(tmp_path, monkeypatch, written):
    index = ArchiveIndex(tmp_path)
    for number in range(512):
        index.put(f"key-{number}".encode(), number + 1)
    original_table = index.slots
    candidates = []
    create_table = index._table
    write = os.pwrite

    def table(capacity):
        candidate = create_table(capacity)
        candidates.append(candidate)
        return candidate

    def short_write(fd, data, offset):
        if candidates and fd == candidates[-1].fileno():
            return write(fd, data[:written], offset) if written else 0
        return write(fd, data, offset)

    with monkeypatch.context() as patch:
        patch.setattr(index, "_table", table)
        patch.setattr(os, "pwrite", short_write)
        with pytest.raises(OSError, match="Short index write"):
            index.put(b"key-512", 513)

    assert index.slots is original_table
    assert not original_table.closed
    assert index.capacity == 1024
    assert candidates[0].closed
    for number in range(513):
        assert index.get(f"key-{number}".encode()) == number + 1
    index.put(b"key-512", 513)
    assert index.capacity == 2048
    assert original_table.closed
    for number in range(513):
        assert index.get(f"key-{number}".encode()) == number + 1


@pytest.mark.parametrize("written", [0, 8, 24])
def test_short_rehash_seals_session_and_repairs_without_recapture(tmp_path, monkeypatch, written):
    from src.common.types import TrafficSnapshot
    from src.ui.observation_session import ObservationSession
    from test_observation_storage_corrections import (
        FakeANFIS, ScriptedDetector, scripted_payload, step,
    )

    session = ObservationSession(
        scripted_payload(tmp_path), tmp_path / "runs",
        detector_factory=ScriptedDetector, anfis_factory=FakeANFIS,
    )
    try:
        while session.engine.time < 2:
            assert step(session)
        engine = session.engine
        engine.update_system(TrafficSnapshot(
            timestamp=engine.time, v_w_pcu=1, queue_meters=None, occupancy_pct=1,
        ))
        for number in range(514):
            engine.predict(f"r{number}")
            engine.predictions.await_idle()
        index = engine.retention.index
        write = os.pwrite
        hit = []

        def short_write(fd, data, offset):
            if fd not in (index.keys.fileno(), index.slots.fileno()) and not hit:
                hit.append(True)
                return write(fd, data[:written], offset) if written else 0
            return write(fd, data, offset)

        with monkeypatch.context() as patch:
            patch.setattr(os, "pwrite", short_write)
            assert session.persist() is False
        assert hit and engine.closed and session.storage_suspended
        assert session.pending_commit is not None
        durable = session.store.lookup("r0")
        committed = session.store.pin()["committed_bytes"]
        assert session.retry_storage()
        session.stop()
        assert session.finalized
        rows = session.store.read()["predictions"]
        assert len(rows) == len({row["request_id"] for row in rows}) == 514
        assert session.predict("r0") == durable
        assert session.store.pin()["committed_bytes"] >= committed
        assert all(engine.retention.requested(f"r{n}") for n in range(513))
        archived = [r for r in session.store.records() if r["kind"] == "predictions"]
        assert len(archived) == len({r["id"] for r in archived})
    finally:
        session.stop()


def test_short_rehash_retry_preserves_crossed_vehicle_identity(tmp_path, monkeypatch):
    import json
    import numpy as np
    from src.experiment.contracts import VehicleObservation
    from src.experiment.export import ExperimentStore
    from test_observation_storage_corrections import make_engine

    retained, oracle = make_engine(), make_engine()
    retained.enable_retention(1, tmp_path)
    store = ExperimentStore(tmp_path, "run", {})
    frame = np.full((2, 2, 3), (0, 255, 0), np.uint8)

    def tick(sequence, timestamp, observations):
        for engine in (retained, oracle):
            engine.observe_acquisition(frame, timestamp, sequence, observations)

    try:
        tick(0, 0., [])
        tick(1, .2, [VehicleObservation(n, "car", 2., (-1, 0), velocity_valid=True) for n in range(513)])
        tick(2, .4, [VehicleObservation(n, "car", 2., (.5, 0), velocity_valid=True) for n in range(513)])
        state, token = retained.prepare_commit()
        store.persist(state)
        index = retained.retention.index
        write = os.pwrite

        def short_write(fd, data, offset):
            if fd not in (index.keys.fileno(), index.slots.fileno()):
                return write(fd, data[:8], offset)
            return write(fd, data, offset)

        with monkeypatch.context() as patch:
            patch.setattr(os, "pwrite", short_write)
            with pytest.raises(OSError, match="Short index write"):
                retained.acknowledge(token)
        assert len(retained.waits.records) == 513
        retained.acknowledge(token)
        tick(3, 2., [])
        tick(4, 2.2, [VehicleObservation(0, "car", 2., (-1, 0), velocity_valid=True)])
        tick(5, 2.4, [VehicleObservation(0, "car", 2., (.5, 0), velocity_valid=True)])
        retained.close()
        oracle.close()
        state, token = retained.prepare_commit()
        store.persist(state)
        retained.acknowledge(token)
        actual = store.read()
        expected = json.loads(json.dumps(oracle.snapshot(0, 10**6, True)))
        assert actual["vehicles"] == expected["vehicles"]
        assert actual["summary"]["observed_vehicle_count"] == 513
        rows = [r for r in store.records() if r["kind"] == "vehicles"]
        assert len(rows) == len({r["id"] for r in rows}) == 513
    finally:
        retained.close()
        oracle.close()
