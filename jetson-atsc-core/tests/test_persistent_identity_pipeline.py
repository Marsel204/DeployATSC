import numpy as np

from src.experiment.contracts import RunPolicy
from src.experiment.engine import ObservationEngine
from src.experiment.identity_contracts import IdentityPolicy
from test_vehicle_identity import detection


def engine():
    return ObservationEngine('logical', RunPolicy(debounce_frames=1, stop_persistence_sec=0),
        {'xmin': 0, 'ymin': 0, 'xmax': 10, 'ymax': 10}, lambda p: -1, lambda *a: None,
        identity_policy=IdentityPolicy(mode='persistent_v1', confirmation_observations=2, confirmation_sec=.1))


def tick(book, sequence, timestamp, items):
    frame = np.full((10, 10, 3), (0, 0, 255), np.uint8)
    return book.observe_acquisition(frame, timestamp, sequence, (), identity_evidence=items)


def test_engine_ids_motion_timers_and_preview_decisions_are_one_transaction():
    book = engine()
    for seq in range(1, 31):
        result = tick(book, seq, (seq-1)/10, (detection(4 if seq < 15 else 228),))
    rows = book.snapshot(evidence=True)['vehicles']
    assert len(rows) == 1 and rows[0]['red_wait_sec'] > 2
    assert result.vehicles[0].logical_id == rows[0]['local_id']
    before = book.snapshot(evidence=True)
    tick(book, 30, 2.9, (detection(999),))
    assert book.snapshot(evidence=True) == before
    book.close()


def test_camera1_time_cannot_expire_b_and_hard_boundary_blocks_stale_work():
    book = engine()
    for seq in range(1, 11):
        tick(book, seq, (seq-1)/10, (detection(),))
    book.observe_system_frame(100)
    tick(book, 11, 1., (detection(228),))
    assert len(book.waits.records) == 1
    generation = book.generation
    book.invalidate('acquisition_stale', 1.)
    result = book.observe_acquisition(np.zeros((10, 10, 3), np.uint8), 1.1, 12, (),
        generation=generation, identity_evidence=(detection(),))
    assert result is None
    assert all(row['completed'] for row in book.waits.records.values())
    book.close()
