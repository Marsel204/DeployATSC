"""Physical transfer guards reproduced from the moving queue, plus image motion."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy
from src.experiment.vehicle_identity import VehicleIdentityRegistry
from test_vehicle_identity import admit, detection, registry


CASES = json.loads((Path(__file__).parent/'fixtures/moving_queue_transfers.json').read_text())['cases']


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['baseline_id'])
def test_recorded_wrong_vehicle_transition_is_not_a_recovery(case):
    book = VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1'), ground_trusted=False)
    for row in case['history']:
        book.resolve((IdentityDetection(**row['detection']),), row['timestamp'], row['sequence'])
    entry = min(book.entries.values(), key=lambda e: e.identity)
    assert entry.admitted, 'Recorded pre-transfer track must be admitted'
    assert book._cost(entry, IdentityDetection(**case['candidate']), case['timestamp']) is None


def test_unreliable_new_native_cannot_inherit_an_admitted_timer():
    book = registry()
    identity = admit(book)
    partial = replace(detection(raw=228), appearance_reliable=False)
    result = book.resolve((partial,), .2, 3)
    assert not any(v.logical_id == identity for v in result.vehicles)


def test_geometry_advances_while_appearance_is_occluded():
    book = registry()
    identity = admit(book)
    for seq in range(3, 23):
        item = replace(detection(x=50+2*(seq-2)), appearance_reliable=False)
        result = book.resolve((item,), (seq-1)/10, seq)
        assert any(v.logical_id == identity for v in result.vehicles)


def test_image_motion_rejects_lateral_transfer_without_a_ground_projection():
    book = registry()
    identity = None
    for seq in range(1, 10):
        item = replace(detection(x=50+2*seq), ground=None)
        result = book.resolve((item,), (seq-1)/10, seq)
        if result.vehicles:identity = result.vehicles[0].logical_id
    entry = book.entries[identity]
    rival = replace(detection(raw=228, x=86), ground=None)
    assert book._cost(entry, rival, .9) is None
    supported = replace(detection(raw=228, x=70), ground=None)
    result = book.resolve((supported,), .9, 10)
    assert result.vehicles[0].logical_id == identity
    assert not result.vehicles[0].velocity_valid
    assert result.vehicles[0].speed_mps is None


def test_image_association_is_invariant_to_unverified_metric_scale():
    books = [VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1'), ground_trusted=False)
        for _ in range(2)]
    for seq in range(1, 12):
        results = [book.resolve((replace(detection(x=50+seq), ground=(scale*seq, 0)),),
            (seq-1)/10, seq) for book, scale in zip(books, (1, 1000))]
        assert [v.logical_id for v in results[0].vehicles] == [v.logical_id for v in results[1].vehicles]
    for book in books:
        rival = replace(detection(raw=228, x=79), ground=(0, 0))
        assert book._cost(next(iter(book.entries.values())), rival, 1.1) is None


def test_sparse_return_cannot_erase_known_motion_and_enable_a_later_transfer():
    book = registry()
    for seq in range(1, 10):
        book.resolve((replace(detection(x=50+2*seq), ground=None),), (seq-1)/10, seq)
    identity = next(iter(book.entries))
    result = book.resolve((replace(detection(x=72), ground=None),), 1., 11)
    assert result.vehicles[0].logical_id == identity
    rival = replace(detection(raw=228, x=72), ground=None)
    assert book._cost(book.entries[identity], rival, 2.) is None


def test_native_change_needs_close_geometry_when_motion_is_not_available():
    book = registry()
    identity = admit(book, replace(detection(), ground=None))
    assert book._cost(book.entries[identity], replace(detection(raw=228, x=71), ground=None), .2) is None


def test_repeated_served_retirement_preserves_post_crossing_image_motion():
    book = registry()
    identity = admit(book)
    for seq in range(3, 12):
        result = book.resolve((replace(detection(x=50+2*(seq-2)), ground=None),), (seq-1)/10, seq)
        assert result.vehicles[0].logical_id == identity
        book.retire(0, identity, crossed=True)
    entry = book.entries[identity]
    rival = replace(detection(raw=228, x=68), ground=None)
    assert book._cost(entry, rival, 2.) is None
