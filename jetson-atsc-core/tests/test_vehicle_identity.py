from dataclasses import replace

import pytest

from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy
from src.experiment.vehicle_identity import VehicleIdentityRegistry


def detection(raw=4, x=50, appearance=(1., 0.), **extra):
    return IdentityDetection((x-10, 10, x+10, 50), (x, 50), native_id=raw,
        ground=(x/10, 5), appearance=appearance, appearance_reliable=True, **extra)


def registry(**kwargs):
    return VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1',
        confirmation_observations=2, confirmation_sec=.1, **kwargs), max_gap_sec=.5)


def admit(book, item=None):
    item = item or detection()
    book.resolve((item,), 0., 1)
    result = book.resolve((item,), .1, 2)
    assert len(result.vehicles) == 1
    return result.vehicles[0].logical_id


def test_native_replacement_preserves_logical_identity_and_motion():
    book = registry()
    identity = admit(book)
    for seq in range(3, 9):
        item = detection(raw=228 if seq > 4 else 4)
        resolved = book.resolve((item,), (seq-1)/10, seq).vehicles[0]
        assert resolved.logical_id == identity
    assert resolved.velocity_valid and resolved.speed_mps == pytest.approx(0)


def test_supported_gap_retains_id_but_requires_motion_warmup():
    book = registry()
    identity = admit(book)
    for seq in range(3, 23):
        book.resolve((), (seq-1)/10, seq)
    result = book.resolve((detection(raw=228),), 2.2, 23).vehicles[0]
    assert result.logical_id == identity
    assert result.recovered and not result.velocity_valid


def test_native_id_cannot_transfer_timer_to_different_appearance():
    book = registry()
    identity = admit(book)
    other = detection(appearance=(0., 1.))
    book.resolve((other,), .2, 3)
    result = book.resolve((other,), .3, 4)
    assert result.vehicles and result.vehicles[0].logical_id != identity


def test_equal_competing_candidates_abstain():
    book = registry()
    for seq in range(1, 3):
        book.resolve((detection(4, 40), detection(5, 60)), (seq-1)/10, seq)
    book.resolve((), .2, 3)
    result = book.resolve((detection(228, 50),), .3, 4)
    assert not result.vehicles
    assert result.unresolved


def test_duplicate_packets_are_idempotent_and_generation_does_not_reuse_ids():
    book = registry()
    identity = admit(book)
    assert book.resolve((detection(raw=88),), .1, 2).duplicate
    book.invalidate('source_reset')
    book.resolve((detection(),), .2, 3)
    resolved = book.resolve((detection(),), .3, 4).vehicles[0]
    assert resolved.logical_id > identity and resolved.generation == 1


def test_capacity_does_not_evict_active_identity():
    book = registry(capacity=1)
    identity = admit(book)
    result = book.resolve((detection(), detection(9, 250)), .2, 3)
    assert result.vehicles[0].logical_id == identity
    assert result.capacity_exhausted


def test_epoch_change_alone_and_corridor_jitter_do_not_reset_identity():
    book = registry()
    identity = admit(book)
    item = replace(detection(), native_epoch=9, in_region=False)
    result = book.resolve((item,), .2, 3).vehicles[0]
    assert result.logical_id == identity and result.exit_pending
    result = book.resolve((detection(),), .3, 4).vehicles[0]
    assert result.logical_id == identity and not result.exit_pending


def test_expired_identity_cannot_resurrect_and_memory_is_bounded():
    book = registry(occlusion_retention_sec=1, max_estimated_gap_sec=.5)
    identity = admit(book)
    book.resolve((), 2., 3)
    book.resolve((detection(),), 2.1, 4)
    result = book.resolve((detection(),), 2.2, 5).vehicles[0]
    assert result.logical_id > identity
    assert len(book.entries) <= book.policy.capacity


def test_partial_return_does_not_admit_a_rival_for_an_occluded_vehicle():
    book=registry()
    identity=admit(book)
    for seq in range(3,20):book.resolve((),(seq-1)/10,seq)
    partial=replace(detection(raw=166),appearance_reliable=False)
    for seq in range(20,26):
        assert not book.resolve((partial,),(seq-1)/10,seq).vehicles
    result=book.resolve((detection(raw=166),),2.5,26)
    assert result.vehicles[0].logical_id==identity


def test_creeping_speed_is_measured_on_logical_history_without_identity_reset():
    book=registry()
    identity=admit(book)
    for seq in range(3,12):
        result=book.resolve((detection(raw=228,x=50+.6*(seq-2)),),(seq-1)/10,seq).vehicles[0]
        assert result.logical_id==identity
    assert result.velocity_valid and result.speed_mps==pytest.approx(.6)
