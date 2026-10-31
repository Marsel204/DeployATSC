import pytest

from src.experiment.contracts import RunPolicy
from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy, ResolvedVehicle
from src.experiment.persistent_vehicle_waits import PersistentVehicleWaits


def observed(raw=4, speed=0, valid=True):
    item = IdentityDetection((10, 10, 30, 50), (20, 50), native_id=raw,
        ground=(2, 5), appearance=(1., 0.), appearance_reliable=True)
    return ResolvedVehicle(1, 0, item, speed, valid)


def ledger(cap=5):
    return PersistentVehicleWaits('run', RunPolicy(stop_persistence_sec=0), lambda p: -1,
        IdentityPolicy(mode='persistent_v1', max_estimated_gap_sec=cap))


def drive(book, start, end, absent=False, raw=4, phase='RED', speed=0):
    for index in range(start, end):
        book.observe_resolved(() if absent else (observed(raw, speed),), phase, index/10, index+1, 1)


def test_oracle_nine_confirmed_three_estimated_one_timer():
    book = ledger()
    drive(book, 0, 41)
    drive(book, 41, 70, absent=True)
    drive(book, 70, 121, raw=228)
    row = book.rows(include_intervals=True)[0]
    assert len(book.records) == 1
    assert row['red_wait_sec'] == pytest.approx(9)
    assert row['estimated_red_wait_sec'] == pytest.approx(3)
    assert row['display_red_wait_sec'] == pytest.approx(12)
    assert row['pending_estimated_red_wait_sec'] == 0
    assert row['unknown_sec'] >= 3 - 1e-8
    before = row['display_red_wait_sec']
    book.observe_resolved((observed(228),), 'RED', 12, 121, 1)
    assert book.rows()[0]['display_red_wait_sec'] == before


def test_cap_withdraws_whole_provisional_gap_without_resetting_confirmed():
    book = ledger(cap=1)
    drive(book, 0, 11)
    drive(book, 11, 16, absent=True)
    assert book.rows()[0]['pending_estimated_red_wait_sec'] > 0
    drive(book, 16, 26, absent=True)
    row = book.rows()[0]
    assert row['pending_estimated_red_wait_sec'] == 0
    assert row['red_wait_sec'] == pytest.approx(1)
    drive(book, 26, 32, raw=228)
    assert book.rows()[0]['estimated_red_wait_sec'] == 0


def test_phase_unknown_and_green_hidden_time_do_not_receive_red_estimates():
    book = ledger()
    drive(book, 0, 11)
    drive(book, 11, 16, absent=True)
    drive(book, 16, 21, absent=True, phase='UNKNOWN')
    drive(book, 21, 26, absent=True, phase='GREEN')
    drive(book, 26, 32, raw=228, phase='GREEN')
    row = book.rows()[0]
    assert row['estimated_red_wait_sec'] == pytest.approx(.6)


def test_moving_return_rejects_gap_but_retains_confirmed_wait():
    book = ledger()
    drive(book, 0, 11)
    drive(book, 11, 21, absent=True)
    drive(book, 21, 27, raw=228, speed=2)
    row = book.rows()[0]
    assert row['estimated_red_wait_sec'] == 0
    assert row['red_wait_sec'] == pytest.approx(1)


def test_source_boundary_finalizes_partial_rows_without_hidden_credit():
    book = ledger()
    drive(book, 0, 11)
    drive(book, 11, 21, absent=True)
    book.invalidate(2., 'acquisition_stale')
    row = book.rows()[0]
    assert row['completed'] and row['pending_estimated_red_wait_sec'] == 0
    assert row['estimated_red_wait_sec'] == 0
    assert row['red_wait_sec'] == pytest.approx(1)
