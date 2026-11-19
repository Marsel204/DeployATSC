"""One physical vehicle keeps one row: native fragment continuation and
exit-line departure inference. Missing time is never imputed as wait."""
from dataclasses import replace

import numpy as np
import pytest

from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.engine import ObservationEngine
from src.experiment.identity_contracts import IdentityPolicy
from src.experiment.persistent_vehicle_waits import PersistentVehicleWaits
from src.experiment.vehicle_waits import VehicleWaits
from test_vehicle_identity import detection

FPS = 30.
LINE = 100.  # side(point) = y - LINE; positive is past the stop line


def car(local_id, y, x=50., height=40., kind='car', epoch=0, in_region=True, speed=None, valid=False):
    return VehicleObservation(local_id, kind, speed, (x, y), velocity_valid=valid,
                              in_region=in_region, epoch=epoch, height=height)


def book(**policy):
    return VehicleWaits('run', RunPolicy(stop_persistence_sec=0, **policy), side=lambda p: p[1]-LINE)


def play(waits, frames, start=0):
    for index, observations in enumerate(frames, start):
        waits.observe(list(observations), 'RED', index/FPS, index+1, 1)
    return start+len(frames)


def approach(local_id=1, frames=10, y0=60., rate=80., **extra):
    """A vehicle moving toward the line at rate px/s (2 box heights/s)."""
    return [[car(local_id, y0+rate*i/FPS, **extra)] for i in range(frames)]


def only_row(waits):
    assert len(waits.records) == 1, list(waits.records)
    return next(iter(waits.records.values()))


# --- exit-line departure inference -------------------------------------------------

def test_vanishing_before_line_is_an_inferred_departure_backdated_to_the_crossing():
    waits = book()
    end = play(waits, approach())            # last seen t=9/30 at y=84 moving 80 px/s
    play(waits, [[]]*45, end)
    row = only_row(waits)
    last = 9/FPS
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'
    assert row['crossing_at'] == pytest.approx(last+.2, abs=.01)
    assert row['completed_at'] == row['crossing_at']
    assert row['last_seen'] == pytest.approx(last)
    assert 'departure_inferred' in row['quality_flags']
    assert 'tracking_lost' not in row['quality_flags']
    evidence = row['crossing_evidence']
    assert evidence['method'] == 'dead_reckoned'
    assert evidence['samples'] == 8 and evidence['last_time'] == pytest.approx(last)
    assert evidence['approach_px_per_sec'] == pytest.approx(80, rel=.01)
    # The hidden span is unknown coverage, never wait.
    gap = row['intervals'][-1]
    assert gap['valid'] is False and gap['phase'] == 'UNKNOWN'
    assert gap['start'] == pytest.approx(last) and gap['end'] == pytest.approx(row['crossing_at'])
    assert row['red_wait_sec'] == 0 and row['unknown_sec'] == pytest.approx(.2, abs=.01)
    assert row['crossing_phase'] == 'UNKNOWN' and row['crossing_cycle'] is None


def test_observed_crossing_is_labelled_observed():
    waits = book()
    play(waits, approach(frames=25))
    row = only_row(waits)
    assert row['crossed_stop_line'] and row['crossing_evidence']['method'] == 'observed'
    assert 'departure_inferred' not in row['quality_flags']


def test_decision_waits_for_confirmation_so_a_return_cancels_it():
    waits = book()
    end = play(waits, approach())
    end = play(waits, [[]]*12, end)              # past the predicted crossing, not yet confirmed
    row = only_row(waits)
    assert not row['completed']
    play(waits, [[car(1, 99.)]], end)            # the same ID comes back before the line
    assert not row['completed'] and not row['crossed_stop_line']
    assert 'departure_inferred' not in row['quality_flags']


@pytest.mark.parametrize('name,frames', [
    ('stationary', [[car(1, 84.)] for _ in range(10)]),
    ('moving_away', approach(y0=84., rate=-80.)),
    ('sideways', [[car(1, 84., x=50+80*i/FPS)] for i in range(10)]),
    ('crossing_beyond_horizon', approach(y0=20., rate=40.)),
    ('slow_creep', approach(y0=93., rate=10.)),
    ('too_few_samples', approach(frames=2, y0=84.)),
    ('unknown_height', approach(height=None)),
])
def test_no_departure_is_inferred_without_motion_evidence(name, frames):
    waits = book()
    play(waits, [*frames, *[[]]*45])
    row = only_row(waits)
    assert not row['crossed_stop_line'], name
    assert 'departure_inferred' not in row['quality_flags']
    assert 'tracking_lost' in row['quality_flags']


def test_stopped_vehicle_is_not_inferred_to_depart():
    waits = book()
    frames = [[car(1, 60+80*i/FPS, speed=0., valid=True)] for i in range(10)]
    play(waits, [*frames, *[[]]*45])
    assert not only_row(waits)['crossed_stop_line']


def test_departure_inference_switch_restores_old_behaviour():
    waits = book(departure_inference=False)
    play(waits, [*approach(), *[[]]*45])
    row = only_row(waits)
    assert not row['crossed_stop_line'] and 'tracking_lost' in row['quality_flags']


def test_occupied_crossing_point_blocks_inference():
    """Another tracked vehicle where this one should be: it was probably hidden behind it."""
    waits = book()
    frames = [[car(1, 60+80*i/FPS), car(9, 60+80*i/FPS, x=58.)] for i in range(10)]
    frames += [[car(9, 60+80*i/FPS, x=58.)] for i in range(10, 55)]
    play(waits, frames)
    lost = waits.records['run:acquisition:0:1']
    assert not lost['crossed_stop_line']
    assert 'departure_inference_blocked' in lost['quality_flags']
    assert waits.records['run:acquisition:0:9']['crossing_evidence']['method'] == 'observed'


def test_inferred_departure_identity_returning_does_not_open_a_second_row():
    waits = book()
    end = play(waits, [*approach(), *[[]]*45])
    play(waits, [[car(1, 120.)]], end)
    assert len(waits.records) == 1


# --- native fragment continuation ---------------------------------------------------

def fragment(gap=3, new_id=2, rate=80., y0=40., **new):
    frames = approach(1, 10, y0=y0, rate=rate)
    frames += [[]]*gap
    t0 = 10+gap
    frames += [[car(new_id, y0+rate*(t0+i)/FPS, **new)] for i in range(4)]
    return frames


def test_unique_newborn_continues_the_lost_vehicle_row():
    waits = book()
    play(waits, fragment())
    row = only_row(waits)
    assert row['vehicle_id'] == 'run:acquisition:0:1' and row['local_id'] == 1
    assert row['native_ids'] == [{'epoch': 0, 'local_id': 1}, {'epoch': 0, 'local_id': 2}]
    assert 'identity_continued' in row['quality_flags']
    link, = row['identity_links']
    assert link['from'] == {'epoch': 0, 'local_id': 1} and link['to'] == {'epoch': 0, 'local_id': 2}
    assert link['gap_sec'] == pytest.approx(4/FPS)
    assert link['error_heights'] <= link['gate_heights']
    gaps = [i for i in row['intervals'] if not i['valid']]
    assert len(gaps) == 1 and gaps[0]['phase'] == 'UNKNOWN'
    assert gaps[0]['end']-gaps[0]['start'] == pytest.approx(4/FPS)
    assert row['red_wait_sec'] == 0
    assert row['last_seen'] == pytest.approx(16/FPS)
    assert waits.physical((0, 2))['vehicle_id'] == row['vehicle_id']


def test_continued_vehicle_crossing_is_counted_once_on_the_original_row():
    waits = book()
    frames = fragment()
    frames += [[car(2, 40+80*i/FPS)] for i in range(17, 30)]
    play(waits, frames)
    row = only_row(waits)
    assert row['crossed_stop_line'] and row['crossing_evidence']['method'] == 'observed'


@pytest.mark.parametrize('name,frames', [
    ('class_mismatch', fragment(kind='motorcycle')),
    ('size_mismatch', fragment(height=70.)),
    ('far_from_prediction', fragment(x=90.)),
    ('gap_too_long', fragment(gap=16)),
    ('new_tracker_epoch', fragment(epoch=1)),
    ('unknown_height', fragment(height=None)),
])
def test_continuation_is_refused_without_a_physical_match(name, frames):
    waits = book()
    play(waits, frames)
    assert len(waits.records) == 2, name
    assert not any(r['identity_links'] for r in waits.records.values())


def test_gap_boundary_at_max_gap_is_linked_and_one_frame_more_is_not():
    # gap=14 missing frames -> 15 frames (0.5 s) between observations
    waits = book()
    play(waits, fragment(gap=14))
    assert len(waits.records) == 1
    waits = book()
    play(waits, fragment(gap=15))
    assert len(waits.records) == 2


def test_competing_newborns_are_not_linked():
    waits = book()
    frames = approach(1, 10, y0=40.) + [[]]*3
    frames += [[car(2, 40+80*(13+i)/FPS), car(3, 40+80*(13+i)/FPS, x=56.)] for i in range(4)]
    play(waits, frames)
    assert len(waits.records) == 3
    assert not any(r['identity_links'] for r in waits.records.values())


def test_competing_lost_vehicles_are_not_linked():
    waits = book()
    frames = [[car(1, 40+80*i/FPS), car(5, 40+80*i/FPS, x=56.)] for i in range(10)] + [[]]*3
    frames += [[car(2, 40+80*(13+i)/FPS, x=53.)] for i in range(4)]
    play(waits, frames)
    assert len(waits.records) == 3
    assert not any(r['identity_links'] for r in waits.records.values())


def test_simultaneous_ids_are_two_vehicles():
    waits = book()
    frames = [[car(1, 40+80*i/FPS)] for i in range(10)]
    frames += [[car(1, 40+80*(10+i)/FPS), car(2, 40+80*(10+i)/FPS, x=52.)] for i in range(4)]
    play(waits, frames)
    assert len(waits.records) == 2


def test_old_native_id_reappearing_starts_a_separate_segment():
    waits = book()
    end = play(waits, fragment())
    continued = waits.records['run:acquisition:0:1']
    before = dict(continued, intervals=list(continued['intervals']))
    play(waits, [[car(2, 40+80*(17+i)/FPS), car(1, 30.)] for i in range(3)], end)
    assert set(waits.records) == {'run:acquisition:0:1', 'run:acquisition:0:1:segment-1'}
    reused = waits.records['run:acquisition:0:1:segment-1']
    assert reused['local_id'] == 1 and 'native_id_reused' in reused['quality_flags']
    assert not reused['identity_links']
    assert continued['first_seen'] == before['first_seen']
    assert continued['last_seen'] == pytest.approx((end+2)/FPS)
    assert waits.physical((0, 1))['vehicle_id'] == reused['vehicle_id']
    assert waits.physical((0, 2))['vehicle_id'] == continued['vehicle_id']


def test_new_row_never_overwrites_an_existing_record():
    waits = book()
    end = play(waits, fragment())
    # Native 2 disappears; long after, native 1 is reused far away.
    end = play(waits, [[]]*40, end)
    play(waits, [[car(1, 20.)]], end)
    assert len(waits.records) == 2
    assert waits.records['run:acquisition:0:1']['identity_links']


def test_continuation_switch_restores_old_behaviour():
    waits = book(fragment_continuation=False)
    play(waits, fragment())
    assert len(waits.records) == 2


def test_invalidation_breaks_continuation():
    waits = book()
    end = play(waits, approach(1, 10, y0=40.))
    waits.invalidate(end/FPS, 'live_resume')
    play(waits, [[car(2, 40+80*(end+1+i)/FPS)] for i in range(3)], end+1)
    assert len(waits.records) == 2


def test_policy_switches_are_validated():
    with pytest.raises(ValueError):RunPolicy(departure_inference='yes')
    with pytest.raises(ValueError):RunPolicy(fragment_continuation=1)


# --- persistent identities --------------------------------------------------------

def test_persistent_waits_never_continue_native_fragments():
    waits = PersistentVehicleWaits('run', RunPolicy(stop_persistence_sec=0), lambda p: p[1]-LINE,
                                   IdentityPolicy(mode='persistent_v1'))
    assert not waits.fragment_continuation


def persistent_engine(**policy):
    return ObservationEngine('departure', RunPolicy(debounce_frames=1, stop_persistence_sec=0, **policy),
        {'xmin': 0, 'ymin': 0, 'xmax': 8, 'ymax': 8}, lambda p: p[1]-LINE, lambda *_: None,
        identity_policy=IdentityPolicy(mode='persistent_v1', confirmation_observations=2,
                                       confirmation_sec=.1, exit_persistence_sec=0))


def test_persistent_identity_departure_is_inferred_from_resolved_boxes():
    engine = persistent_engine()
    frame = np.full((8, 8, 3), (0, 0, 255), np.uint8)
    for seq in range(1, 11):
        y = 60+80*(seq-1)/FPS
        item = replace(detection(), bbox=(40, y-40, 60, y), point=(50, y), ground=(5, y/10))
        engine.observe_acquisition(frame, (seq-1)/FPS, seq, (), identity_evidence=(item,))
    for seq in range(11, 80):
        engine.observe_acquisition(frame, (seq-1)/FPS, seq, (), identity_evidence=())
    row = only_row(engine.waits)
    assert row['crossed_stop_line'] and 'departure_inferred' in row['quality_flags']
    assert row['crossing_evidence']['method'] == 'dead_reckoned'
    assert row['crossing_at'] == pytest.approx(9/FPS+.2, abs=.01)
    assert row['identity_state'] == 'CROSSED'
    # The hidden span ends at the inferred crossing and stays unresolved coverage.
    assert row['intervals'][-1]['end'] == pytest.approx(row['crossing_at'])
    assert row['estimated_red_wait_sec'] == 0


def test_native_engine_snapshot_exposes_physical_vehicle_and_raw_ids():
    engine = ObservationEngine('run', RunPolicy(debounce_frames=1, stop_persistence_sec=0),
        {'xmin': 0, 'ymin': 0, 'xmax': 8, 'ymax': 8}, lambda p: p[1]-LINE, lambda *_: None)
    frame = np.full((8, 8, 3), (0, 0, 255), np.uint8)
    for index, observations in enumerate(fragment()):
        engine.observe_acquisition(frame, index/FPS, index+1, observations)
    vehicles = engine.snapshot(evidence=True)['vehicles']
    assert len(vehicles) == 1
    assert [n['local_id'] for n in vehicles[0]['native_ids']] == [1, 2]
