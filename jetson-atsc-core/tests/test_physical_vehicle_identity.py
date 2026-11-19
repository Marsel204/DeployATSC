"""Opt-in native fragment continuation: one vehicle row across native ID
fragments, without imputing wait, crossings or motion across a handoff, and
with every raw ID keeping its own history through archive eviction."""
import time

import numpy as np
import pytest

from src.experiment import vehicle_waits as vw
from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.identity_contracts import IdentityPolicy
from src.experiment.persistent_vehicle_waits import PersistentVehicleWaits
from src.experiment.retention import Retention
from src.experiment.vehicle_waits import VehicleWaits

FPS = 30.
LINE = 100.  # side(point) = y - LINE; positive is past the stop line
RED = np.full((8, 8, 3), (0, 0, 255), np.uint8)
GREEN = np.full((8, 8, 3), (0, 255, 0), np.uint8)


def car(local_id, y, x=50., height=40., kind='car', epoch=0, in_region=True, speed=2., valid=True,
        stable=True, width=None, bbox=True):
    box = None
    if bbox and height is not None:
        half = (width if width is not None else height*1.2)/2
        box = (x-half, y-height, x+half, y)
    return VehicleObservation(local_id, kind, speed, (x, y), stable_id=stable, velocity_valid=valid,
                              in_region=in_region, epoch=epoch, height=height, bbox=box)


def book(**policy):
    policy = {'stop_persistence_sec': 0, 'fragment_continuation': True, **policy}
    return VehicleWaits('run', RunPolicy(**policy), side=lambda p: p[1]-LINE)


def play(waits, frames, start=0, phase='RED'):
    for index, observations in enumerate(frames, start):
        waits.observe(list(observations), phase, index/FPS, index+1, 1)
    return start+len(frames)


def approach(local_id=1, frames=10, y0=60., rate=80., start=0, **extra):
    """A vehicle moving toward the line at rate px/s (2 box heights/s)."""
    return [[car(local_id, y0+rate*(start+i)/FPS, **extra)] for i in range(frames)]


def only_row(waits):
    assert len(waits.records) == 1, list(waits.records)
    return next(iter(waits.records.values()))


def fragment(gap=3, new_id=2, rate=80., y0=40., **new):
    frames = approach(1, 10, y0=y0, rate=rate)
    frames += [[]]*gap
    t0 = 10+gap
    frames += [[car(new_id, y0+rate*(t0+i)/FPS, **new)] for i in range(4)]
    return frames


# --- switch, defaults and unchanged behaviour ---------------------------------------

def test_continuation_is_opt_in_and_validated():
    assert RunPolicy().fragment_continuation is False
    assert not hasattr(RunPolicy(), 'departure_inference')
    with pytest.raises(ValueError):RunPolicy(fragment_continuation=1)
    with pytest.raises(TypeError):RunPolicy(departure_inference=True)
    assert RunPolicy(fragment_continuation=True).fragment_continuation is True


def test_default_policy_keeps_one_row_per_native_id():
    waits = VehicleWaits('run', RunPolicy(stop_persistence_sec=0), side=lambda p: p[1]-LINE)
    play(waits, fragment())
    assert len(waits.records) == 2
    assert not any('native_ids' in r or 'identity_links' in r for r in waits.records.values())
    assert waits.handed_off == {} and waits.outside_seen == {}


def test_switch_off_rows_are_identical_to_a_run_without_box_evidence():
    with_boxes, without = book(fragment_continuation=False), book(fragment_continuation=False)
    play(with_boxes, fragment()+[[car(2, 40+80*i/FPS)] for i in range(17, 30)])
    play(without, [[VehicleObservation(o.local_id, o.class_name, o.speed_mps, o.point, stable_id=True, velocity_valid=True)
                    for o in f] for f in fragment()+[[car(2, 40+80*i/FPS)] for i in range(17, 30)]])
    assert with_boxes.records == without.records
    assert all('history' not in s for s in with_boxes.active.values())


@pytest.mark.parametrize('continuation', [False, True])
def test_a_vanishing_vehicle_is_never_a_guessed_departure(continuation):
    """Departure inference was removed: absence before the line is tracking loss."""
    waits = book(fragment_continuation=continuation)
    play(waits, [*approach(), *[[]]*45])
    row = only_row(waits)
    assert not row['crossed_stop_line'] and row['tracking_state'] == 'LOST'
    assert 'tracking_lost' in row['quality_flags']
    assert row['crossing_evidence'] is None


def test_observed_crossing_evidence_keeps_the_existing_schema():
    waits = book()
    play(waits, approach(frames=25))
    evidence = only_row(waits)['crossing_evidence']
    assert set(evidence) == {'from_point', 'to_point', 'from_time', 'to_time', 'from_side', 'to_side'}


# --- linking -----------------------------------------------------------------------

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
    assert waits.physical((0, 2))['vehicle_id'] == row['vehicle_id']
    assert waits.physical((0, 1)) is None
    assert waits.handed_off == {(0, 1): {'row_id': row['vehicle_id'], 'segments': 1}}


def test_continued_vehicle_crossing_is_observed_on_the_new_id_only():
    waits = book()
    play(waits, fragment()+[[car(2, 40+80*i/FPS)] for i in range(17, 30)])
    row = only_row(waits)
    assert row['crossed_stop_line']
    assert row['crossing_evidence']['from_time'] >= row['identity_links'][0]['linked_at']


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
    assert not any(r.get('identity_links') for r in waits.records.values())


def test_gap_boundary_at_max_gap_is_linked_and_one_frame_more_is_not():
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
    assert not any(r.get('identity_links') for r in waits.records.values())


def test_competing_lost_vehicles_are_not_linked():
    waits = book()
    frames = [[car(1, 40+80*i/FPS), car(5, 40+80*i/FPS, x=56.)] for i in range(10)] + [[]]*3
    frames += [[car(2, 40+80*(13+i)/FPS, x=53.)] for i in range(4)]
    play(waits, frames)
    assert len(waits.records) == 3
    assert not any(r.get('identity_links') for r in waits.records.values())


def test_simultaneous_ids_are_two_vehicles():
    waits = book()
    frames = [[car(1, 40+80*i/FPS)] for i in range(10)]
    frames += [[car(1, 40+80*(10+i)/FPS), car(2, 40+80*(10+i)/FPS, x=52.)] for i in range(4)]
    play(waits, frames)
    assert len(waits.records) == 2


def test_newborn_competitor_of_another_size_still_makes_the_link_ambiguous():
    """f66 regression: N67's newborn #296 shared its body with #297 (99% of the
    smaller box inside the larger, footpoints 0.04 heights apart). #297 failed
    the size filter, which used to run before the uniqueness check."""
    waits = book()
    y = 40+80*13/FPS
    frames = approach(1, 10, y0=40.) + [[]]*3
    frames += [[car(2, y, x=50.), car(3, y+1., x=50.5, height=72., width=40.)]]
    play(waits, frames)
    assert len(waits.records) == 3 and not any(r.get('identity_links') for r in waits.records.values())


@pytest.mark.parametrize('competitor', [
    car(None, 40+80*13/FPS+1, x=51., stable=False),                  # unconfirmed detection on the body
    car(9, 40+80*13/FPS+1, x=51., kind='truck', height=90., width=60.),   # tracked box of another class
])
def test_any_observed_box_on_the_same_body_blocks_the_link(competitor):
    waits = book()
    frames = [[car(1, 40+80*i/FPS)] + ([car(9, 300.)] if competitor.local_id == 9 else []) for i in range(10)]
    frames += [[]]*3 + [[car(2, 40+80*13/FPS), competitor]]
    play(waits, frames)
    assert not any(r.get('identity_links') for r in waits.records.values())


def test_overlapping_box_with_a_distinct_footpoint_does_not_block():
    """#191->#213 (f44): the overlapping #29 stood more than a box height away."""
    waits = book()
    frames = [[car(1, 40+80*i/FPS), car(9, 140., x=56., height=60.)] for i in range(10)]
    frames += [[car(9, 140., x=56., height=60.)]]*3
    frames += [[car(2, 40+80*(13+i)/FPS), car(9, 140., x=56., height=60.)] for i in range(4)]
    play(waits, frames)
    assert waits.records['run:acquisition:0:1']['native_ids'][-1] == {'epoch': 0, 'local_id': 2}


def test_raw_ids_seen_outside_the_region_are_not_newborns():
    waits = book()
    frames = approach(1, 10, y0=40.) + [[car(2, 0., in_region=False)]] + [[]]*2
    frames += [[car(2, 40+80*(13+i)/FPS)] for i in range(4)]
    play(waits, frames)
    assert len(waits.records) == 2


def test_invalidation_breaks_continuation():
    waits = book()
    end = play(waits, approach(1, 10, y0=40.))
    waits.invalidate(end/FPS, 'live_resume')
    play(waits, [[car(2, 40+80*(end+1+i)/FPS)] for i in range(3)], end+1)
    assert len(waits.records) == 2


def test_without_motion_evidence_only_a_very_short_gap_is_bridged():
    for gap, linked in ((2, True), (6, False)):
        waits = book()
        play(waits, [[car(1, 60.)]]*3 + [[]]*gap + [[car(2, 64.)]])
        assert (len(waits.records) == 1) == linked, gap


@pytest.mark.parametrize('samples', [1, 2])
def test_a_fragment_seen_in_fewer_than_three_frames_is_not_continued(samples):
    """Junction f475: raw #1717, a one-frame sliver at the image edge, was handed to
    the white car #1721 two frames later and came back itself at f499."""
    waits = book()
    play(waits, [[car(1, 60.)]]*samples + [[]]*2 + [[car(2, 62.)]])
    assert len(waits.records) == 2


def test_stationary_vehicle_is_bridged_only_across_flicker():
    for gap, linked in ((2, True), (8, False)):
        waits = book()
        play(waits, [[car(1, 60.)] for _ in range(10)] + [[]]*gap + [[car(2, 66.)]])
        assert (len(waits.records) == 1) == linked, gap


# --- nothing is credited across a heuristic handoff ---------------------------------

def test_adjacent_frame_handoff_across_the_line_is_not_an_observed_crossing():
    """Sol probe: ID 1 at y=97 then ID 2 at y=102 on consecutive green frames."""
    waits = book()
    frames = [[car(1, 73.+3*i, speed=2.)] for i in range(9)]       # ... y=97 at frame 8
    frames += [[car(2, 102.)]]
    play(waits, frames, phase='GREEN')
    row = only_row(waits)
    assert row['identity_links'][0]['gap_sec'] == pytest.approx(1/FPS)
    assert not row['crossed_stop_line']
    handoff = row['intervals'][-1]
    assert handoff['valid'] is False and handoff['phase'] == 'UNKNOWN' and handoff['cycle_id'] is None
    assert 'observation_gap' in row['quality_flags']


def test_adjacent_frame_handoff_leaving_the_region_is_not_a_crossing():
    waits = book()
    frames = [[car(1, 73.+3*i)] for i in range(9)] + [[car(2, 99.)]] + [[car(2, 102., in_region=False)]]
    play(waits, frames, phase='GREEN')
    row = only_row(waits)
    assert row['identity_links'] and row['crossed_stop_line']   # the new ID crossed on its own frames
    assert row['crossing_evidence']['from_time'] == pytest.approx(9/FPS)


def test_stop_state_and_wait_are_not_carried_across_a_handoff():
    waits = book(stop_persistence_sec=.3)
    stopped = [[car(1, 70.+.01*i, speed=0.)] for i in range(30)]             # stopped for 1 s
    end = play(waits, stopped)
    before = only_row(waits)['red_wait_sec']
    assert before > .5
    play(waits, [[car(2, 70.4, speed=0.)] for _ in range(5)], end+1)    # handoff after one empty frame
    row = only_row(waits)
    assert row['identity_links']
    # The handoff gap and the new ID's frames before stop persistence are not red wait.
    assert row['red_wait_sec'] == pytest.approx(before)
    assert row['unknown_sec'] == pytest.approx(2/FPS)


def test_motion_history_is_not_projected_across_a_handoff():
    waits = book()
    end = play(waits, fragment())
    state = waits.active[(0, 2)]
    assert [round(h[0]*FPS) for h in state['history']] == [13, 14, 15, 16]
    assert all(h[0] >= state['row']['identity_links'][0]['linked_at'] for h in state['history'])
    # Moments after a link the fresh history is too short for another long-gap link.
    waits2 = book()
    end = play(waits2, approach(1, 10, y0=40.) + [[]]*3 + [[car(2, 40+80*13/FPS)]])
    play(waits2, [[]]*5 + [[car(3, 40+80*(end+5)/FPS)]], end)
    assert len(waits2.records) == 2


# --- raw ID history: returns, aliases, chains ---------------------------------------

def test_old_native_id_returning_starts_its_own_next_segment_and_contests_the_link():
    waits = book()
    end = play(waits, fragment())
    continued = waits.records['run:acquisition:0:1']
    play(waits, [[car(2, 40+80*(17+i)/FPS), car(1, 30.)] for i in range(3)], end)
    assert set(waits.records) == {'run:acquisition:0:1', 'run:acquisition:0:1:segment-1'}
    returned = waits.records['run:acquisition:0:1:segment-1']
    assert returned['left_censored'] and 'identity_links' not in returned
    assert 'identity_link_contested' in continued['quality_flags']
    assert continued['identity_links'][0]['contested_at'] == pytest.approx(end/FPS)
    assert waits.physical((0, 1))['vehicle_id'] == returned['vehicle_id']
    assert waits.physical((0, 2))['vehicle_id'] == continued['vehicle_id']


def test_handed_off_id_is_never_relinked_back():
    """f73-f75 flip-flop (#296 -> #319 -> #296): a returning ID is not a newborn."""
    waits = book()
    end = play(waits, fragment())
    end = play(waits, [[]]*2, end)
    play(waits, [[car(1, 40+80*(end+i)/FPS)] for i in range(3)], end)
    row = waits.records['run:acquisition:0:1']
    assert len(row['identity_links']) == 1
    assert 'run:acquisition:0:1:segment-1' in waits.records


def test_middle_alias_return_is_contested_and_gets_its_own_row():
    waits = book()
    frames = fragment()                                   # 1 -> 2 at f13
    frames += [[car(2, 40+80*(17+i)/FPS)] for i in range(6)] + [[]]*3
    frames += [[car(3, 40+80*(26+i)/FPS)] for i in range(4)]     # 2 -> 3 at f26
    end = play(waits, frames)
    row = only_row(waits)
    assert [n['local_id'] for n in row['native_ids']] == [1, 2, 3]
    play(waits, [[car(3, 40+80*(30+i)/FPS), car(2, 20., x=10.)] for i in range(2)], end)
    assert 'identity_link_contested' in row['quality_flags']
    assert [l.get('contested_at') is not None for l in row['identity_links']] == [False, True]
    alias = waits.records['run:acquisition:0:2']
    assert alias['left_censored'] and alias['local_id'] == 2


def test_return_after_the_row_finished_does_not_rewrite_it():
    waits = book()
    end = play(waits, fragment()+[[]]*40)                 # the continued row is lost and finished
    row = waits.records['run:acquisition:0:1']
    assert row['completed']
    flags = list(row['quality_flags'])
    play(waits, [[car(1, 20.)]], end)
    assert row['quality_flags'] == flags
    assert 'run:acquisition:0:1:segment-1' in waits.records


def test_long_chains_are_bounded_per_row():
    waits = book()
    frames, y, ident = [], 0., 1
    for link in range(40):
        frames += [[car(ident, y+80*i/FPS, x=50.)] for i in range(4)]
        y += 80*5/FPS; ident += 1
        frames += [[]]
    play(waits, frames)
    for row in waits.records.values():
        assert len(row.get('identity_links', ())) <= vw.MAX_LINKS_PER_ROW
    assert len(waits.handed_off) <= vw.MAX_LINKS_PER_ROW*len(waits.records)


# --- bounded work -------------------------------------------------------------------

def dense(waits, n):
    """n lost vehicles (on a grid) and n newborns in one frame."""
    rows = [(50.+60*(k % 8), 40.+70*(k//8)) for k in range(n)]
    for f in range(6):
        waits.observe([car(1+k, y+2*f, x=x) for k, (x, y) in enumerate(rows)], 'RED', f/FPS, f+1, 1)
    waits.observe([car(10_000+k, y+14, x=x) for k, (x, y) in enumerate(rows)], 'RED', 7/FPS, 8, 1)


@pytest.mark.parametrize('n', [32, 64])
def test_dense_frame_matching_is_bounded(n):
    waits = book()
    dense(waits, n)
    waits = book()
    rows = [(50.+60*(k % 8), 40.+70*(k//8)) for k in range(n)]
    for f in range(6):
        waits.observe([car(1+k, y+2*f, x=x) for k, (x, y) in enumerate(rows)], 'RED', f/FPS, f+1, 1)
    newborns = [car(10_000+k, y+14, x=x) for k, (x, y) in enumerate(rows)]
    started = time.perf_counter()
    waits._continue_fragments(newborns, 7/FPS)
    elapsed = time.perf_counter()-started
    assert elapsed < .05, elapsed


def test_frames_beyond_the_pair_budget_skip_continuation():
    waits = book()
    dense(waits, 70)                                    # 70 x 70 > MAX_PAIR_CHECKS
    assert waits.continuation_skipped_frames == 1
    assert not any(r.get('identity_links') for r in waits.records.values())


# --- archive eviction, acknowledgment order and index rebuilds -----------------------

def engine(**policy):
    policy = {'debounce_frames': 1, 'stop_persistence_sec': 0, 'fragment_continuation': True, **policy}
    return ObservationEngine('run', RunPolicy(**policy), {'xmin': 0, 'ymin': 0, 'xmax': 8, 'ymax': 8},
                             lambda p: p[1]-LINE, lambda *_: None)


PROJECTION = ('vehicle_id', 'local_id', 'native_ids', 'crossed_stop_line', 'crossing_at', 'completed_at',
              'left_censored', 'right_censored', 'tracking_state', 'quality_flags')


def project(rows):
    out = {}
    for r in rows:
        item = {k: r.get(k) for k in PROJECTION}
        for k in ('red_wait_sec', 'unknown_sec', 'phase_unknown_sec'):item[k] = r[k]
        item = {k: round(v, 6) if isinstance(v, float) else v for k, v in item.items()}
        item['links'] = [{k: l[k] for k in ('from', 'to') if k in l} | {'contested': 'contested_at' in l}
                         for l in r.get('identity_links', [])]
        out[r['vehicle_id']] = item
    return out


def run_engine(frames, tmp_path=None, every=None, signal=None):
    e = engine()
    store = None
    if every:
        e.enable_retention(1, tmp_path)
        store = ExperimentStore(tmp_path, e.run_id, {})
    for index, observations in enumerate(frames):
        e.observe_acquisition(signal(index) if signal else RED, index/FPS, index+1, observations)
        if every and index % every == 0:
            state, token = e.prepare_commit(); store.persist(state); e.acknowledge(token)
    e.close()
    if not every:
        return project(e.waits.records.values()), e, None
    state, token = e.prepare_commit(); store.persist(state); e.acknowledge(token)
    return project(store.read()['vehicles']), e, store


def alias_scenario():
    """Sol P1 #3: A->B->C; B returns beside C, crosses; later C exits by the side;
    much later B is seen again past the line."""
    f = approach(1, 10, y0=0.) + [[]]*3
    f += [[car(2, 80*(13+i)/FPS)] for i in range(6)] + [[]]*3            # 1 -> 2 at f13
    f += [[car(3, 80*(22+i)/FPS)] for i in range(3)]                     # 2 -> 3 at f22
    f += [[car(3, 80*25/FPS), car(2, 60.+2*i, x=5.)] for i in range(18)]  # 2 back beside 3, which stops
    f += [[car(3, 80*25/FPS), car(2, 110., x=5., in_region=False)]]       # 2 crosses
    f += [[car(3, 80*25/FPS, x=50.+3*i)] for i in range(5)]
    f += [[car(3, 90., x=400., in_region=False)]]                           # 3 leaves by the side
    f += [[]]*40 + [[car(2, 95., x=5.)] for _ in range(3)] + [[car(2, 105., x=5., in_region=False)]]
    f += [[car(1, 30.)] for _ in range(3)] + [[]]*40 + [[car(3, 30.)] for _ in range(3)]
    return f


@pytest.mark.parametrize('every', [1, 2, 7])
def test_alias_history_is_invariant_to_archive_timing(tmp_path, every):
    frames = alias_scenario()
    live, _, _ = run_engine(frames)
    archived, e, store = run_engine(frames, tmp_path, every)
    assert archived == live
    row = live['run:acquisition:0:1']
    assert [n['local_id'] for n in row['native_ids']] == [1, 2, 3]
    assert row['links'][1]['contested']
    assert live['run:acquisition:0:2']['crossed_stop_line']
    assert sum(r['crossed_stop_line'] for r in live.values()) == 1     # B never crosses twice
    assert 'run:acquisition:0:1:segment-1' in live and 'run:acquisition:0:3' in live
    rebuilt = e.retention.rebuilt(store.records())
    for raw in (1, 2, 3):
        base = f'run:acquisition:0:{raw}'
        assert e.retention.identity(base) == rebuilt.identity(base)
    assert e.retention.identity('run:acquisition:0:2') == (1, True)
    assert e.waits.handed_off == {}


def test_returned_base_id_rows_match_with_and_without_eviction(tmp_path):
    """Sol probe: A->B, A returns. Before eviction 2 rows, after eviction 1 row."""
    frames = fragment() + [[car(2, 40+80*(17+i)/FPS)] for i in range(13)]   # B crosses
    frames += [[]]*50 + [[car(1, 30.)] for _ in range(3)]
    live, _, _ = run_engine(frames)
    archived, _, _ = run_engine(frames, tmp_path, 1)
    assert archived == live and len(live) == 2
    assert live['run:acquisition:0:1:segment-1']['left_censored']


def test_derived_index_is_independent_of_archive_order():
    rows = [
        {'vehicle_id': 'r:acquisition:0:1', 'tracker_epoch': 0, 'local_id': 1, 'crossed_stop_line': False,
         'native_ids': [{'epoch': 0, 'local_id': 1}, {'epoch': 0, 'local_id': 2}, {'epoch': 0, 'local_id': 3}]},
        {'vehicle_id': 'r:acquisition:0:2', 'tracker_epoch': 0, 'local_id': 2, 'crossed_stop_line': True},
        {'vehicle_id': 'r:acquisition:0:1:segment-1', 'tracker_epoch': 0, 'local_id': 1, 'crossed_stop_line': True},
        {'vehicle_id': 'r:acquisition:0:5', 'tracker_epoch': 0, 'local_id': 5, 'crossed_stop_line': True,
         'native_ids': [{'epoch': 0, 'local_id': 5}, {'epoch': 0, 'local_id': 6}]},
    ]
    keys = [f'r:acquisition:0:{n}' for n in (1, 2, 3, 5, 6, 7)]
    results = []
    for order in (rows, rows[::-1], [rows[2], rows[0], rows[3], rows[1]]):
        retention = Retention(4)
        for row in order:retention._derive('vehicles', row)
        results.append([retention.identity(k) for k in keys])
    assert results[0] == results[1] == results[2]
    assert results[0] == [(2, True), (1, True), (0, False), (1, False), (0, True), None]


def test_long_chain_references_are_released_after_final_eviction(tmp_path):
    frames, y, ident = [], 0., 1
    for _ in range(64):
        frames += [[car(ident, y+80*i/FPS)] for i in range(4)] + [[]]
        y += 80*5/FPS; ident += 1
        if y > 55:y = 0.
    frames += [[]]*60
    _, e, _ = run_engine(frames, tmp_path, 1)
    assert e.waits.records == {} and e.waits.handed_off == {}
    assert len(e.waits.active) == 0


# --- other identity modes and the session label --------------------------------------

def test_persistent_waits_never_continue_native_fragments():
    waits = PersistentVehicleWaits('run', RunPolicy(stop_persistence_sec=0, fragment_continuation=True),
                                   lambda p: p[1]-LINE, IdentityPolicy(mode='persistent_v1'))
    assert not waits.fragment_continuation


def test_native_engine_snapshot_exposes_physical_vehicle_and_raw_ids():
    e = engine()
    for index, observations in enumerate(fragment()):
        e.observe_acquisition(RED, index/FPS, index+1, observations)
    vehicles = e.snapshot(evidence=True)['vehicles']
    assert len(vehicles) == 1
    assert [n['local_id'] for n in vehicles[0]['native_ids']] == [1, 2]
