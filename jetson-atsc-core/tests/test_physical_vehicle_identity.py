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


def book(side=None, **policy):
    policy = {'stop_persistence_sec': 0, 'fragment_continuation': True, **policy}
    return VehicleWaits('run', RunPolicy(**policy), side=(lambda p: p[1]-LINE) if side is None else side)


def play(waits, frames, start=0, phase='RED'):
    for index, observations in enumerate(frames, start):
        # Complete evidence: the tracker withheld nothing (an omitted value is unavailable).
        waits.observe(list(observations), phase, index/FPS, index+1, 1, unassigned=())
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
        waits.observe([car(1+k, y+2*f, x=x) for k, (x, y) in enumerate(rows)], 'RED', f/FPS, f+1, 1, unassigned=())
    waits.observe([car(10_000+k, y+14, x=x) for k, (x, y) in enumerate(rows)], 'RED', 7/FPS, 8, 1, unassigned=())


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
    waits._continue_fragments(newborns, 7/FPS, ())
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
        e.observe_acquisition(signal(index) if signal else RED, index/FPS, index+1, observations, unassigned_boxes=())
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
        e.observe_acquisition(RED, index/FPS, index+1, observations, unassigned_boxes=())
    vehicles = e.snapshot(evidence=True)['vehicles']
    assert len(vehicles) == 1
    assert [n['local_id'] for n in vehicles[0]['native_ids']] == [1, 2]


# --- review follow-ups: emitted source boxes, return-on-crossing order, stale sightings ---

def lost_and_newborn(competitor):
    """ID 1 lost at x=50 moving down; ID 2 is born in its gate; ``competitor`` joins the newborn's frame."""
    frames = fragment(gap=3, width=6.)     # narrow boxes (as in the review): no overlap veto
    frames[13] = frames[13]+[competitor]
    return frames


def test_tracked_competitor_vetoes_by_its_raw_source_box_too():
    # The predicted point is (50, 40+80*13/30); the gate is .35+.3*4/30 = .39 heights = 15.6 px.
    # Competitor ID 3 is tracked already (seen since frame 0, far away): its Kalman box is 16.8 px
    # from the predicted point (outside the gate), its raw detection 12.0 px (inside).
    y = 40+80*13/FPS
    def three(x, source_x=None):
        observation = car(3, y, x=x, width=6.)
        if source_x is None:return observation
        half = observation.bbox[2]-x
        return VehicleObservation(3, 'car', 2., (x, y), height=40., bbox=observation.bbox,
                                  source_bbox=(source_x-half, y-40, source_x+half, y))
    def run(competitor):
        waits = book()
        frames = lost_and_newborn(competitor)
        for i in range(13):frames[i] = frames[i]+[car(3, y, x=300., width=6.)]
        play(waits, frames)
        return [(l['from']['local_id'], l['to']['local_id']) for r in waits.records.values() for l in r.get('identity_links', ())]
    assert run(three(33.2, source_x=38.)) == []
    # The same smoothed box without its source attribution does not compete: the review's 1 -> 3 case.
    assert run(three(33.2)) == [(1, 2)]
    # A competitor whose raw box is also outside the gate does not veto.
    assert run(three(33.2, source_x=30.)) == [(1, 2)]


def test_newborns_own_source_box_never_vetoes_it():
    waits = book()
    frames = fragment(gap=3)
    newborn = frames[13][0]
    # Its raw detection sits well inside the gate but is its own: not a competitor.
    frames[13] = [VehicleObservation(2, 'car', 2., newborn.point, height=40., bbox=newborn.bbox,
                                     source_bbox=(newborn.bbox[0]+3, newborn.bbox[1]+2, newborn.bbox[2]+3, newborn.bbox[3]+2))]
    play(waits, frames)
    assert [n['local_id'] for n in only_row(waits)['native_ids']] == [1, 2]


@pytest.mark.parametrize('returned_first', [False, True])
def test_return_on_the_crossing_frame_contests_the_link_in_either_order(returned_first):
    waits = book()
    end = play(waits, fragment())
    end = play(waits, [[car(2, 40+80*i/FPS)] for i in range(end, 23)], end)
    crossing, back = car(2, 40+80*23/FPS), car(1, 30.)
    play(waits, [[back, crossing] if returned_first else [crossing, back]], end)
    row = waits.records['run:acquisition:0:1']
    # The crossing is still observed on ID 2's own consecutive frames ...
    assert row['crossed_stop_line'] and row['crossing_at'] == pytest.approx(23/FPS)
    # ... and the simultaneous return contradicts the live link whatever the observation order.
    assert 'identity_link_contested' in row['quality_flags']
    assert row['identity_links'][0]['contested_at'] == pytest.approx(23/FPS)
    assert 'run:acquisition:0:1:segment-1' in waits.records


def test_stale_outside_sighting_expires_before_matching():
    def run(outside_at):
        waits = book()
        waits.observe([car(2, 0., in_region=False)], 'RED', outside_at, 1, 1, unassigned=())
        for k, t in enumerate((.4, .6, .8)):
            waits.observe([car(1, 40+80*(t-.4))], 'RED', t, k+2, 1, unassigned=())
        waits.observe([car(2, 72+80*.24)], 'RED', 1.04, 5, 1, unassigned=())
        return [n['local_id'] for n in waits.records['run:acquisition:0:1'].get('native_ids', ())]
    # Seen outside 1.04 s earlier (> association_gap_sec 1.0): a newborn on this very frame.
    assert run(0.) == [1, 2]
    # Seen outside within the association timeout: not a newborn.
    assert run(.1) == []


@pytest.mark.parametrize('gap_frames', [1, 2])
def test_three_nonconsecutive_sightings_within_max_gap_are_enough_history(gap_frames):
    # The rule counts box sightings, each within max_gap_sec of the previous, not consecutive frames.
    waits = book()
    frames = []
    for i in range(3):
        frames += [[car(1, 40+80*(i*(gap_frames+1))/FPS)]]+[[]]*gap_frames
    t0 = len(frames)+1
    frames += [[]]+[[car(2, 40+80*(t0+i)/FPS)] for i in range(3)]
    play(waits, frames)
    assert [n['local_id'] for n in only_row(waits)['native_ids']] == [1, 2]


# --- review follow-ups: the newborn's own raw source box ----------------------------------
# fragment(gap=3): ID 1 lost at x=50 moving down; the predicted point is (50, NEWBORN_Y) and the
# gate .39 heights = 15.6 px. Boxes are 6 px wide (no incidental overlap) and 40 px tall.

NEWBORN_Y = 40+80*13/FPS


def narrow(x, y=NEWBORN_Y, height=40.):
    return (x-3., y-height, x+3., y)


def newborn(local_id=2, x=50., source=None, bbox=True, y=NEWBORN_Y):
    return VehicleObservation(local_id, 'car', 2., (x, y), height=40., bbox=narrow(x, y) if bbox else None,
                              source_bbox=source)


def observe_newborn_frame(frames, last, unassigned=()):
    waits = book()
    end = play(waits, frames)
    waits.observe(last, 'RED', end/FPS, end+1, 1, unassigned=unassigned)
    return [(l['from']['local_id'], l['to']['local_id']) for r in waits.records.values() for l in r.get('identity_links', ())]


def lost_one():
    return fragment(gap=3)[:13]


@pytest.mark.parametrize('competitor', ['withheld', 'tracked_source'])
def test_newborn_source_box_overlap_vetoes_when_its_smoothed_box_does_not(competitor):
    # Competitor at x=68: 18 px from the predicted point (outside the gate). The newborn's Kalman
    # box (x=50) misses it; its raw detection (x=66) covers 4/6 of it, footpoints 2 px apart.
    frames = lost_one()
    if competitor == 'tracked_source':
        # An existing track whose own Kalman box is far away while its raw detection is at x=68.
        for i in range(13):frames[i] = frames[i]+[car(3, NEWBORN_Y, x=300., width=6.)]
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), height=40., bbox=narrow(100.),
                                   source_bbox=narrow(68.))
        run = lambda last:observe_newborn_frame(frames, last+[other])
    else:
        run = lambda last, withheld=(narrow(68.),):observe_newborn_frame(frames, last, unassigned=withheld)
    source = narrow(66.)
    assert vw._overlap_of_smaller(source, narrow(68.)) > .5 > vw._overlap_of_smaller(narrow(50.), narrow(68.))
    assert run([newborn(source=source)]) == []
    # Without the raw box the smoothed box alone does not compete: the link is made.
    assert run([newborn()]) == [(1, 2)]
    # No emitted box at all: the raw box still competes.
    assert run([newborn(source=source, bbox=False)]) == []
    if competitor == 'withheld':
        # Positive control: no competitor, the same newborn (raw box included) links.
        assert run([newborn(source=source)], withheld=()) == [(1, 2)]
        assert run([newborn(source=source, bbox=False)], withheld=()) == [(1, 2)]


def test_newborn_source_footpoint_in_another_lost_gate_makes_the_pair_ambiguous():
    # Lost A (ID 1, x=50) and B (ID 5, x=80): the newborn's smoothed point is in A's gate only,
    # its raw detection's footpoint (x=78) in B's gate only.
    with_b = [[car(1, 40+80*i/FPS), car(5, 40+80*i/FPS, x=80.)] for i in range(10)]+[[]]*3
    raw_near_b = newborn(source=narrow(78.))
    assert observe_newborn_frame(with_b, [raw_near_b]) == []
    # Positive control: without B the same newborn continues A.
    assert observe_newborn_frame(lost_one(), [raw_near_b]) == [(1, 2)]


def test_second_newborn_source_footpoint_in_the_gate_makes_the_pair_ambiguous():
    # Newborn 2's smoothed point is in the gate; newborn 3's Kalman point is 40 px off but its
    # raw detection's footpoint is 10 px from the predicted point.
    assert observe_newborn_frame(lost_one(), [newborn(), newborn(3, x=90., source=narrow(60.))]) == []
    assert observe_newborn_frame(lost_one(), [newborn(), newborn(3, x=90., source=narrow(110.))]) == [(1, 2)]


def test_raw_source_footpoint_alone_never_links():
    # The raw detection is 2 px from the predicted point, the emitted (smoothed) point 20 px.
    assert observe_newborn_frame(lost_one(), [newborn(x=70., source=narrow(52.))]) == []
    assert observe_newborn_frame(lost_one(), [newborn(x=52., source=narrow(52.))]) == [(1, 2)]


@pytest.mark.parametrize('source', [
    (47., NEWBORN_Y-40, float('nan'), NEWBORN_Y),      # non-finite
    (53., NEWBORN_Y-40, 47., NEWBORN_Y),               # x2 <= x1
    (47., NEWBORN_Y, 53., NEWBORN_Y),                  # zero height
    (47., NEWBORN_Y-40),                               # too short
    'unreadable',
])
def test_malformed_newborn_source_box_fails_closed(source):
    assert observe_newborn_frame(lost_one(), [newborn(source=source)]) == []
    assert observe_newborn_frame(lost_one(), [newborn(source=narrow(50.))]) == [(1, 2)]


@pytest.mark.parametrize('case', ['truncated_competitor_source', 'truncated_competitor_bbox',
                                  'truncated_competitor_bbox_default_height', 'overflow_competitor_bbox',
                                  'mapping_competitor_bbox', 'huge_endpoints_competitor_bbox',
                                  'truncated_unassigned', 'nonfinite_unassigned', 'mapping_unassigned'])
def test_malformed_competitor_or_unassigned_box_makes_frame_evidence_unavailable(case):
    """Sol S2/S7: unreadable, mapping, overflow, or huge-endpoint competitor or unassigned
    boxes must fail closed for the frame, even with default height=None."""
    frames = lost_one()
    other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), height=40., bbox=narrow(100.),
                               source_bbox=narrow(68.))
    unassigned = ()
    if case == 'truncated_competitor_source':
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), height=40., bbox=narrow(100.),
                                   source_bbox=(65., NEWBORN_Y-40.))
    elif case == 'truncated_competitor_bbox':
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), height=40., bbox=(97., NEWBORN_Y-40.),
                                   source_bbox=narrow(68.))
    elif case == 'truncated_competitor_bbox_default_height':
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), bbox=(97., NEWBORN_Y-40.),
                                   source_bbox=narrow(68.))
    elif case == 'overflow_competitor_bbox':
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), bbox=(97., NEWBORN_Y-40., 10**400, NEWBORN_Y),
                                   source_bbox=narrow(68.))
    elif case == 'mapping_competitor_bbox':
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), bbox={}, source_bbox=narrow(68.))
    elif case == 'huge_endpoints_competitor_bbox':
        other = VehicleObservation(3, 'car', 2., (100., NEWBORN_Y), bbox=(97., -1e308, 103., 1e308),
                                   source_bbox=narrow(68.))
    elif case == 'truncated_unassigned':
        other = None
        unassigned = [(65., NEWBORN_Y-40.)]
    elif case == 'mapping_unassigned':
        other = None
        unassigned = {}
    else:
        other = None
        unassigned = [(65., NEWBORN_Y-40., float('nan'), NEWBORN_Y)]

    waits = book()
    end = play(waits, frames)
    last = [newborn(source=narrow(66.))] + ([other] if other is not None else [])
    waits.observe(last, 'RED', end/FPS, end+1, 1, unassigned=unassigned)
    row_links = [(l['from']['local_id'], l['to']['local_id'])
                 for r in waits.records.values() for l in r.get('identity_links', ())]
    assert row_links == []
    assert waits.continuation_unavailable_frames == 1


def test_huge_finite_endpoints_overflowing_height_fail_closed():
    """Sol S7: finite box endpoints whose difference or area overflows to inf must not produce
    an infinite height or bypass distance bounds."""
    from src.experiment.vehicle_waits import _height, _raw_box

    huge_box = (1045., -1e308, 1055., 1e308)
    wide_area_box = (0., 0., 1e200, 1e200)
    far_newborn = VehicleObservation(2, 'car', 2., (1050., NEWBORN_Y), bbox=huge_box)
    assert _raw_box(huge_box) is None
    assert _raw_box(wide_area_box) is None
    assert _height(far_newborn) is None

    waits = book()
    end = play(waits, lost_one())
    waits.observe([far_newborn], 'RED', end/FPS, end+1, 1, unassigned=())
    row_links = [(l['from']['local_id'], l['to']['local_id'])
                 for r in waits.records.values() for l in r.get('identity_links', ())]
    assert row_links == []

    waits_explicit = book()
    end2 = play(waits_explicit, lost_one())
    far_with_height = VehicleObservation(2, 'car', 2., (1050., NEWBORN_Y), height=40., bbox=huge_box)
    waits_explicit.observe([far_with_height], 'RED', end2/FPS, end2+1, 1, unassigned=())
    assert [(l['from']['local_id'], l['to']['local_id'])
            for r in waits_explicit.records.values() for l in r.get('identity_links', ())] == []
    assert waits_explicit.continuation_unavailable_frames == 1


# --- Idea A1: occlusion-tail velocity de-biasing and stationary reappearance (> 0.10 s) ---

def _occluded_tail(local_id=1, x=150., y1=100., h_peak=100., h_end=55., frames=8):
    """A stationary car whose bottom edge y2 is clipped upward while y1 stays pinned."""
    out = []
    for i in range(frames):
        frac = i/(frames-1)
        h = h_peak-(h_peak-h_end)*frac
        out.append([car(local_id, y1+h, x=x, height=h, speed=0.)])
    return out


def _links(waits):
    return [(l['from']['local_id'], l['to']['local_id'])
            for r in waits.records.values() for l in r.get('identity_links', ())]


@pytest.mark.parametrize('gap_frames', [4, 6, 10])
def test_adversarial_queue_pull_up_attack_never_links_past_flicker(gap_frames):
    """Stopped car A disappears without an occlusion tail and car B behind it pulls
    into A's exact vacated spot after > 0.10 s: must never link A -> B."""
    waits = book(side=lambda p: -1.)
    # Car A stationary at y1=100, y2=200 (h=100), then vanishes; newborn B fills exact spot.
    frames = [[car(1, 200., x=150., height=100., speed=0.)] for _ in range(8)]
    frames += [[]]*gap_frames
    frames += [[car(2, 200., x=150., height=100., speed=0.)]]
    play(waits, frames)
    assert _links(waits) == []
    assert len(waits.records) == 2


def test_adversarial_queue_pull_up_with_mild_jitter_or_departure_foreshortening_never_links():
    """Neither minor detector height jitter (<25%) nor a departing car foreshortening
    upward may trigger occlusion-tail continuation onto a car pulling into the vacated spot."""
    # Case 1: 12% height jitter on a stopped car before disappearing, gap 0.20 s.
    waits_jitter = book(side=lambda p: -1.)
    jitter_frames = [[car(1, 200.-1.5*i, x=150., height=100.-1.5*i, speed=0.)] for i in range(8)]
    jitter_frames += [[]]*5 + [[car(2, 200., x=150., height=100., speed=0.)]]
    play(waits_jitter, jitter_frames)
    assert _links(waits_jitter) == []

    # Case 2: Car A drives away upward (both y1 and y2 move up, h foreshortens 100 -> 70),
    # and newborn B pulls into A's vacated spot 0.20 s later.
    waits_depart = book(side=lambda p: -1.)
    depart_frames = [[car(1, 200.-10.*i, x=150., height=100.-4.*i, speed=3.)] for i in range(8)]
    depart_frames += [[]]*5 + [[car(2, 200., x=150., height=100., speed=1.)]]
    play(waits_depart, depart_frames)
    assert _links(waits_depart) == []


def test_adversarial_shrinking_occlusion_tail_never_fabricates_upward_velocity():
    """Bottom-edge shrinkage on a stationary car must not extrapolate upward onto
    an upstream vehicle or widen distance gates."""
    waits = book(side=lambda p: -1.)
    # Stationary car at y1=100 shrinks y2: 200 -> 150 over 7/30 s (apparent vy2 ~ -214 px/s).
    # 0.30 s later, phantom extrapolation lands at y2 ~ 86: an upstream car C appears there.
    frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=50., frames=8)
    frames += [[]]*8
    frames += [[car(2, 86., x=150., height=50., speed=0.)]]
    play(waits, frames)
    assert _links(waits) == []

    # Horizontal motion with sudden bottom-edge collapse must not extrapolate upward either.
    waits_horiz = book(side=lambda p: -1.)
    horiz = [[car(1, 200.-7.*i, x=100.+4.*i, height=100.-7.*i, speed=2.)] for i in range(8)]
    horiz += [[]]*5 + [[car(2, 151.-42., x=128.+24., height=51., speed=2.)]]
    play(waits_horiz, horiz)
    assert _links(waits_horiz) == []


@pytest.mark.parametrize('scenario', [
    'active_occluder_duplicate_on_body',
    'occluder_reborn_in_front',
    'larger_foreground_occluder_newborn',
    'active_competitor_at_peak_footpoint',
    'withheld_competitor_at_top_gate',
])
def test_adversarial_same_class_occluder_never_links_or_swaps(scenario):
    """A same-class vehicle crossing in front of a bottom-occluded stationary car
    must never be linked to the occluded car or cause an ID swap."""
    waits = book(side=lambda p: -1.)
    if scenario == 'active_occluder_duplicate_on_body':
        # Active same-class car #9 crosses right across car #1's box; a duplicate newborn
        # #2 appears sharing #9's box at car #1's top-center.
        frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8)
        for i, f in enumerate(frames):
            f.append(car(9, 210., x=90.+8.*i, height=95., speed=3.))
        frames += [[car(9, 205., x=150., height=95., speed=2.)] for _ in range(5)]
        frames += [[car(9, 205., x=150., height=95., speed=2.),
                    car(2, 165., x=150., height=65., speed=0.)]]
        play(waits, frames)
    elif scenario == 'occluder_reborn_in_front':
        # Same-class occluder #9 crosses directly over #1, both vanish, then #9 returns as
        # newborn #2 right inside BOTH #1's occlusion top-gate and #9's moving gate: ambiguous!
        frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8)
        for i, f in enumerate(frames):
            f.append(car(9, 208., x=106.+4.*i, height=105., speed=2.))
        frames += [[]]*5
        frames += [[car(2, 208., x=158., height=105., speed=2.)]]
        play(waits, frames)
    elif scenario == 'larger_foreground_occluder_newborn':
        # Unseen larger/closer same-class car appears with matching top-center y1=100
        # but foreground bottom y2=230 (> y2_peak + 0.15*h_peak and h=130 > 1.2*h_peak).
        frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8)
        frames += [[]]*5 + [[car(2, 230., x=150., height=130., speed=2.)]]
        play(waits, frames)
    elif scenario == 'active_competitor_at_peak_footpoint':
        # Another car sits right at car #1's pre-occlusion footpoint (150, 200) at reappearance.
        frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8)
        for f in frames:
            f.append(car(9, 210., x=300., height=80., speed=0.))
        frames += [[car(9, 210., x=200., height=80., speed=1.)]]*5
        frames += [[car(9, 205., x=152., height=80., speed=0.),
                    car(2, 160., x=150., height=60., speed=0.)]]
        play(waits, frames)
    else:
        # Withheld detection sitting at car #1's pinned top-center vetoes the occlusion-tail link.
        frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8) + [[]]*5
        end = play(waits, frames)
        waits.observe([car(2, 160., x=150., height=60., speed=0.)], 'RED', end/FPS, end+1, 1,
                      unassigned=[(120., 102., 180., 162.)])
    assert _links(waits) == [], scenario


@pytest.mark.parametrize('gap_frames,new_h,expect_link', [
    (5, 60., True),    # dt=0.200 s, still partially clipped (746->966 pattern)
    (9, 95., True),    # dt=0.333 s, recovered back near h_peak (8.3.0 pattern)
    (11, 60., False),  # dt=0.400 s > OCCLUSION_MAX_GAP_SEC (0.35 s)
])
def test_stationary_bottom_occlusion_tail_reappearance_links_within_gap_bound(gap_frames, new_h, expect_link):
    waits = book(side=lambda p: -1.)
    frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8)
    frames += [[]]*gap_frames
    frames += [[car(2, 100.+new_h, x=151., height=new_h, speed=0.)]]
    play(waits, frames)
    if not expect_link:
        assert _links(waits) == []
        return
    assert _links(waits) == [(1, 2)]
    link, = only_row(waits)['identity_links']
    assert link['moving_fit'] is False
    assert link['rule'] == 'mutually_unique_uncontested_occlusion_tail_gate'
    assert link['predicted_point'] == pytest.approx([150., 100.])
    assert link['observed_point'] == pytest.approx([151., 100.])
    assert link['error_heights'] <= link['gate_heights']


def test_occlusion_tail_respects_policy_max_gap_and_source_box_ambiguity():
    # Policy max_gap_sec=0.20 caps OCCLUSION_MAX_GAP_SEC=0.35.
    waits_cap = book(max_gap_sec=.20, side=lambda p: -1.)
    frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8) + [[]]*7
    frames += [[car(2, 160., x=150., height=60., speed=0.)]]   # dt=0.267 s > 0.20 s
    play(waits_cap, frames)
    assert _links(waits_cap) == []

    # Raw source_bbox widens ambiguity (second newborn's raw top-center inside gate vetoes),
    # and raw source_bbox alone never creates a link.
    base = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8) + [[]]*5
    nb_ok = VehicleObservation(2, 'car', 0., (150., 160.), height=60., bbox=(120., 100., 180., 160.),
                               source_bbox=(120., 100., 180., 160.))
    nb_raw_rival = VehicleObservation(3, 'car', 0., (260., 160.), height=60., bbox=(230., 100., 290., 160.),
                                      source_bbox=(125., 102., 185., 162.))
    waits_ambig = book(side=lambda p: -1.)
    end = play(waits_ambig, base)
    waits_ambig.observe([nb_ok, nb_raw_rival], 'RED', end/FPS, end+1, 1, unassigned=())
    assert _links(waits_ambig) == []

    nb_raw_only = VehicleObservation(2, 'car', 0., (220., 160.), height=60., bbox=(190., 100., 250., 160.),
                                     source_bbox=(120., 100., 180., 160.))
    waits_raw = book(side=lambda p: -1.)
    end = play(waits_raw, base)
    waits_raw.observe([nb_raw_only], 'RED', end/FPS, end+1, 1, unassigned=())
    assert _links(waits_raw) == []

    # Non-finite point on a competitor fails closed and increments continuation_unavailable_frames.
    waits_nan = book(side=lambda p: -1.)
    end = play(waits_nan, base)
    bad = VehicleObservation(9, 'car', 0., (float('nan'), 160.), height=60.)
    waits_nan.observe([nb_ok, bad], 'RED', end/FPS, end+1, 1, unassigned=())
    assert _links(waits_nan) == [] and waits_nan.continuation_unavailable_frames == 1


def test_sol_s1_mild_bottom_clip_with_horizontal_motion_preserves_vx_without_upward_lift():
    """Sol S1: horizontal motion with pinned top edge and mild 10 px bottom clip must not
    extrapolate upward onto an upstream vehicle, while preserving horizontal tracking."""
    # Moving horizontally at +120 px/s (+4 px/frame) with pinned y1=100; bottom clips 200 -> 190.
    frames = [[car(1, 200.-10.*i/7, x=100.+4.*i, height=100.-10.*i/7, speed=3.)] for i in range(8)]
    # Gap of 14 frames (dt=0.50 s, gate=0.50*90=45 px): unlifted prediction is (188, 190);
    # phantom upward lift would land at y=168.6 and falsely capture an upstream car at y=135.
    waits_upstream = book(side=lambda p: -1.)
    play(waits_upstream, frames + [[]]*14 + [[car(2, 135., x=188., height=90., speed=3.)]])
    assert _links(waits_upstream) == []

    # Positive control: newborn continuing horizontally at (188, 190) links with zero upward lift.
    waits_horiz = book(side=lambda p: -1.)
    play(waits_horiz, frames + [[]]*14 + [[car(2, 190., x=188., height=90., speed=3.)]])
    assert _links(waits_horiz) == [(1, 2)]
    assert only_row(waits_horiz)['identity_links'][0]['predicted_point'] == pytest.approx([188., 190.])

    # Sol S1 Round 2: 20 px height drop with 11 px upward top-edge drift must not qualify as moving.
    waits_drift = book(side=lambda p: -1.)
    drift = [[car(1, 200.-31.*i/7, x=150., height=100.-20.*i/7, speed=0.)] for i in range(8)]
    play(waits_drift, drift + [[]]*5 + [[car(2, 145., x=150., height=80., speed=0.)]])
    assert _links(waits_drift) == []

    # Sol S8: 8 px upward top-edge drift + 7 px height loss must not qualify as moving either.
    waits_s8 = book(side=lambda p: -1.)
    drift_s8 = [[car(1, 200.-15.*i/7, x=150., height=100.-7.*i/7, speed=0.)] for i in range(8)]
    play(waits_s8, drift_s8 + [[]]*5 + [[car(2, 185., x=150., height=93., speed=0.)]])
    assert _links(waits_s8) == []

    # Sol S10: alternating tall/clipped heights with minor horizontal drift must scale motion
    # qualification against max history height (100 px), not the clipped height (35 px).
    waits_s10 = book(side=lambda p: -1.)
    heights_s10 = [100., 35., 100., 35., 35., 35., 35., 35.]
    drift_s10 = [[car(1, 100.+h, x=150.+5.*i/7, height=h, speed=0.)] for i, h in enumerate(heights_s10)]
    play(waits_s10, drift_s10 + [[]]*5 + [[car(2, 135., x=155., height=35., speed=0.)]])
    assert _links(waits_s10) == []


def test_sol_s9_tiny_positive_height_at_large_y_does_not_raise_zero_division():
    """Sol S9: carrying validated h_last directly avoids ZeroDivisionError when y - (y - h_last)
    would round to 0.0 in IEEE 754 float arithmetic."""
    waits = book(side=lambda p: -1.)
    # Stationary top pinned at y1=100; baseline height=100 (2+ frames), ending at tiny h=1e-15.
    heights = [100., 100., 100., 100., 50., 25., 10., 1e-15]
    frames = [[VehicleObservation(1, 'car', 0., (150., 100.+h), height=h)] for h in heights]
    frames += [[]]*5
    end = play(waits, frames)
    nb = VehicleObservation(2, 'car', 0., (150., 150.), height=50., bbox=(125., 100., 175., 150.))
    competitor = VehicleObservation(9, 'car', 0., (300., 300.), height=50., bbox=(275., 250., 325., 300.))
    waits.observe([nb, competitor], 'RED', end/FPS, end+1, 1, unassigned=())
    assert _links(waits) == [(1, 2)]


@pytest.mark.parametrize('neighbour_sightings', [8, 2])
def test_sol_s2_lost_stationary_neighbour_still_vetoes_extended_occlusion_link_after_0_10s(neighbour_sightings):
    """Sol S2: a plain stationary car (even with only 2 sightings) lost 0.20 s earlier cannot
    link past 0.10 s, but must still veto a nearby occluded car from stealing its spot."""
    waits = book(side=lambda p: -1.)
    frames = _occluded_tail(1, x=150., y1=100., h_peak=100., h_end=55., frames=8)
    for i, f in enumerate(frames):
        if i >= len(frames)-neighbour_sightings:
            # Plain stationary neighbour #5 parked 20 px right (within #1's 25 px top gate).
            f.append(car(5, 160., x=170., height=60., speed=0.))
    frames += [[]]*5
    # Newborn #2 appears at #5's exact vacated box (170, 160, h=60 -> top 170, 100).
    frames += [[car(2, 160., x=170., height=60., speed=0.)]]
    play(waits, frames)
    assert _links(waits) == []


@pytest.mark.parametrize('heights', [
    [60., 60., 60., 100., 60., 60., 60., 60.],
    [60., 100., 60., 100., 60., 60., 60., 60.],
])
def test_sol_s3_s6_isolated_tall_outlier_frames_never_create_an_occlusion_tail(heights):
    """Sol S3/S6: one or two isolated tall outlier frames in a plain stationary track
    must not qualify as an occlusion tail or allow a replacement link after 0.20 s."""
    waits = book(side=lambda p: -1.)
    frames = [[car(1, 100.+h, x=150., height=h, speed=0.)] for h in heights]
    frames += [[]]*5 + [[car(2, 160., x=150., height=60., speed=0.)]]
    play(waits, frames)
    assert _links(waits) == []


@pytest.mark.parametrize('with_distant_occlusion', [False, True])
def test_sol_s5_s7_stationary_flicker_pairs_are_not_overcharged_in_pair_budget(with_distant_occlusion):
    """Sol S5/S7: 46 stationary flicker pairs stay inside MAX_PAIR_CHECKS both with zero
    occlusion candidates and with one distant unmatched occlusion candidate."""
    waits = book(side=lambda p: -1.)
    grid = [(50.+80.*(k % 8), 80.+90.*(k // 8)) for k in range(46)]
    for f in range(8):
        obs = []
        if with_distant_occlusion:
            h = 100.-45.*f/7
            obs.append(car(999, 900.+h, x=900., height=h, speed=0.))
        if f >= 4:
            obs += [car(1+k, y, x=x, height=40., speed=0.) for k, (x, y) in enumerate(grid)]
        waits.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    waits.observe([car(1000+k, y, x=x, height=40., speed=0.) for k, (x, y) in enumerate(grid)],
                  'RED', 10/FPS, 11, 1, unassigned=())
    assert waits.continuation_skipped_frames == 0
    assert len(_links(waits)) == 46


# --- bottom-edge occlusion stop-line crossing resilience ----------------------------

def test_bottom_clipped_stationary_car_does_not_falsely_cross_and_links_occlusion_tail():
    """#407->#733 / #300->#755 pattern: bottom-edge occlusion clipping a stationary car
    upward past the stop line must not prematurely complete it as SERVED; the open row
    then continues via A1's occlusion-tail gate and crosses once it genuinely departs."""
    stop_y = 270.
    waits = book(side=lambda p: stop_y-p[1])
    # Stationary car at y1=237, baseline h=55 (y2=292, behind stop_y=270), clipped upward to y2=264.
    frames = [[car(407, 292., x=185., height=55., speed=0.)] for _ in range(4)]
    for h in (48., 41., 36., 31., 28., 26.):  # y2 crosses 270 when h=31 (y2=268 < 270)
        frames.append([car(407, 237.+h, x=185.5, height=h, speed=0.)])
    end = play(waits, frames, phase='RED')
    row = only_row(waits)
    assert not row['completed'] and not row['crossed_stop_line']
    assert row['tracking_state'] != 'SERVED'

    # Car vanishes behind occluder for 6 frames (dt=0.233 s), reappears as #733 still clipped,
    # then recovers to full height behind the stop line and genuinely departs upward.
    recovery = [[]]*6
    recovery += [[car(733, 266., x=183., height=29., speed=0.)]]
    recovery += [[car(733, 295., x=183., height=58., speed=0.)] for _ in range(5)]
    for i in range(1, 12):
        recovery.append([car(733, 295.-4.*i, x=183., height=58., speed=2.)])  # genuine upward move
    play(waits, recovery, start=end, phase='GREEN')
    row = only_row(waits)
    assert _links(waits) == [(407, 733)]
    assert row['identity_links'][0]['rule'] == 'mutually_unique_uncontested_occlusion_tail_gate'
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'
    assert row['crossing_phase'] == 'GREEN'
    assert row['crossing_at'] > (end+11)/FPS


def test_bottom_clipped_stationary_car_recovers_behind_line_and_keeps_accruing_wait():
    """#232 (f134) pattern: temporary bottom-edge clip past the stop line cancels cleanly
    when the occlusion clears and the footpoint returns behind the line, preserving RED wait."""
    stop_y = 270.
    waits = book(side=lambda p: stop_y-p[1])
    frames = [[car(232, 278., x=162., height=63., speed=0.)] for _ in range(5)]
    for h in (60., 56., 52., 48., 51., 56.):  # y1=215 -> y2 dips to 263 (< 270) then recovers to 271 (> 270)
        frames.append([car(232, 215.+h, x=162., height=h, speed=0.)])
    # Remains stopped behind the line on RED for 1.0 s (30 frames), then departs on GREEN.
    frames += [[car(232, 278., x=162., height=63., speed=0.)] for _ in range(30)]
    end_red = play(waits, frames, phase='RED')
    row = only_row(waits)
    assert not row['completed'] and not row['crossed_stop_line']
    assert row['red_wait_sec'] >= 1.0

    depart = [[car(232, 278.-4.*i, x=162., height=63., speed=2.)] for i in range(1, 6)]
    play(waits, depart, start=end_red, phase='GREEN')
    assert row['crossed_stop_line'] and row['crossing_phase'] == 'GREEN'
    assert row['crossing_at'] >= end_red/FPS


def test_genuine_departures_still_cross_including_clipped_acceleration_and_downward_orientation():
    """Genuine departures must never be dropped: slow unclipped upward moves, downward
    crossings, isolated tall outliers, and clipped cars whose unclipped rear later clears."""
    # 1. Slow genuine upward departure (both y1 and y2 advance across y=270).
    waits_slow = book(side=lambda p: 270.-p[1])
    slow = [[car(1, 286.-3.*i, x=150., height=55., speed=1.5)] for i in range(10)]
    play(waits_slow, slow, phase='GREEN')
    assert only_row(waits_slow)['crossed_stop_line']

    # 2. Downward crossing (side = y - 200) with mild shrinkage is never suppressed.
    waits_down = book(side=lambda p: p[1]-200.)
    down = [[car(1, 190.+2.*i, x=150., height=60.-2.*i, speed=1.)] for i in range(8)]
    play(waits_down, down, phase='GREEN')
    assert only_row(waits_down)['crossed_stop_line']

    # 3. Initially bottom-clipped stationary car that then drives forward while still
    #    partially clipped confirms once its unclipped rear (y1 + h_base) clears the line.
    waits_confirm = book(side=lambda p: 270.-p[1])
    frames = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    frames += [[car(1, 220.+45., x=150., height=45., speed=0.)]]  # clipped y2=265 (<270), y1+60=280 (>270)
    end_clip = play(waits_confirm, frames, phase='RED')
    row = only_row(waits_confirm)
    assert not row['crossed_stop_line']
    # Now car genuinely drives upward while height stays clipped at 45: y1 moves 220 -> 205 (y1+60=265 < 270).
    drive = [[car(1, (220.-5.*i)+45., x=150., height=45., speed=2.)] for i in range(1, 5)]
    play(waits_confirm, drive, start=end_clip, phase='GREEN')
    assert row['crossed_stop_line'] and row['crossing_phase'] == 'GREEN'
    assert row['crossing_evidence']['from_side'] < 0 <= row['crossing_evidence']['to_side']

    # 4. Single isolated tall outlier frame cannot fabricate h_base to suppress a real crossing.
    waits_outlier = book(side=lambda p: 270.-p[1])
    heights = [50., 50., 80., 50., 50., 50.]
    ys = [275., 274., 274., 273., 272., 269.]
    outlier = [[car(1, y, x=150., height=h, speed=0.5)] for y, h in zip(ys, heights)]
    play(waits_outlier, outlier, phase='GREEN')
    assert only_row(waits_outlier)['crossed_stop_line']

    # 5. Pending crossing clears on an observation gap (never synthesized across a gap).
    waits_gap = book(side=lambda p: 270.-p[1])
    end_gap = play(waits_gap, frames, phase='RED')
    play(waits_gap, [[]] + [[car(1, 250., x=150., height=45., speed=2.)]], start=end_gap, phase='GREEN')
    assert not only_row(waits_gap)['crossed_stop_line']


# --- opt-in ROI boundary exit hysteresis (Idea C) -----------------------------------

def test_exit_persistence_validation_default_off_and_persistent_isolation():
    assert RunPolicy().exit_persistence_sec == 0.0
    for bad in (True, False, -0.1, 5.1, float('nan'), float('inf'), '0.3'):
        with pytest.raises(ValueError):
            RunPolicy(exit_persistence_sec=bad)
    assert RunPolicy(exit_persistence_sec=0.3).exit_persistence_sec == pytest.approx(0.3)

    # PersistentVehicleWaits zeroes RunPolicy.exit_persistence_sec (uses IdentityPolicy only).
    from src.experiment.identity_contracts import IdentityPolicy
    from src.experiment.persistent_vehicle_waits import PersistentVehicleWaits
    pwaits = PersistentVehicleWaits('run', RunPolicy(exit_persistence_sec=0.5), lambda p: -1., IdentityPolicy(mode='persistent_v1'))
    assert pwaits.policy.exit_persistence_sec == 0.0


def test_roi_exit_persistence_bridges_border_dips_without_splitting_or_accruing_wait():
    """#350/#1429/#1468/#1924 pattern: brief border dips keep one vehicle row without
    accruing outside wait; default-off (0 s) still splits immediately."""
    def run_dip(exit_sec):
        waits = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, exit_persistence_sec=exit_sec), side=lambda p: -1.)
        for i in range(5):
            waits.observe([car(1, 120., speed=0.)], 'RED', i/FPS, i+1, 1)
        # Dip outside ROI for 3 frames (0.10 s) while stopped on RED.
        mid_states = []
        for i in range(5, 8):
            waits.observe([car(1, 120., speed=0., in_region=False)], 'RED', i/FPS, i+1, 1)
            mid_states.append((waits.records['run:acquisition:0:1']['tracking_state'],
                               waits.records['run:acquisition:0:1']['completed'],
                               waits.active[(0, 1)].get('exit_candidate')))
        # Return inside ROI for 5 more frames.
        for i in range(8, 13):
            waits.observe([car(1, 120., speed=0.)], 'RED', i/FPS, i+1, 1)
        return waits, mid_states

    off_waits, _ = run_dip(0.0)
    assert set(off_waits.records) == {'run:acquisition:0:1', 'run:acquisition:0:1:segment-1'}
    assert off_waits.records['run:acquisition:0:1']['tracking_state'] == 'EXITED'

    on_waits, mid = run_dip(0.3)
    assert list(on_waits.records) == ['run:acquisition:0:1']
    assert all(state == 'EXIT_PENDING' and not done and cand == pytest.approx(5/FPS) for state, done, cand in mid)
    row = only_row(on_waits)
    assert not row['completed'] and row['tracking_state'] == 'STOPPED'
    assert 'observation_gap' in row['quality_flags']
    # Outside dip [4/FPS, 8/FPS] = 4/FPS s is unknown_sec, never fabricated red wait.
    assert row['unknown_sec'] == pytest.approx(4/FPS)
    assert row['red_wait_sec'] == pytest.approx(8/FPS)  # 4 intervals before + 4 intervals after


def test_roi_exit_persistence_confirms_real_exits_preserves_exit_crossings_and_handles_gaps():
    # 1. Real consecutive exit >= 0.30 s completes at exact first-outside timestamp; later return starts segment-1.
    waits = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, exit_persistence_sec=0.3), side=lambda p: -1.)
    for i in range(3):
        waits.observe([car(1, 120., speed=0.)], 'RED', i/FPS, i+1, 1)
    for i in range(3, 12):  # 3..11 -> dt = 8/30 = 0.267 s < 0.30 s
        waits.observe([car(1, 120., speed=0., in_region=False)], 'RED', i/FPS, i+1, 1)
        assert not waits.records['run:acquisition:0:1']['completed']
    waits.observe([car(1, 120., speed=0., in_region=False)], 'RED', 12/FPS, 13, 1)  # dt = 9/30 = 0.300 s
    row = waits.records['run:acquisition:0:1']
    assert row['completed'] and row['tracking_state'] == 'EXITED' and not row['right_censored']
    assert row['completed_at'] == pytest.approx(3/FPS)
    assert row['last_seen'] == pytest.approx(2/FPS)
    # Re-entry after confirmed exit starts a new segment.
    waits.observe([car(1, 120., speed=0., in_region=True)], 'RED', 20/FPS, 21, 1)
    assert set(waits.records) == {'run:acquisition:0:1', 'run:acquisition:0:1:segment-1'}

    # 2. Genuine stop-line crossing on the exit frame still completes immediately as SERVED.
    waits_cross = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, exit_persistence_sec=0.3), side=lambda p: p[1]-150.)
    waits_cross.observe([car(1, 145., speed=2.)], 'GREEN', 0., 1, 1)
    waits_cross.observe([car(1, 155., speed=2., in_region=False)], 'GREEN', 1/FPS, 2, 1)
    assert only_row(waits_cross)['crossed_stop_line'] and only_row(waits_cross)['tracking_state'] == 'SERVED'

    # 3. Missing-frame gap while outside restarts exit timer (requires consecutive outside evidence),
    #    and vanishing before confirmation closes honestly as tracking_lost after association_gap_sec.
    waits_gap = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, exit_persistence_sec=0.3, association_gap_sec=0.5),
                             side=lambda p: -1.)
    waits_gap.observe([car(1, 120., speed=0.)], 'RED', 0., 1, 1)
    waits_gap.observe([car(1, 120., speed=0., in_region=False)], 'RED', 1/FPS, 2, 1)
    waits_gap.observe([], 'RED', 2/FPS, 3, 1)  # dropout while outside
    waits_gap.observe([car(1, 120., speed=0., in_region=False)], 'RED', 10/FPS, 11, 1)  # gap > 0: restarts timer!
    assert not only_row(waits_gap)['completed']
    waits_gap.observe([], 'RED', 1.0, 31, 1)  # vanishes > association_gap_sec without confirming exit
    row_gap = only_row(waits_gap)
    assert row_gap['completed'] and 'tracking_lost' in row_gap['quality_flags']
    assert row_gap['tracking_state'] == 'LOST'

    # 4. An EXIT_PENDING track that vanishes does not link to a newborn inside the ROI.
    waits_cont = book(exit_persistence_sec=0.3, side=lambda p: -1.)
    end = play(waits_cont, [[car(1, 120., x=100., height=40., speed=0.)] for _ in range(5)])
    waits_cont.observe([car(1, 120., x=100., height=40., speed=0., in_region=False)], 'RED', end/FPS, end+1, 1, unassigned=())
    waits_cont.observe([car(2, 120., x=100., height=40., speed=0., in_region=True)], 'RED', (end+1)/FPS, end+2, 1, unassigned=())
    assert _links(waits_cont) == []
    assert len(waits_cont.records) == 2


def test_sustained_bottom_clip_requires_geometric_recovery_before_clearing_anchor():
    """Sol S2: sustained clipping that ages full-height samples out of history must not
    lose its anchor on a partial (45->51) or single-frame recovery behind the line."""
    stop_y = 270.
    waits = book(side=lambda p: stop_y-p[1])
    # 5 frames full height (y1=220, h=60 -> y2=280), then 10 clipped frames (h=45 -> y2=265).
    frames = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    frames += [[car(1, 265., x=150., height=45., speed=0.)] for _ in range(10)]
    # Partial recovery behind line (h=51 -> y2=271, side=-1) then back to clipped h=45.
    frames += [[car(1, 271., x=150., height=51., speed=0.)],
               [car(1, 265., x=150., height=45., speed=0.)]]
    # Repeated partial recovery (h=54, 90% < 95%) then back to clipped h=45.
    frames += [[car(1, 274., x=150., height=54., speed=0.)] for _ in range(3)]
    frames += [[car(1, 265., x=150., height=45., speed=0.)]]
    # Single-frame full recovery bounce (h=60 for 1 frame) then back to clipped h=45.
    frames += [[car(1, 280., x=150., height=60., speed=0.)],
               [car(1, 265., x=150., height=45., speed=0.)]]
    end = play(waits, frames, phase='RED')
    row = only_row(waits)
    assert not row['completed'] and not row['crossed_stop_line']
    assert waits.active[(0, 1)].get('pending_crossing') is not None

    # Two consecutive full-height behind-line frames clear the anchor cleanly, and
    # populate history so a subsequent clip is still caught while genuine motion crosses.
    end = play(waits, [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(2)], start=end, phase='RED')
    assert waits.active[(0, 1)].get('pending_crossing') is None
    end = play(waits, [[car(1, 265., x=150., height=45., speed=0.)]], start=end, phase='RED')
    assert not row['crossed_stop_line']
    # Genuine forward motion while clipped confirms once y1 + 60 clears stop_y=270.
    depart = [[car(1, (220.-5.*i)+45., x=150., height=45., speed=2.)] for i in range(1, 5)]
    play(waits, depart, start=end, phase='GREEN')
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'
    assert row['crossing_phase'] == 'GREEN'

    # Sol S2 Round 2: stationary 60 px box clips to 45 px (aging 60 px out of history),
    # recovers to 57 px for two frames (clearing pending_crossing), then re-clips to 51 px
    # past stop_y=272 while top stays fixed at y1=220 -> must NOT falsely become SERVED.
    waits_r2 = book(side=lambda p: 272.-p[1])
    seq_r2 = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    seq_r2 += [[car(1, 265., x=150., height=45., speed=0.)] for _ in range(10)]
    seq_r2 += [[car(1, 277., x=150., height=57., speed=0.)] for _ in range(2)]
    seq_r2 += [[car(1, 271., x=150., height=51., speed=0.)]]
    play(waits_r2, seq_r2, phase='RED')
    row_r2 = only_row(waits_r2)
    assert not row_r2['completed'] and not row_r2['crossed_stop_line']
    assert waits_r2.active[(0, 1)].get('pending_crossing') is not None

    # Positive control: after two 57 px behind-line frames, a genuine full-height 57 px
    # departure (top moves upward to y1=214, y2=271 past stop_y=272) confirms SERVED immediately.
    waits_dep = book(side=lambda p: 272.-p[1])
    seq_dep = seq_r2[:-1] + [[car(1, 271., x=150., height=57., speed=2.)]]
    play(waits_dep, seq_dep, phase='GREEN')
    assert only_row(waits_dep)['crossed_stop_line'] and only_row(waits_dep)['tracking_state'] == 'SERVED'

    # Sol S2 Round 3: stationary clip starting behind the line (60 -> 51 x10 -> 45 at y1=220,
    # stop_y=270) must save the 60 px baseline at 51 px before 60 px ages out of history.
    waits_r3 = book(side=lambda p: 270.-p[1])
    seq_r3 = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    seq_r3 += [[car(1, 271., x=150., height=51., speed=0.)] for _ in range(10)]
    seq_r3 += [[car(1, 265., x=150., height=45., speed=0.)]]
    play(waits_r3, seq_r3, phase='RED')
    row_r3 = only_row(waits_r3)
    assert not row_r3['completed'] and not row_r3['crossed_stop_line']
    assert waits_r3.active[(0, 1)].get('pending_crossing') is not None

    # Sol S2 Round 4: gradual staircase shrinkage behind the line (60 x5 -> 54 x10 -> 48
    # at y1=220, stop_y=270) where neither step alone reaches 15% must still preserve the
    # 60 px baseline while pinned and catch the cumulative 20% shrink at the crossing.
    waits_r4 = book(side=lambda p: 270.-p[1])
    seq_r4 = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    seq_r4 += [[car(1, 274., x=150., height=54., speed=0.)] for _ in range(10)]
    seq_r4 += [[car(1, 268., x=150., height=48., speed=0.)]]
    play(waits_r4, seq_r4, phase='RED')
    row_r4 = only_row(waits_r4)
    assert not row_r4['completed'] and not row_r4['crossed_stop_line']
    assert waits_r4.active[(0, 1)].get('pending_crossing') is not None

    # Sol S2 Round 5: equal-height newer stationary geometry (y1=216, h=60 x6 -> y1=220, h=60 x12
    # -> y1=220, h=48 at stop_y=270) must refresh the saved baseline's rear to y2=280 rather than
    # retaining the stale y2=276 rear (which puts bottom share at 8/12 < 0.70 and falsely crosses).
    waits_r5 = book(side=lambda p: 270.-p[1])
    seq_r5 = [[car(1, 276., x=150., height=60., speed=0.)] for _ in range(6)]
    seq_r5 += [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
    seq_r5 += [[car(1, 268., x=150., height=48., speed=0.)]]
    play(waits_r5, seq_r5, phase='RED')
    row_r5 = only_row(waits_r5)
    assert not row_r5['completed'] and not row_r5['crossed_stop_line']
    assert waits_r5.active[(0, 1)].get('pending_crossing') is not None

    # Sol S2 Round 6: corroborated recent baselines AND saved baselines are all evaluated so
    # neither an equal-height tie-break nor a 1 px taller saved baseline hides a qualifying clip.
    for label, seq_r6 in (
        ('recent_shift_up_equal_h',
         [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(6)]
         + [[car(1, 276., x=150., height=60., speed=0.)] for _ in range(2)]
         + [[car(1, 268., x=150., height=48., speed=0.)]]),
        ('saved_1px_taller_before_recent',
         [[car(1, 276., x=150., height=61., speed=0.)] for _ in range(6)]
         + [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
         + [[car(1, 268., x=150., height=48., speed=0.)]]),
        ('saved_1px_taller_then_partial_shrink',
         [[car(1, 276., x=150., height=61., speed=0.)] for _ in range(6)]
         + [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
         + [[car(1, 274., x=150., height=54., speed=0.)] for _ in range(10)]
         + [[car(1, 268., x=150., height=48., speed=0.)]]),
        # Sol S2 Round 7: interior nondominated (h=60, rear=238.0) between (61, 233.3) and (54, 239.2).
        ('nondominated_interior_height_rear',
         [[car(1, 276., x=150., height=61., speed=0.)] for _ in range(6)]
         + [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
         + [[car(1, 277., x=150., height=54., speed=0.)] for _ in range(10)]
         + [[car(1, 268., x=150., height=48., speed=0.)]]),
        # Sol S11 Round 7: equal-score horizontal anchor (x=157) not dominated by older x=150 anchor.
        ('equal_score_distinct_x_drift_gate',
         [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(6)]
         + [[car(1, 280., x=157., height=60., speed=0.)] for _ in range(12)]
         + [[car(1, 274., x=157., height=54., speed=0.)] for _ in range(10)]
         + [[car(1, 268., x=160., height=48., speed=0.)]]),
        # Sol S2 Round 8: over-cap Pareto frontier jitter must not evict high-coverage (280, 60)
        # (tested both at 17 points and 40 points > MAX_CLIP_BASELINES).
        ('over_cap_frontier_preserves_qualifying_corner_17',
         [[car(1, 276., x=150., height=61., speed=0.)] for _ in range(6)]
         + [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
         + [obs for i in range(17) for obs in [[car(1, 278.25, x=150., height=56.25-i/16., speed=0.)]]*2]
         + [[car(1, 278.25, x=150., height=56.25-16/16., speed=0.)] for _ in range(12)]
         + [[car(1, 268., x=150., height=48., speed=0.)]]),
        ('over_cap_frontier_preserves_qualifying_corner_40',
         [[car(1, 276., x=150., height=61., speed=0.)] for _ in range(6)]
         + [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
         + [obs for i in range(40) for obs in [[car(1, 278.25, x=150., height=56.25-i/40., speed=0.)]]*2]
         + [[car(1, 278.25, x=150., height=56.25-39/40., speed=0.)] for _ in range(12)]
         + [[car(1, 268., x=150., height=48., speed=0.)]]),
        # Sol S11 Round 8: equal-score horizontal jitter must not evict older x=157 drift anchor
        # (tested both at 17 jitter values and 40 jitter values > MAX_CLIP_BASELINES).
        ('over_cap_equal_score_x_jitter_17',
         [[car(1, 280., x=157., height=60., speed=0.)] for _ in range(12)]
         + [obs for i in range(17) for obs in [[car(1, 280., x=150.+i/32., height=60., speed=0.)]]*2]
         + [[car(1, 274., x=157., height=54., speed=0.)] for _ in range(12)]
         + [[car(1, 268., x=160., height=48., speed=0.)]]),
        ('over_cap_equal_score_x_jitter_40',
         [[car(1, 280., x=157., height=60., speed=0.)] for _ in range(12)]
         + [obs for i in range(40) for obs in [[car(1, 280., x=150.+i/80., height=60., speed=0.)]]*2]
         + [[car(1, 274., x=157., height=54., speed=0.)] for _ in range(12)]
         + [[car(1, 268., x=160., height=48., speed=0.)]]),
        # Sol S11 Round 9: lossless drift-gate containment preserves x=157 across 40 higher-rear
        # anchors and subsequent slow horizontal drift to x=165.9375 (within 9 px of x=157).
        ('lossless_x_drift_containment_round9',
         [[car(1, 280., x=157., height=60., speed=0.)] for _ in range(12)]
         + [[car(1, 280.125, x=156.875, height=60., speed=0.)] for _ in range(12)]
         + [obs for i in range(40) for obs in [[car(1, 280.125, x=150.+0.171875*i, height=60., speed=0.)]]*2]
         + [[car(1, 274., x=min(156.703125+0.5*i, 165.9375), height=54., speed=0.)] for i in range(32)]
         + [[car(1, 268., x=165.9375, height=48., speed=0.)]]),
    ):
        waits_r6 = book(side=lambda p: 270.-p[1])
        play(waits_r6, seq_r6, phase='RED')
        row_r6 = only_row(waits_r6)
        assert not row_r6['completed'] and not row_r6['crossed_stop_line'], label
        assert waits_r6.active[(0, 1)].get('pending_crossing') is not None, label

    # Sol S2 Round 9: lossless nondominated frontier preserves (280, 60) across 40 finer steps
    # at stop_y=274 so (273.65, 50.95) with top pinned at y1=222.7 stays pending, not SERVED.
    waits_r9 = book(side=lambda p: 274.-p[1])
    seq_r9 = ([[car(1, 280., x=150., height=60.1, speed=0.)] for _ in range(12)]
              + [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(12)]
              + [obs for i in range(40) for obs in [[car(1, 280., x=150., height=59.75-0.18*i, speed=0.)]]*2]
              + [[car(1, 278.7, x=150., height=56., speed=0.)] for _ in range(12)]
              + [[car(1, 273.65, x=150., height=50.95, speed=0.)]])
    play(waits_r9, seq_r9, phase='RED')
    row_r9 = only_row(waits_r9)
    assert not row_r9['completed'] and not row_r9['crossed_stop_line']
    assert waits_r9.active[(0, 1)].get('pending_crossing') is not None


def test_sustained_stationary_jitter_per_frame_cost_stays_bounded():
    """Sol S12/S13: 6 stopped vehicles jittering < 1 px (1D, 3D, Pareto) for 60 s
    keep stationary, clipped, and genuine-departure crossing times bounded with O(1)
    skyline stop-line checks; slow drift compacts column arrays; growing height evicts
    dominated entries from live indexes."""
    waits = book(side=lambda p: 270.-p[1])
    tail_ms = []
    for f in range(1800):
        dx = (f % 1799)/1800.  # 1799 distinct sub-pixel x positions inside < 1 px
        obs = [car(1+k, 280., x=100.+80.*k+dx, height=60., speed=0.) for k in range(6)]
        started = time.perf_counter()
        waits.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
        elapsed_ms = (time.perf_counter()-started)*1000.
        if f >= 1650:
            tail_ms.append(elapsed_ms)
    median_ms = sorted(tail_ms)[len(tail_ms)//2]
    assert median_ms < 5.0, f'1D jitter median tail frame time {median_ms:.2f} ms exceeds budget'
    # Clipped frame on all 6 vehicles after 60 s of jitter is still caught as pending, not SERVED.
    waits.observe([car(1+k, 268., x=100.+80.*k+0.5, height=48., speed=0.) for k in range(6)],
                  'RED', 1800/FPS, 1801, 1, unassigned=())
    assert all(not r['crossed_stop_line'] for r in waits.records.values())

    # 3D (x, y2, h) sub-pixel jitter for 60 s: both stationary tail AND crossing frame stay fast.
    waits_3d = book(side=lambda p: 270.-p[1])
    tail_3d_ms = []
    for f in range(1800):
        obs = []
        for k in range(6):
            dx = (((f*37+k*13) % 997)/997.-0.5)*0.9
            dy = (((f*53+k*29) % 991)/991.-0.5)*0.9
            dh = (((f*71+k*41) % 983)/983.-0.5)*0.9
            obs.append(car(1+k, 280.+dy, x=100.+80.*k+dx, height=60.+dh, speed=0.))
        started = time.perf_counter()
        waits_3d.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
        elapsed_ms = (time.perf_counter()-started)*1000.
        if f >= 1650:
            tail_3d_ms.append(elapsed_ms)
    median_3d_ms = sorted(tail_3d_ms)[len(tail_3d_ms)//2]
    assert median_3d_ms < 5.0, f'3D jitter median tail frame time {median_3d_ms:.2f} ms exceeds budget'
    started = time.perf_counter()
    waits_3d.observe([car(1+k, 268., x=100.+80.*k, height=48., speed=0.) for k in range(6)],
                     'RED', 1800/FPS, 1801, 1, unassigned=())
    cross_3d_ms = (time.perf_counter()-started)*1000.
    assert cross_3d_ms < 10.0, f'3D jitter crossing frame time {cross_3d_ms:.2f} ms exceeds budget'
    assert all(not r['crossed_stop_line'] for r in waits_3d.records.values())

    # Sol S12: genuine departure after 60 s of Pareto accumulation (900 unique (h, y2)/track),
    # INCLUDING partial drift pruning on the departure frame, must avoid rebuilding/sorting
    # and check only the tallest eligible baseline on the straight stop line.
    side_calls = [0]
    def counted_side(p):
        side_calls[0] += 1
        return 275.-p[1]
    waits_dep = book(side=counted_side)
    for f in range(1800):
        step = f//2
        dx = -0.5 if step < 15 else 0.0
        h = 60.-0.001*step
        y2 = 276.-0.0004*step
        obs = [car(1+k, y2, x=100.+80.*k+dx, height=h, speed=0.) for k in range(6)]
        waits_dep.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    assert len(waits_dep.active[(0, 1)]['clip_baselines']) >= 850
    side_calls[0] = 0
    started = time.perf_counter()
    # x=+8.6 prunes the 15 step<15 baselines while ~885 survive; y1=214 clears y=275 for all.
    waits_dep.observe([car(1+k, 262., x=100.+80.*k+8.6, height=48., speed=1.) for k in range(6)],
                      'GREEN', 1800/FPS, 1801, 1, unassigned=())
    depart_ms = (time.perf_counter()-started)*1000.
    assert depart_ms < 15.0, f'genuine departure after partial prune took {depart_ms:.2f} ms'
    assert side_calls[0] <= 120, f'genuine departure scanned baselines ({side_calls[0]} side calls)'
    assert all(r['tracking_state'] == 'SERVED' and r['crossed_stop_line'] for r in waits_dep.records.values())

    # Sol S12 (Round 13/14): scattered equal-height partial pruning across mixed leaves must
    # stay fast on BOTH genuine departure AND clipped crossing without refreshing KD leaves.
    for mode, stop_y, dep_py, dep_spd, expect_served in (
        ('depart', 275., 262., 1., True),
        ('clipped', 270., 264., 0., False),
    ):
        waits_eq = book(side=lambda p, sy=stop_y: sy-p[1])
        for f in range(1800):
            step = f//2
            dx = -0.5 if (step % 4 == 0 and step < 800) else 0.0  # 200 scattered across all leaves
            y2 = 276.+0.002*step  # 900 distinct y2 values sharing the EXACT SAME height=60
            obs = [car(1+k, y2, x=100.+80.*k+dx, height=60., speed=0.) for k in range(6)]
            waits_eq.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
        assert len(waits_eq.active[(0, 1)]['clip_baselines']._blocks) >= 4
        started = time.perf_counter()
        waits_eq.observe([car(1+k, dep_py, x=100.+80.*k+8.6, height=48., speed=dep_spd) for k in range(6)],
                         'GREEN', 1800/FPS, 1801, 1, unassigned=())
        eq_ms = (time.perf_counter()-started)*1000.
        assert eq_ms < 15.0, f'scattered equal-height partial-prune {mode} took {eq_ms:.2f} ms'
        assert all(r['crossed_stop_line'] is expect_served for r in waits_eq.records.values())

    # Sol S13: growing height with fixed top/center evicts dominated live baselines (retaining 1/track).
    waits_grow = book(side=lambda p: 270.-p[1])
    for f in range(1800):
        h = 60.+0.88*(f/1799.)
        y2 = 220.+h
        obs = [car(1+k, y2, x=100.+80.*k, height=h, speed=0.) for k in range(6)]
        waits_grow.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    for k in range(6):
        saved = waits_grow.active[(0, 1+k)]['clip_baselines']
        assert len(saved) == 1 and len(saved._x) == 1

    # Sol S13: taller baseline also evicts previously accumulated spatially-spread dominated entries.
    waits_spread = book(side=lambda p: 270.-p[1])
    seq_spread = ([obs for i in range(40) for obs in [[car(1, 280., x=150.+0.02*i, height=60., speed=0.)]]*2]
                  + [[car(1, 290., x=150.4, height=70., speed=0.)] for _ in range(12)])
    play(waits_spread, seq_spread, phase='RED')
    saved_spread = waits_spread.active[(0, 1)]['clip_baselines']
    assert len(saved_spread) == 1 and saved_spread._blocks[0].h[0] == 70.
    assert saved_spread._h[saved_spread._root.idxs[0]] == 70.

    # Slow unidirectional drift AND mass pruning followed by covered frames compact dead slots.
    waits_creep = book(side=lambda p: 270.-p[1])
    for f in range(1850):
        waits_creep.observe([car(1, 280., x=100.+0.02*f, height=60., speed=0.)],
                            'RED', f/FPS, f+1, 1, unassigned=())
    saved_creep = waits_creep.active[(0, 1)].get('clip_baselines')
    assert saved_creep is not None and len(saved_creep) > 0
    assert sum(saved_creep._alive) == len(saved_creep)
    assert len(saved_creep._x) <= 4*max(len(saved_creep), 64)

    # Sol Round 14: pruning 901 baselines down to 1 live + 300 stable covered frames compacts slots;
    # dense pruning (901 -> ~200 live) also compacts incrementally without leaving dead slots resident.
    waits_prune1 = book(side=lambda p: 270.-p[1])
    for f in range(1802):
        step = f//2
        x = 150. if step < 900 else 155.
        waits_prune1.observe([car(1, 276.+0.001*step, x=x, height=60., speed=0.)],
                             'RED', f/FPS, f+1, 1, unassigned=())
    for f in range(1802, 2102):
        waits_prune1.observe([car(1, 277., x=160., height=60., speed=0.)],
                             'RED', f/FPS, f+1, 1, unassigned=())
    saved_p1 = waits_prune1.active[(0, 1)]['clip_baselines']
    assert len(saved_p1) == len(saved_p1._x) <= 8

    waits_prune_dense = book(side=lambda p: 270.-p[1])
    for f in range(1800):
        step = f//2
        x = 150. if step < 700 else 155.
        waits_prune_dense.observe([car(1, 276.+0.001*step, x=x, height=60., speed=0.)],
                                  'RED', f/FPS, f+1, 1, unassigned=())
    for f in range(1800, 1850):
        waits_prune_dense.observe([car(1, 277., x=160., height=60., speed=0.)],
                                  'RED', f/FPS, f+1, 1, unassigned=())
    saved_pd = waits_prune_dense.active[(0, 1)]['clip_baselines']
    assert 150 <= len(saved_pd) == len(saved_pd._x) <= 250

    # Sol Round 15 (S18): holding a pending crossing across the stop line must ALSO advance
    # compaction instead of stalling 201 live anchors in 901 slots.
    waits_pending_compact = book(side=lambda p: 270.-p[1])
    for f in range(1800):
        step = f//2
        x = 150. if step < 700 else 155.
        waits_pending_compact.observe([car(1, 276.+0.001*step, x=x, height=60., speed=0.)],
                                      'RED', f/FPS, f+1, 1, unassigned=())
    for f in range(1800, 2101):
        waits_pending_compact.observe([car(1, 264., x=160., height=48., speed=0.)],
                                      'RED', f/FPS, f+1, 1, unassigned=())
    assert not only_row(waits_pending_compact)['crossed_stop_line']
    assert waits_pending_compact.active[(0, 1)].get('pending_crossing') is not None
    saved_pc = waits_pending_compact.active[(0, 1)]['clip_baselines']
    assert 150 <= len(saved_pc) == len(saved_pc._x) <= 250

    # Sol Round 15 (S12): monotonic multi-thousand-entry compaction must stay within per-frame
    # budget without whole-subtree KD/R-tree rebuild spikes.
    waits_mono = book(side=lambda p: 270.-p[1])
    for f in range(4000):
        x = 150. if f < 2100 else 155.
        waits_mono.observe([car(1, 276.+0.0005*f, x=x, height=60., speed=0.)],
                           'RED', f/FPS, f+1, 1, unassigned=())
    mono_times = []
    for k in range(50):
        f = 4000+k
        started = time.perf_counter()
        waits_mono.observe([car(1, 278.0, x=160., height=60., speed=0.)],
                           'RED', f/FPS, f+1, 1, unassigned=())
        mono_times.append((time.perf_counter()-started)*1000.)
    saved_mono = waits_mono.active[(0, 1)]['clip_baselines']
    assert max(mono_times) < 15.0, f'monotonic compaction spike {max(mono_times):.2f} ms'
    assert 1500 <= len(saved_mono) == len(saved_mono._x) <= 2000

    # Sol Round 16 (S12 budgeted dominance, R-tree level balance, and shared 64-copy budget):
    # 1. Anti-correlated near-dominators across 6 tracks stay inside the frame budget on hold.
    waits_dom = book(side=lambda p: 270.-p[1])
    for i in range(600):
        t = 1.+i/600.
        d = -1e-6 if i % 2 == 0 else 1e-6
        h = 60.+t/0.15
        for rep in range(2):
            f = 2*i+rep
            obs = [car(1+k, 216.+1.0000001*t-d+h, x=150.+80.*k+1.0000001*t+d, height=h, speed=0.) for k in range(6)]
            waits_dom.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    hold_dom = []
    for k_f in range(5):
        f = 1200+k_f
        started = time.perf_counter()
        waits_dom.observe([car(1+k, 276., x=150.+80.*k, height=60., speed=0.) for k in range(6)],
                          'RED', f/FPS, f+1, 1, unassigned=())
        hold_dom.append((time.perf_counter()-started)*1000.)
    assert max(hold_dom) < 15.0, f'anti-correlated dominance frame took {max(hold_dom):.2f} ms'

    # 2. Dominance deletion preserves uniform R-tree leaf depth across non-root levels.
    def leaf_depths(node, depth=0):
        if node is None: return set()
        if node.idxs is not None: return {depth}
        out = set()
        for child in node.children: out |= leaf_depths(child, depth+1)
        return out
    waits_bal = book(side=lambda p: 270.-p[1])
    for f in range(702):
        waits_bal.observe([car(1, 280., x=150.+min(f, 699)/200., height=60., speed=0.)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    for k_f in range(20):
        f = 702+k_f
        waits_bal.observe([car(1, 290., x=149.7, height=70., speed=0.)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    assert len(leaf_depths(waits_bal.active[(0, 1)]['clip_baselines']._root)) == 1

    # 3. Small prunes across 6 tracks (13 live/track = 78 total) respect the shared 64-copy budget.
    waits_b64 = book(side=lambda p: 270.-p[1])
    for f in range(1800):
        step = f//2
        x_off = 0. if step < 887 else 5.
        waits_b64.observe([car(1+k, 276.+0.001*step, x=150.+80.*k+x_off, height=60., speed=0.) for k in range(6)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    copied_on_1800 = [0]
    from src.experiment.vehicle_waits import _ClipBaselines
    orig_step = _ClipBaselines._step_compact
    def spy_step(self, budget):
        n = orig_step(self, budget)
        copied_on_1800[0] += n
        return n
    _ClipBaselines._step_compact = spy_step
    try:
        waits_b64.observe([car(1+k, 264., x=150.+80.*k+10., height=48., speed=0.) for k in range(6)],
                          'RED', 1800/FPS, 1801, 1, unassigned=())
    finally:
        _ClipBaselines._step_compact = orig_step
    assert copied_on_1800[0] <= 64
    for f in range(1801, 1805):
        waits_b64.observe([car(1+k, 264., x=150.+80.*k+10., height=48., speed=0.) for k in range(6)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    for k in range(6):
        cb_k = waits_b64.active[(0, 1+k)]['clip_baselines']
        assert len(cb_k) == len(cb_k._x) <= 20

    # Sol Round 17 (S12):
    # 1. Exact monotonic height_eligible cutoff stops scanning immediately at top_speed == 0.35*h + 5e-11.
    waits_spd = book(side=lambda p: 270.-p[1])
    for f in range(3000):
        x_off = 0.9*(f//2)/1500.
        waits_spd.observe([car(1+k, 276., x=150.+80.*k+x_off, height=60., speed=0.) for k in range(6)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    for j in range(8):
        f = 3000+j
        waits_spd.observe([car(1+k, 276., x=150.9+80.*k+0.7*j, height=60., speed=0.) for k in range(6)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    started = time.perf_counter()
    waits_spd.observe([car(1+k, 264., x=150.9+80.*k+0.7*8+5e-11, height=48., speed=1.) for k in range(6)],
                      'GREEN', 3008/FPS, 3009, 1, unassigned=())
    spd_ms = (time.perf_counter()-started)*1000.
    assert spd_ms < 10.0, f'top-speed cutoff crossing took {spd_ms:.2f} ms'
    assert all(r['crossed_stop_line'] for r in waits_spd.records.values())

    # 2. Six-track dominance wave queues deferred eviction even after budget exhaustion and converges fairly.
    waits_wave6 = book(side=lambda p: 270.-p[1])
    for f in range(2000):
        waits_wave6.observe([car(1+k, 280., x=150.+80.*k+(f//2)/1000., height=60., speed=0.) for k in range(6)],
                            'RED', f/FPS, f+1, 1, unassigned=())
    for j in range(301):
        f = 2000+j
        waits_wave6.observe([car(1+k, 290., x=150.5+80.*k, height=70., speed=0.) for k in range(6)],
                            'RED', f/FPS, f+1, 1, unassigned=())
    for k in range(6):
        cb_k = waits_wave6.active[(0, 1+k)]['clip_baselines']
        assert len(cb_k) == 1 and len(cb_k._x) <= 8

    # Sol Round 18 (S12):
    # 1. Equal-height block rear rejection jumps past all blocks in O(log B); dead blocks are skipped via live_count.
    from src.experiment.vehicle_waits import _rear_err, OCCLUSION_BOTTOM_SHARE
    cb_eq = _ClipBaselines()
    for i in range(4096):
        b = (150.+1e-5*i, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.))
        cb_eq._insert_raw(*b)
    side_calls_eq = [0]
    def counted_eq_side(p):
        side_calls_eq[0] += 1
        return 270.-p[1]
    started = time.perf_counter()
    res_eq = cb_eq.best_for_crossing(
        lambda hb: True, lambda y2b, hb: (y2b-269.) >= OCCLUSION_BOTTOM_SHARE*(hb-48.)-1e-9,
        269.-OCCLUSION_BOTTOM_SHARE*48.-1e-9-_rear_err(269., 48.), 48., 221., 150., counted_eq_side, None,
    )
    assert res_eq is None and side_calls_eq[0] == 0 and (time.perf_counter()-started)*1000. < 2.0

    # Dead h=60 blocks above live h=55 blocks must be skipped without scanning dead slots
    # on both genuine crossing (h=48 > 0.85*55) and clipped crossing (h=46 <= 0.85*55).
    for cross_h, expect_crossed in ((48., True), (46., False)):
        waits_dead = book(side=lambda p: 270.-p[1])
        for f in range(1800):
            j = 0.9*(f//2)/900.
            waits_dead.observe([car(1+k, 276., x=150.+80.*k+j, height=60., speed=0.) for k in range(6)],
                               'RED', f/FPS, f+1, 1, unassigned=())
        for f in range(1800, 3600):
            j = 0.9*((f-1800)//2)/900.
            waits_dead.observe([car(1+k, 271., x=155.+80.*k+j, height=55., speed=0.) for k in range(6)],
                               'RED', f/FPS, f+1, 1, unassigned=())
        for j in range(16):
            f = 3600+j
            waits_dead.observe([car(1+k, 271., x=160.+80.*k, height=55., speed=0.) for k in range(6)],
                               'RED', f/FPS, f+1, 1, unassigned=())
        started = time.perf_counter()
        waits_dead.observe([car(1+k, 216.+cross_h, x=160.+80.*k, height=cross_h, speed=0.) for k in range(6)],
                           'RED', 3616/FPS, 3617, 1, unassigned=())
        assert (time.perf_counter()-started)*1000. < 10.0
        assert all(r['crossed_stop_line'] is expect_crossed for r in waits_dead.records.values())

    # 2. Reserved sweep budget prevents busy jittering tracks from starving a queued eviction,
    # and independent queued dominators (h=70 @ x=150.25 and h=71 @ x=154.25) do not overwrite each other.
    import random
    rng_s = random.Random(23)
    waits_starve = book(side=lambda p: 270.-p[1])
    for f in range(1800):
        obs = [car(1+k, 280., x=150.+80.*k+0.02*rng_s.random(), height=60., speed=0.) for k in range(5)]
        obs.append(car(6, 280., x=550.+(f//2)/900., height=60., speed=0.))
        waits_starve.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    for f in range(1800, 2200):
        obs = [car(1+k, 280., x=150.+80.*k+0.02*rng_s.random(), height=60., speed=0.) for k in range(5)]
        obs.append(car(6, 290., x=550.5, height=70., speed=0.))
        waits_starve.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    cb_6 = waits_starve.active[(0, 6)]['clip_baselines']
    assert len(cb_6) == 1 and len(cb_6._x) <= 64

    cb_q = _ClipBaselines()
    for base_x in (150.25, 154.25):
        for i in range(128):
            bx = base_x-0.2+0.4*i/127.
            cb_q.add((bx, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
                     compact=False, dom_budget=[0])
    cb_q.add((150.25, 220., 290., 70., 290.-OCCLUSION_BOTTOM_SHARE*70., _rear_err(290., 70.)),
             compact=False, dom_budget=[0])
    cb_q.add((154.25, 220., 291., 71., 291.-OCCLUSION_BOTTOM_SHARE*71., _rear_err(291., 71.)),
             compact=False, dom_budget=[0])
    while cb_q.has_dom_sweep():
        cb_q.step_dom_sweep(64)
    assert len(cb_q) == 2

    # Sol Round 19 (S12):
    # 1. Mixed-height blocks reject by rear bound before per-height checks (both live and after
    #    pruning higher-rear outliers in every block).
    waits_mix = book(side=lambda p: 270.-p[1])
    for f in range(4096):
        h = 60.+0.00001*(f//2)
        waits_mix.observe([car(1+k, 276., x=150.+80.*k, height=h, speed=0.) for k in range(6)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    started = time.perf_counter()
    waits_mix.observe([car(1+k, 269., x=150.+80.*k, height=48., speed=0.) for k in range(6)],
                      'RED', 4096/FPS, 4097, 1, unassigned=())
    mix_ms = (time.perf_counter()-started)*1000.
    assert mix_ms < 10.0, f'mixed-height block rear rejection took {mix_ms:.2f} ms'
    assert all(r['crossed_stop_line'] for r in waits_mix.records.values())

    waits_mix_pruned = book(side=lambda p: 270.-p[1])
    for f in range(4096):
        step = f//2
        outlier = step % 128 == 0
        h = 60.+0.00001*step
        y2 = 278. if outlier else 276.
        x_off = 0. if outlier else 5.
        waits_mix_pruned.observe([car(1+k, y2, x=150.+80.*k+x_off, height=h, speed=0.) for k in range(6)],
                                 'RED', f/FPS, f+1, 1, unassigned=())
    started = time.perf_counter()
    waits_mix_pruned.observe([car(1+k, 269., x=160.+80.*k, height=48., speed=0.) for k in range(6)],
                             'RED', 4096/FPS, 4097, 1, unassigned=())
    mix_pruned_ms = (time.perf_counter()-started)*1000.
    assert mix_pruned_ms < 10.0, f'pruned mixed-height rear rejection took {mix_pruned_ms:.2f} ms'
    assert all(r['crossed_stop_line'] for r in waits_mix_pruned.records.values())

    # 2. Deferred dominance enqueueing stays O(1) without copying _dom_queue, child-envelope checks
    #    prevent false-positive queue growth, and compaction remaps large queues within budget.
    rng19 = random.Random(19)
    waits_q19 = book(side=lambda p: 270.-p[1])
    q19_tail = []
    for f in range(2500):
        if f < 15:
            obs = [car(1+k, 218.5+60., x=150.+80.*k, height=60., speed=0.) for k in range(6)]
        elif f < 30:
            obs = [car(1+k, 215.+61., x=150.+80.*k, height=61., speed=0.) for k in range(6)]
        else:
            obs = [car(1+k, 216.+61., x=150.+80.*k+0.18*(rng19.random()-0.5), height=61., speed=0.) for k in range(6)]
        started = time.perf_counter()
        waits_q19.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
        if f >= 2350:
            q19_tail.append((time.perf_counter()-started)*1000.)
    q19_median = sorted(q19_tail)[len(q19_tail)//2]
    assert q19_median < 3.0, f'Round 19 enqueue tail median {q19_median:.2f} ms'
    assert all(len(waits_q19.active[(0, 1+k)]['clip_baselines']._dom_queue) == 0 for k in range(6))

    # Sol Round 20:
    # 1. Queue compaction with > 64 queued entries genuinely enters multi-step compaction,
    #    caps work per step at budget=64, and preserves surviving queue order.
    cb_bigq = _ClipBaselines()
    for i in range(500):
        bx = 150.+0.001*i
        cb_bigq.add((bx, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.)),
                    compact=False, dom_budget=[0])
    started = time.perf_counter()
    for j in range(200):
        bx = 150.+0.0005*j
        cb_bigq.add((bx, 216., 277., 61., 277.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(277., 61.)),
                    compact=False, dom_budget=[0])
    assert (time.perf_counter()-started)*1000. < 50.0
    assert len(cb_bigq._dom_queue) == 199
    # Prune all 500 h=60 entries (y1_hi=225.0 < 225.1) while keeping all 200 h=61 entries (y1_hi=225.15).
    cb_bigq.prune(150.0, 225.1)
    assert cb_bigq.needs_compact() and len(cb_bigq) == 200
    expected_q = [(cb_bigq._x[q], cb_bigq._y1[q], cb_bigq._y2[q], cb_bigq._h[q])
                  for q in cb_bigq._dom_queue if cb_bigq._alive[q]]
    assert len(expected_q) == 199
    compact_steps = 0
    while cb_bigq.needs_compact():
        step_work = cb_bigq.maybe_compact(64)
        assert 0 < step_work <= 64
        compact_steps += 1
    assert compact_steps >= 6
    actual_q = [(cb_bigq._x[q], cb_bigq._y1[q], cb_bigq._y2[q], cb_bigq._h[q]) for q in cb_bigq._dom_queue]
    assert actual_q == expected_q
    assert len(cb_bigq) == len(cb_bigq._x) == 200

    # 2. Deferred eviction makes forward progress even with a 1-unit per-track sweep budget
    #    (300 tracks > sweep_budget=256): after 30 jitter frames at h=60 and 90 frames at h=80,
    #    every track finishes its sweep and converges to a single dominant baseline.
    waits_300 = book(side=lambda p: 270.-p[1])
    for f in range(30):
        dx = 0.02*(f % 15)
        waits_300.observe([car(1+k, 280., x=150.+80.*k+dx, height=60., speed=0.) for k in range(300)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    for f in range(30, 120):
        waits_300.observe([car(1+k, 300., x=150.+80.*k+0.15, height=80., speed=0.) for k in range(300)],
                          'RED', f/FPS, f+1, 1, unassigned=())
    for k in range(300):
        cb_k = waits_300.active[(0, 1+k)]['clip_baselines']
        assert len(cb_k) == 1 and not cb_k.has_dom_sweep()

    # Sol Round 21:
    # 1. Live exact-key index deduplicates repeating baselines (> 32 distinct keys) in O(1)
    #    before spending dominance budget, preventing unbounded duplicate growth.
    waits_dup = book(side=lambda p: 270.-p[1])
    for f in range(400):
        obs = [car(1+k, 280., x=150.+80.*k+0.02*(f % 50), height=60., speed=0.) for k in range(6)]
        waits_dup.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    for k in range(6):
        cb_d = waits_dup.active[(0, 1+k)]['clip_baselines']
        assert len(cb_d) == 50 and len(cb_d._x) == 50

    # 2. Fixed copy range at rebuild start + mirroring later inserts lets compaction finish
    #    even when 65 drifting tracks (> 64-unit budget) prune and append 1 baseline per frame.
    waits_drift65 = book(side=lambda p: 270.-p[1])
    for f in range(800):
        obs = [car(1+k, 280., x=150.+80.*k+0.2*f, height=60., speed=0.) for k in range(65)]
        waits_drift65.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    for k in range(65):
        cb_65 = waits_drift65.active[(0, 1+k)]['clip_baselines']
        assert len(cb_65) <= 60 and len(cb_65._x) <= 220

    # Sol Round 22:
    # 1. Compaction preserves one authoritative FIFO for _dom_queue and never requeues
    #    already-completed candidates or lets newly appended work jump ahead of older entries.
    cb_fifo = _ClipBaselines()
    for i in range(500):
        cb_fifo.add((150.+0.001*i, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.)),
                    compact=False, dom_budget=[0])
    for j in range(200):
        cb_fifo.add((150.+0.0005*j, 216., 277., 61., 277.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(277., 61.)),
                    compact=False, dom_budget=[0])
    cb_fifo.prune(150.0, 225.1)  # prune all 500 h=60 -> 200 live h=61, 199 queued
    assert cb_fifo.needs_compact() and len(cb_fifo._dom_queue) == 199
    # Advance compaction through Phase 1 + 198 of 199 Phase 2 queue items, leaving 1 unremapped.
    for _ in range(6):
        cb_fifo.maybe_compact(64)
    assert cb_fifo._rebuild is not None
    # Append a newer candidate mid-compaction, then pop/finish 2 sweeps via step_dom_sweep:
    # the oldest queued entry must run before the newly appended one, and completed sweeps
    # must never reappear after compaction finishes.
    newest_x = 150.15
    cb_fifo.add((newest_x, 216., 278., 62., 278.-OCCLUSION_BOTTOM_SHARE*62., _rear_err(278., 62.)),
                compact=False, dom_budget=[0])
    # Note: 62 px may evict recent 61 px tail items; verify FIFO order of surviving queue across swap.
    pre_swap_keys = [(cb_fifo._x[q], cb_fifo._h[q]) for q in cb_fifo._dom_queue if q >= 0 and cb_fifo._alive[q]]
    assert pre_swap_keys[-1] == (newest_x, 62.)
    # Pop the front-most pending item via step_dom_sweep (must be pre_swap_keys[0], NOT newest_x!).
    if cb_fifo._dom_sweep is not None:
        cb_fifo._dom_pending[cb_fifo._dom_sweep[0]] = 0
        cb_fifo._dom_sweep = None
    cb_fifo.step_dom_sweep(1)
    assert cb_fifo._dom_sweep is not None
    assert (cb_fifo._x[cb_fifo._dom_sweep[0]], cb_fifo._h[cb_fifo._dom_sweep[0]]) == pre_swap_keys[0]
    # Finish this front-most sweep and finish compaction; completed item must NOT reappear,
    # and all remaining pending items keep exact FIFO order ending with (newest_x, 62.).
    while cb_fifo._dom_sweep is not None and cb_fifo._dom_sweep[0] == 501:
        cb_fifo.step_dom_sweep(1)
    remaining_before_swap = [(cb_fifo._x[q], cb_fifo._h[q]) for q in cb_fifo._dom_queue if q >= 0 and cb_fifo._alive[q]]
    assert remaining_before_swap == pre_swap_keys[1:]
    while cb_fifo.needs_compact():
        cb_fifo.maybe_compact(64)
    remaining_after_swap = [(cb_fifo._x[q], cb_fifo._h[q]) for q in cb_fifo._dom_queue if q >= 0 and cb_fifo._alive[q]]
    assert remaining_after_swap == remaining_before_swap
    assert pre_swap_keys[0] not in remaining_after_swap and remaining_after_swap[-1] == (newest_x, 62.)

    # Also verify draining the entire queue (including a mid-compaction append) before swap
    # leaves _dom_queue completely empty after swap (zero requeued completed candidates).
    cb_drain = _ClipBaselines()
    for i in range(500):
        cb_drain.add((150.+0.001*i, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.)),
                     compact=False, dom_budget=[0])
    for j in range(100):
        cb_drain.add((150.+0.0005*j, 216., 277., 61., 277.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(277., 61.)),
                     compact=False, dom_budget=[0])
    cb_drain.prune(150.0, 225.1)
    for _ in range(3):
        cb_drain.maybe_compact(64)
    assert cb_drain._rebuild is not None
    cb_drain.add((150.15, 216., 278., 62., 278.-OCCLUSION_BOTTOM_SHARE*62., _rear_err(278., 62.)),
                 compact=False, dom_budget=[0])
    while cb_drain.has_dom_sweep():
        cb_drain.step_dom_sweep(256)
    while cb_drain.needs_compact():
        cb_drain.maybe_compact(64)
    assert not cb_drain.has_dom_sweep() and len(cb_drain._dom_queue) == 0

    # 2. Frozen sweep_end + hot dominator cache prevents continuous h=60 inserts on 300 tracks
    #    from starving a second queued 61 px sweep or accumulating dominated baselines.
    waits_s300 = book(side=lambda p: 270.-p[1])
    seq_s = 0
    for i in range(6):
        waits_s300.observe([car(1+k, 280., x=150.+80.*k+0.01*i, height=60., speed=0.) for k in range(300)],
                           'RED', seq_s/FPS, seq_s+1, 1, unassigned=()); seq_s += 1
    for i in range(4):
        waits_s300.observe([car(1+k, 281., x=150.+80.*k, height=61., speed=0.) for k in range(300)],
                           'RED', seq_s/FPS, seq_s+1, 1, unassigned=()); seq_s += 1
    for i in range(4):
        waits_s300.observe([car(1+k, 281., x=150.5+80.*k, height=61., speed=0.) for k in range(300)],
                           'RED', seq_s/FPS, seq_s+1, 1, unassigned=()); seq_s += 1
    for f in range(400):
        waits_s300.observe([car(1+k, 280., x=150.5+80.*k+0.0001*(f+1), height=60., speed=0.) for k in range(300)],
                           'RED', seq_s/FPS, seq_s+1, 1, unassigned=()); seq_s += 1
    for k in range(300):
        cb_s = waits_s300.active[(0, 1+k)]['clip_baselines']
        assert len(cb_s) == 2 and not cb_s.has_dom_sweep()

    # Sol Round 23:
    # 1. Queueing a live dominator via covered() during compaction Phase 1 (before its slot
    #    is copied into shadow) must not record -1 or drop the sweep across the swap.
    cb_p1 = _ClipBaselines()
    for i in range(160):
        cb_p1.add((150.+0.01*i, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.)),
                  compact=False, dom_budget=[0])
    # Add 4 late slots surviving at x=165..165.4: one h=60 dominated by one h=61, plus two others.
    cb_p1.add((165.0, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.)),
              compact=False, dom_budget=[0])
    cb_p1.add((165.0, 216., 277., 61., 277.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(277., 61.)),
              compact=False, dom_budget=[0])
    cb_p1.add((165.4, 216., 277., 61., 277.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(277., 61.)),
              compact=False, dom_budget=[0])
    cb_p1.add((165.8, 216., 277., 61., 277.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(277., 61.)),
              compact=False, dom_budget=[0])
    cb_p1._dom_sweep = None; cb_p1._dom_queue.clear(); cb_p1._dom_cache.clear()
    cb_p1._dom_pending[:] = bytes(len(cb_p1._dom_pending))
    cb_p1.prune(165.4, 216.)  # kills first 160 slots, leaving 4 live slots (160..163)
    assert len(cb_p1) == 4 and cb_p1.needs_compact()
    # Keep a dummy active sweep on slot 162 so covered() appends slot 161 to _dom_queue during Phase 1.
    cb_p1._dom_sweep = (162, -2, len(cb_p1._alive))
    cb_p1._dom_pending[162] = 1
    cb_p1.maybe_compact(1)  # copies slot 160 only; slot 161 is not mapped yet!
    assert cb_p1._rebuild is not None and 161 not in cb_p1._rebuild[3]
    assert cb_p1.covered((165.01, 216., 276., 60., 276.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(276., 60.)),
                         dom_budget=[64])
    assert 161 in list(cb_p1._dom_queue)
    while cb_p1.needs_compact():
        cb_p1.maybe_compact(64)
    while cb_p1.has_dom_sweep():
        cb_p1.step_dom_sweep(64)
    assert len(cb_p1) == 3

    # 2. Resumable cover-probe + fair unused-budget distribution finds a dominator at the tail
    #    of 32+ anchors even across 25 busy tracks (never accumulating 1,000 dominated sightings).
    waits_25 = book(side=lambda p: 270.-p[1])
    seq_25 = 0
    for a in range(32):
        for _ in range(2):
            obs = [car(1+k, 281., x=150.+80.*k+0.25*a, height=61., speed=0.) for k in range(25)]
            waits_25.observe(obs, 'RED', seq_25/FPS, seq_25+1, 1, unassigned=()); seq_25 += 1
    for f in range(400):
        obs = [car(1+k, 280., x=150.+80.*k+0.25*31+0.0001*(f+1), height=60., speed=0.) for k in range(25)]
        waits_25.observe(obs, 'RED', seq_25/FPS, seq_25+1, 1, unassigned=()); seq_25 += 1
    for k in range(25):
        cb_25 = waits_25.active[(0, 1+k)]['clip_baselines']
        assert len(cb_25) == 32 and not cb_25.has_dom_sweep()

    # 3. _SlotFIFO indexed reads and chunked growth appends stay O(1) at 131,072 entries.
    from src.experiment.vehicle_waits import _SlotFIFO
    fifo = _SlotFIFO()
    for i in range(131072):
        fifo.append(i)
    started = time.perf_counter()
    fifo.append(131072)
    assert (time.perf_counter()-started)*1000. < 1.0
    started = time.perf_counter()
    mid_vals = [fifo[65536+j] for j in range(64)]
    assert (time.perf_counter()-started)*1000. < 1.0
    assert mid_vals == [65536+j for j in range(64)]

    # Sol Round 24:
    # 1. Pending sweep membership bitmap deduplicates repeated dominator enqueues across
    #    >16-entry queues (25 tracks cycling 32 nearby anchors over 800 frames).
    waits_cyc = book(side=lambda p: 270.-p[1])
    seq_c = 0
    for a in range(32):
        for _ in range(2):
            obs = [car(1+k, 281., x=150.+80.*k+0.25*a, height=61., speed=0.) for k in range(25)]
            waits_cyc.observe(obs, 'RED', seq_c/FPS, seq_c+1, 1, unassigned=()); seq_c += 1
    for f in range(400):
        a = f % 32
        obs = [car(1+k, 280., x=150.+80.*k+0.25*a+0.0001*(f+1), height=60., speed=0.) for k in range(25)]
        waits_cyc.observe(obs, 'RED', seq_c/FPS, seq_c+1, 1, unassigned=()); seq_c += 1
    for k in range(25):
        cb_c = waits_cyc.active[(0, 1+k)]['clip_baselines']
        assert len(cb_c._dom_queue) <= len(cb_c._x)
        q_list = list(cb_c._dom_queue)
        assert len(q_list) == len(set(q_list))

    # 2. Lossless resumable cover bitmap never drops overflow candidates (20 tall + 20 short
    #    with dom_budget=[0] drains to exact 20 live baselines with 0 covered survivors).
    cb_20 = _ClipBaselines()
    for i in range(20):
        bx = 150.+0.3*i
        cb_20.add((bx, 220., 281., 61., 281.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(281., 61.)),
                  compact=False, dom_budget=[0])
    cb_20._dom_cache.clear()
    for i in range(20):
        bx = 150.+0.3*i+0.01
        cb_20.add((bx, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
                  compact=False, dom_budget=[0])
    while cb_20.has_dom_sweep():
        cb_20.step_dom_sweep(1)
    assert len(cb_20) == 20 and not cb_20.has_dom_sweep()

    # 3. Dedicated 1-unit scheduling flag alternates fairly between eviction and cover jobs
    #    even when a new cover candidate arrives before every 1-unit maintenance step.
    cb_alt = _ClipBaselines()
    for i in range(25):
        bx = 150.+0.3*i
        cb_alt.add((bx, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
                   compact=False, dom_budget=[0])
    for i in range(25):
        bx = 150.+0.3*i
        cb_alt.add((bx, 220., 281., 61., 281.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(281., 61.)),
                   compact=False, dom_budget=[0])
    cb_alt._dom_cache.clear()
    ev_calls = [0]; cov_calls = [0]
    orig_ev = _ClipBaselines._step_evict_sweep; orig_cov = _ClipBaselines._step_cover_probe
    _ClipBaselines._step_evict_sweep = lambda self, b: (ev_calls.__setitem__(0, ev_calls[0]+1), orig_ev(self, b))[1]
    _ClipBaselines._step_cover_probe = lambda self, b: (cov_calls.__setitem__(0, cov_calls[0]+1), orig_cov(self, b))[1]
    try:
        for i in range(30):
            bx = 150.+0.3*(i % 25)+0.001*(i+1)
            cb_alt.add((bx, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
                       compact=False, dom_budget=[0])
            cb_alt.step_dom_sweep(1)
    finally:
        _ClipBaselines._step_evict_sweep = orig_ev
        _ClipBaselines._step_cover_probe = orig_cov
    assert ev_calls[0] >= 10 and cov_calls[0] >= 10

    # 4. Wrapped dominance window (n_obs=25, dom_start=2) distributes budget fairly in input order
    #    so track 0 does not consume all 384 units while tracks 2..24 get 0.
    waits_wrap = book(side=lambda p: 270.-p[1])
    for f in range(5):
        obs = [car(1+k, 281., x=150.+80.*k+0.01*f, height=61., speed=0.) for k in range(25)]
        waits_wrap.observe(obs, 'RED', f/FPS, f+1, 1, unassigned=())
    waits_wrap._dom_cursor = 2  # dom_start = (2*24) % 25 = 23 -> wraps around track 0..21
    seen_shares = []
    orig_eval = waits_wrap._evaluate_crossing
    def spy_eval(state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                 valid_motion, continuing, keep_pending, dom_budget):
        seen_shares.append((observation.local_id, dom_budget[0]))
        dom_budget[0] = 0  # simulate busy track consuming its whole share
        return orig_eval(state, prev_obs, observation, phase, timestamp, sequence, cycle_id,
                         valid_motion, continuing, keep_pending, dom_budget)
    waits_wrap._evaluate_crossing = spy_eval
    waits_wrap.observe([car(1+k, 281., x=150.+80.*k+0.2, height=61., speed=0.) for k in range(25)],
                       'RED', 5/FPS, 6, 1, unassigned=())
    nonzero = [sh for _, sh in seen_shares if sh > 0]
    assert len(nonzero) == 24 and max(nonzero) == 16 and min(nonzero) == 16

    # Sol Round 25:
    # 1. Indexed resumable cover search + batch eviction keeps 64 tracks cycling 32 nearby
    #    anchors under sustained short-baseline jitter from growing unboundedly.
    waits_64 = book(side=lambda p: 270.-p[1])
    seq_64 = 0
    for a in range(32):
        for _ in range(2):
            obs = [car(1+k, 281., x=150.+80.*k+0.24*a, height=61., speed=0.) for k in range(64)]
            waits_64.observe(obs, 'RED', seq_64/FPS, seq_64+1, 1, unassigned=()); seq_64 += 1
    for f in range(1000):
        a = f % 32
        obs = [car(1+k, 280., x=150.+80.*k+0.24*a+1e-5*(f+1), height=60., speed=0.) for k in range(64)]
        waits_64.observe(obs, 'RED', seq_64/FPS, seq_64+1, 1, unassigned=()); seq_64 += 1
    live_64 = sorted(len(waits_64.active[(0, 1+k)]['clip_baselines']) for k in range(64))
    assert live_64[32] <= 64, f'64-track cycling median live {live_64[32]} grew unbounded'

    # 2. Bulk drift pruning immediately clears both _dom_pending and _cover_pending and
    #    decrements _cover_count in both live and shadow indexes (zero dead flags survive swap).
    cb_pr = _ClipBaselines()
    for i in range(700):
        bx = 150.+0.001*i
        cb_pr.add((bx, 220., 281., 61., 281.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(281., 61.)),
                  compact=False, dom_budget=[0])
    cb_pr._dom_cache.clear()
    for i in range(700):
        bx = 150.+0.001*i+1e-5
        cb_pr.add((bx, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
                  compact=False, dom_budget=[0])
    # Surviving tall/short pair at x=165.0
    cb_pr.add((165.0, 220., 281., 61., 281.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(281., 61.)),
              compact=False, dom_budget=[0])
    cb_pr._dom_cache.clear()
    cb_pr.add((165.01, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
              compact=False, dom_budget=[0])
    assert cb_pr._cover_count >= 700
    cb_pr.prune(165.0, 220.)
    assert len(cb_pr) == 2 and cb_pr._cover_count == 1
    assert sum(cb_pr._cover_pending) == 1 and sum(cb_pr._dom_pending) <= 1
    while cb_pr.needs_compact():
        cb_pr.maybe_compact(64)
    assert sum(cb_pr._cover_pending) == cb_pr._cover_count == 1
    while cb_pr.has_dom_sweep():
        cb_pr.step_dom_sweep(16)
    assert len(cb_pr) == 1 and cb_pr._cover_count == 0 and not cb_pr.has_dom_sweep()

    # 3. Cross-block (h, y2) runs (>256 entries) resume cleanly across block boundaries under
    #    1-unit budgets for both indexed cover probes and indexed batch eviction.
    cb_xb = _ClipBaselines()
    for i in range(400):
        bx = 100.+0.25*i
        cb_xb.add((bx, 220., 281., 61., 281.-OCCLUSION_BOTTOM_SHARE*61., _rear_err(281., 61.)),
                  compact=False, dom_budget=[0])
    assert len(cb_xb._blocks) >= 2
    cb_xb._dom_cache.clear()
    for bx in (100.01, 100.26, 150.01, 199.74):
        cb_xb.add((bx, 220., 280., 60., 280.-OCCLUSION_BOTTOM_SHARE*60., _rear_err(280., 60.)),
                  compact=False, dom_budget=[0])
    for _ in range(2000):
        if not cb_xb.has_dom_sweep():
            break
        cb_xb.step_dom_sweep(1)
    assert len(cb_xb) == 400 and not cb_xb.has_dom_sweep()

    # 4. Fixed-depth radix _SlotFIFO has zero directory resizing across growth + negative prepend.
    import sys
    fifo_2l = _SlotFIFO()
    for i in range(262144):
        fifo_2l.append(i)
    root_sz = sys.getsizeof(fifo_2l._root)
    started = time.perf_counter()
    fifo_2l.append(262144)
    assert (time.perf_counter()-started)*1000. < 1.0
    assert sys.getsizeof(fifo_2l._root) == root_sz and fifo_2l[262144] == 262144
    for i in range(1, 300):
        fifo_2l.appendleft(-i)
    assert fifo_2l[0] == -299 and fifo_2l.popleft() == -299 and fifo_2l.pop() == 262144

    # Sol Round 26:
    # 1. Vertical jitter (32 y2 anchors) AND 3D height+x jitter across 64 tracks stay bounded.
    for mode in ('vertical', '3d_height_x'):
        waits_v = book(side=lambda p: 270.-p[1])
        seq_v = 0
        for a in range(32):
            for _ in range(2):
                if mode == 'vertical':
                    obs = [car(1+k, 281.+0.24*a, x=150.+80.*k, height=61., speed=0.) for k in range(64)]
                else:
                    obs = [car(1+k, 281.+0.01*a, x=150.+80.*k+0.24*a, height=61.+0.01*a, speed=0.) for k in range(64)]
                waits_v.observe(obs, 'RED', seq_v/FPS, seq_v+1, 1, unassigned=()); seq_v += 1
        for f in range(1000):
            a = f % 32
            if mode == 'vertical':
                obs = [car(1+k, 280.+0.24*a, x=150.+80.*k+1e-5*(f+1), height=60., speed=0.) for k in range(64)]
            else:
                obs = [car(1+k, 280., x=150.+80.*k+0.24*a+1e-5*(f+1), height=60., speed=0.) for k in range(64)]
            waits_v.observe(obs, 'RED', seq_v/FPS, seq_v+1, 1, unassigned=()); seq_v += 1
        live_v = sorted(len(waits_v.active[(0, 1+k)]['clip_baselines']) for k in range(64))
        assert live_v[32] <= 48, f'64-track {mode} cycling median live {live_v[32]} grew unbounded'

    # 2. Same-height probes never retire multiple candidates on a 1-unit budget returning 0.
    cb_sh = _ClipBaselines()
    for y in (279., 281.):
        cb_sh.add((150., y-60., y, 60., y-42., _rear_err(y, 60.)), compact=False, dom_budget=[0])
    for i in range(32):
        y = 280.+0.001*i
        cb_sh.add((150., y-60., y, 60., y-42., _rear_err(y, 60.)), compact=False, dom_budget=[0])
    # Force pending cover probes even after strict max_h <= bh skips queueing them, and verify 1-unit charge.
    for s in range(2, 34):
        cb_sh._queue_cover_probe(s)
    used_1 = cb_sh.step_dom_sweep(1)
    assert used_1 == 1 and cb_sh.has_dom_sweep()

    # 3. Synthetic non-standard geometry equal-height baseline with larger y2 is still evicted.
    cb_syn = _ClipBaselines()
    cb_syn.add((0., 0., 62., 61., 90., 1e-12), compact=False, dom_budget=[0])
    cb_syn.add((1., 0., 60., 60., 90., 1e-12), compact=False, dom_budget=[0])
    cb_syn.add((0., 0., 61., 61., 100., 1e-12), compact=False, dom_budget=[0])
    while cb_syn.has_dom_sweep():
        cb_syn.step_dom_sweep(1)
    assert len(cb_syn) == 2 and cb_syn._alive[0] == 0 and cb_syn._alive[1] == 1 and cb_syn._alive[2] == 1

    # Sol Round 27:
    # 1. Combined 2D (8x4) x/y grid + 32 distinct heights + drifting short height across 64 tracks
    #    stays inside split-safe R-tree DFS without accumulating dominated shorter baselines.
    waits_grid = book(side=lambda p: 270.-p[1])
    seq_g = 0
    for a in range(32):
        for _ in range(2):
            y1_a = 220.+0.35*(a//8); h_a = 61.+0.015*a
            obs = [car(1+k, y1_a+h_a, x=150.+24.*k+0.35*(a % 8), height=h_a, speed=0.) for k in range(64)]
            waits_grid.observe(obs, 'RED', seq_g/FPS, seq_g+1, 1, unassigned=()); seq_g += 1
    for f in range(1000):
        a = f % 32
        y1_f = 220.+0.35*(a//8); h_f = 60.-1e-5*f
        obs = [car(1+k, y1_f+h_f, x=150.+24.*k+0.35*(a % 8)+1e-5*(f+1), height=h_f, speed=0.) for k in range(64)]
        waits_grid.observe(obs, 'RED', seq_g/FPS, seq_g+1, 1, unassigned=()); seq_g += 1
    live_g = sorted(len(waits_grid.active[(0, 1+k)]['clip_baselines']) for k in range(64))
    assert live_g[32] <= 64, f'64-track 2D+height grid median live {live_g[32]} grew unbounded'

    # 2. Equal-key synthetic scans (513 tuples with identical x, y2, h) strictly obey budget=1
    #    for BOTH evict sweeps and cover probes even across R-tree splits.
    cb_eqev = _ClipBaselines()
    for i in range(513):
        cb_eqev.add((0., i*1e-6, 60., 60., 42., 1e-12), compact=False, dom_budget=[0])
    cb_eqev.add((0., 0., 61., 61., 50., 1e-12), compact=False, dom_budget=[0])
    assert cb_eqev.step_dom_sweep(1) == 1
    for i in range(4):
        cb_eqev.add((100.+10.*i, 200., 275., 75., 0., 1e-12), compact=False, dom_budget=[0])
    assert cb_eqev.step_dom_sweep(1) <= 1
    assert cb_eqev.step_dom_sweep(1) <= 1
    cb_eqcv = _ClipBaselines()
    for i in range(513):
        cb_eqcv.add((0., i*1e-6, 61., 61., 50., 1e-12), compact=False, dom_budget=[0])
    cb_eqcv.add((0., 10.0, 60., 60., 42., 1e-12), compact=False, dom_budget=[0])
    cb_eqcv._queue_cover_probe(513)
    assert cb_eqcv.step_dom_sweep(1) == 1
    for i in range(4):
        cb_eqcv.add((100.+10.*i, 200., 275., 75., 0., 1e-12), compact=False, dom_budget=[0])
    assert cb_eqcv.step_dom_sweep(1) <= 1
    assert cb_eqcv.step_dom_sweep(1) <= 1


def test_missed_frame_clears_crossing_clip_evidence_but_preserves_fragment_history():
    """Sol S14: a missed frame clears saved clip baselines AND crossing clip history so
    pre-gap boxes cannot suppress a post-gap crossing, while gap-tolerant motion history
    is still preserved for fragment continuation."""
    waits = book(side=lambda p: 270.-p[1])
    for f in range(5):
        waits.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', f/FPS, f+1, 1, unassigned=())
    waits.observe([], 'RED', 5/FPS, 6, 1, unassigned=())  # one missed frame
    waits.observe([car(1, 274., x=150., height=54., speed=0.)], 'RED', 6/FPS, 7, 1, unassigned=())
    waits.observe([car(1, 268., x=150., height=48., speed=0.)], 'RED', 7/FPS, 8, 1, unassigned=())
    row = only_row(waits)
    assert row['tracking_state'] == 'SERVED' and row['crossed_stop_line']

    # Fragment matching across a brief missed frame still retains gap-tolerant motion history.
    waits_link = book(side=lambda p: -1.)
    for f in range(5):
        waits_link.observe([car(1, 200.+3.*f, x=150., height=60., speed=3.)], 'GREEN', f/FPS, f+1, 1, unassigned=())
    waits_link.observe([], 'GREEN', 5/FPS, 6, 1, unassigned=())  # missed frame inside max_gap_sec
    waits_link.observe([car(1, 200.+3.*6, x=150., height=60., speed=3.)], 'GREEN', 6/FPS, 7, 1, unassigned=())
    assert len(waits_link.active[(0, 1)]['history']) == 6
    waits_link.observe([], 'GREEN', 7/FPS, 8, 1, unassigned=())
    waits_link.observe([car(2, 200.+3.*8, x=150., height=60., speed=3.)], 'GREEN', 8/FPS, 9, 1, unassigned=())
    assert len(waits_link.records) == 1 and only_row(waits_link).get('identity_links')

    # Sol S14 (Round 12): invalid geometry (zero-width box, even with explicit height) must
    # clear pending_crossing, clip_baselines, and clip_history so subsequent crossings serve.
    from dataclasses import replace
    waits_bad = book(side=lambda p: 270.-p[1])
    for f in range(5):
        waits_bad.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', f/FPS, f+1, 1, unassigned=())
    waits_bad.observe([car(1, 268., x=150., height=48., speed=0.)], 'RED', 5/FPS, 6, 1, unassigned=())
    assert waits_bad.active[(0, 1)].get('pending_crossing') is not None
    # Zero-width malformed box with explicit height=48: must clear all clip state and not enter clip_history.
    bad_obs = replace(car(1, 268., x=150., height=48., speed=0.), bbox=(150., 220., 150., 268.), height=48.)
    waits_bad.observe([bad_obs], 'RED', 6/FPS, 7, 1, unassigned=())
    assert waits_bad.active[(0, 1)].get('pending_crossing') is None
    assert waits_bad.active[(0, 1)].get('clip_baselines') is None
    assert len(waits_bad.active[(0, 1)].get('clip_history', ())) == 0
    # Next two valid observations cross the stop line -> must complete SERVED.
    waits_bad.observe([car(1, 274., x=150., height=54., speed=0.)], 'RED', 7/FPS, 8, 1, unassigned=())
    waits_bad.observe([car(1, 268., x=150., height=48., speed=0.)], 'RED', 8/FPS, 9, 1, unassigned=())
    row_bad = only_row(waits_bad)
    assert row_bad['tracking_state'] == 'SERVED' and row_bad['crossed_stop_line']

    # Sol S14 (Round 13): clip -> partial recovery behind line -> zero-width box ACROSS line
    # clears pending_crossing WITHOUT falling through to SERVED; and a malformed box BEHIND
    # the line cannot serve as the starting point for a crossing on the very next frame.
    waits_hold = book(side=lambda p: 270.-p[1])
    for f in range(5):
        waits_hold.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', f/FPS, f+1, 1, unassigned=())
    waits_hold.observe([car(1, 268., x=150., height=48., speed=0.)], 'RED', 5/FPS, 6, 1, unassigned=())
    waits_hold.observe([car(1, 274., x=150., height=54., speed=0.)], 'RED', 6/FPS, 7, 1, unassigned=())
    assert waits_hold.active[(0, 1)].get('pending_crossing') is not None
    bad_across = replace(car(1, 268., x=150., height=48., speed=0.), bbox=(150., 220., 150., 268.), height=48.)
    waits_hold.observe([bad_across], 'RED', 7/FPS, 8, 1, unassigned=())
    assert waits_hold.active[(0, 1)].get('pending_crossing') is None
    assert not only_row(waits_hold)['crossed_stop_line']
    # Malformed box BEHIND line cannot start a crossing on the next valid frame across the line.
    bad_behind = replace(car(1, 274., x=150., height=54., speed=0.), bbox=(150., 220., 150., 274.), height=54.)
    waits_hold.observe([bad_behind], 'RED', 8/FPS, 9, 1, unassigned=())
    waits_hold.observe([car(1, 268., x=150., height=48., speed=0.)], 'RED', 9/FPS, 10, 1, unassigned=())
    assert not only_row(waits_hold)['crossed_stop_line']

    # Sol Round 19: explicit oversized height (bbox=None) or out-of-bounds inferred top edge
    # (py - height beyond _MAX_PIXEL_COORD) must NOT confirm a pending crossing; it clears clip state.
    for bad_pt, bad_h in (((150., 265.), 1e40), ((150., -5e14), 8e14)):
        waits_big_h = book(side=lambda p: 270.-p[1])
        for f in range(6):
            waits_big_h.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', f/FPS, f+1, 1, unassigned=())
        waits_big_h.observe([car(1, 265., x=150., height=45., speed=0.)], 'RED', 6/FPS, 7, 1, unassigned=())
        assert waits_big_h.active[(0, 1)].get('pending_crossing') is not None
        huge_obs = VehicleObservation(1, 'car', 0., bad_pt, in_region=True, velocity_valid=True,
                                      epoch=0, bbox=None, height=bad_h)
        waits_big_h.observe([huge_obs], 'RED', 7/FPS, 8, 1, unassigned=())
        assert waits_big_h.active[(0, 1)].get('pending_crossing') is None
        assert not only_row(waits_big_h)['crossed_stop_line']

    # Sol Round 13 (stale re-entry): after acquisition invalidation clears previous=None, a fresh
    # outside EXIT_PENDING observation behind the line followed by a consecutive in-ROI crossing
    # must evaluate the crossing independently of wait interval accounting and complete SERVED.
    waits_stale = book(exit_persistence_sec=0.3, side=lambda p: 270.-p[1])
    waits_stale.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', 0., 1, 1, unassigned=())
    waits_stale.invalidate(0.5, 'acquisition_stale')
    assert waits_stale.active[(0, 1)]['previous'] is None
    waits_stale.observe([car(1, 276., x=150., height=60., speed=2., in_region=False)], 'GREEN', 1.0, 2, 2, unassigned=())
    assert waits_stale.active[(0, 1)]['previous'] is None and waits_stale.active[(0, 1)]['exit_last'] is not None
    waits_stale.observe([car(1, 264., x=150., height=60., speed=2., in_region=True)], 'GREEN', 1.0+1/FPS, 3, 2, unassigned=())
    row_stale = only_row(waits_stale)
    assert row_stale['tracking_state'] == 'SERVED' and row_stale['crossed_stop_line'] and row_stale['crossing_cycle'] == 2

    # Sol Round 14 (outside-track veto after invalidation): a fresh outside EXIT_PENDING track
    # with previous=None must still veto an ambiguous occlusion-tail 1 -> 2 handoff, matching
    # the non-invalidated veto while the control without track 3 still links 1 -> 2.
    def run_post_invalidation_veto(include_track3, invalidate_track3):
        w = book(exit_persistence_sec=0.3, side=lambda p: -1.)
        if include_track3:
            w.observe([car(3, 280., x=150., height=60., speed=0., in_region=True)], 'RED', 0., 1, 1, unassigned=())
            if invalidate_track3:
                w.invalidate(0.1, 'acquisition_stale')
                assert w.active[(0, 3)]['previous'] is None
        t0 = 0.2
        seq_v = [
            [car(1, 280., x=150., height=60., speed=0.)],
            [car(1, 280., x=150., height=60., speed=0.)],
            [car(1, 280., x=150., height=60., speed=0.)],
            [car(1, 280., x=150., height=60., speed=0.)],
            [car(1, 265., x=150., height=45., speed=0.)],
            [car(1, 260., x=150., height=40., speed=0.)] + (
                [car(3, 280., x=150., height=60., speed=0., in_region=False)] if include_track3 else []
            ),
        ]
        for i, obs in enumerate(seq_v):
            w.observe(obs, 'RED', t0+i/FPS, 2+i, 1, unassigned=())
        if include_track3:
            assert w.active[(0, 3)]['exit_candidate'] is not None
            if invalidate_track3:
                assert w.active[(0, 3)]['previous'] is None
        for i in range(6):
            w.observe([], 'RED', t0+(len(seq_v)+i)/FPS, 2+len(seq_v)+i, 1, unassigned=())
        w.observe([car(2, 280., x=150., height=60., speed=0.)],
                  'RED', t0+(len(seq_v)+6)/FPS, 2+len(seq_v)+6, 1, unassigned=())
        return [l for r in w.records.values() for l in r.get('identity_links', ())]

    assert run_post_invalidation_veto(include_track3=False, invalidate_track3=False)  # control links 1 -> 2
    assert run_post_invalidation_veto(include_track3=True, invalidate_track3=False) == []
    assert run_post_invalidation_veto(include_track3=True, invalidate_track3=True) == []


def test_bottom_clip_rear_share_cutoff_uses_original_float_comparison():
    """Sol S15: conservative skyline bounds must retain boundary candidates and apply the
    exact original ``(y2_base - py) >= 0.7 * (h_base - height) - 1e-9`` comparison."""
    import struct
    f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
    y2_base, h_base = 275.0, f32(61.1)
    py_cross, h_cross = 265.8860023508535, f32(48.08)
    # Confirm this coordinate set distinguishes original vs rearranged float evaluation.
    assert (y2_base-py_cross) >= 0.7*(h_base-h_cross)-1e-9
    assert not ((y2_base-0.7*h_base) >= (py_cross-0.7*h_cross)-1e-9)
    waits = book(side=lambda p: 270.-p[1])
    y1_mid = py_cross-h_cross
    # Establish baseline (y2_base, h_base), then age it out of clip_history into _ClipBaselines.
    seq = ([[car(1, y2_base, x=150., height=h_base, speed=0.)] for _ in range(10)]
           + [[car(1, y1_mid+55., x=150., height=55., speed=0.)] for _ in range(12)]
           + [[car(1, py_cross, x=150., height=h_cross, speed=0.)]])
    play(waits, seq, phase='RED')
    row = only_row(waits)
    assert not row['completed'] and not row['crossed_stop_line']
    assert waits.active[(0, 1)].get('pending_crossing') is not None


def test_pending_crossing_confirms_across_consecutive_exit_pending_outside_frames():
    """Sol S3: an active crossing anchor is retained across consecutive outside frames
    while EXIT_PENDING, and confirms SERVED once the reconstructed rear clears the line."""
    stop_y = 270.
    # 1. Clip starts inside ROI, vehicle steps outside (EXIT_PENDING), clears rear next frame.
    waits = book(exit_persistence_sec=0.3, side=lambda p: stop_y-p[1])
    frames = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    frames += [[car(1, 265., x=150., height=45., speed=0.)]]  # clipped inside ROI -> pending set
    end = play(waits, frames, phase='GREEN')
    waits.observe([car(1, 260., x=150., height=45., speed=2., in_region=False)], 'GREEN', end/FPS, end+1, 1, unassigned=())
    row = only_row(waits)
    assert not row['completed'] and row['tracking_state'] == 'EXIT_PENDING'
    waits.observe([car(1, 253., x=150., height=45., speed=2., in_region=False)], 'GREEN', (end+1)/FPS, end+2, 1, unassigned=())
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'
    assert row['crossing_phase'] == 'GREEN'
    assert row['crossing_evidence']['from_side'] < 0 <= row['crossing_evidence']['to_side']

    # 2. Clip starts on the first outside frame, then clears rear on the next consecutive outside frame.
    waits_first = book(exit_persistence_sec=0.3, side=lambda p: stop_y-p[1])
    end2 = play(waits_first, [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)], phase='GREEN')
    waits_first.observe([car(1, 265., x=150., height=45., speed=0., in_region=False)], 'GREEN', end2/FPS, end2+1, 1, unassigned=())
    row2 = only_row(waits_first)
    assert not row2['completed'] and row2['tracking_state'] == 'EXIT_PENDING'
    waits_first.observe([car(1, 253., x=150., height=45., speed=2., in_region=False)], 'GREEN', (end2+1)/FPS, end2+2, 1, unassigned=())
    assert row2['crossed_stop_line'] and row2['tracking_state'] == 'SERVED'

    # 3. Dropped frame while EXIT_PENDING discards pending crossing (no crossing across a gap).
    waits_gap = book(exit_persistence_sec=0.3, side=lambda p: stop_y-p[1])
    end3 = play(waits_gap, frames, phase='GREEN')
    waits_gap.observe([car(1, 260., x=150., height=45., speed=2., in_region=False)], 'GREEN', end3/FPS, end3+1, 1, unassigned=())
    waits_gap.observe([], 'GREEN', (end3+1)/FPS, end3+2, 1, unassigned=())
    waits_gap.observe([car(1, 253., x=150., height=45., speed=2., in_region=False)], 'GREEN', (end3+2)/FPS, end3+3, 1, unassigned=())
    assert not only_row(waits_gap)['crossed_stop_line']

    # 4. Preceding outside frame with UNKNOWN phase yields crossing_phase == UNKNOWN.
    waits_unk = book(exit_persistence_sec=0.3, side=lambda p: stop_y-p[1])
    end4 = play(waits_unk, frames, phase='GREEN')
    waits_unk.observe([car(1, 260., x=150., height=45., speed=2., in_region=False)], 'UNKNOWN', end4/FPS, end4+1, None, unassigned=())
    waits_unk.observe([car(1, 253., x=150., height=45., speed=2., in_region=False)], 'GREEN', (end4+1)/FPS, end4+2, 1, unassigned=())
    row4 = only_row(waits_unk)
    assert row4['crossed_stop_line'] and row4['crossing_phase'] == 'UNKNOWN'
    assert 'crossing_phase_unknown' in row4['quality_flags']


def test_roi_exit_persistence_longer_than_association_gap_uses_outside_sighting_for_expiry():
    """Sol S4: when exit_persistence_sec > association_gap_sec, a single dropped frame after
    > association_gap_sec of tracked outside sightings must not close the row or split a return."""
    waits = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, exit_persistence_sec=1.5, association_gap_sec=1.0),
                         side=lambda p: -1.)
    for i in range(3):
        waits.observe([car(1, 120., speed=0.)], 'RED', i/FPS, i+1, 1)
    # Tracked outside continuously for 32 frames (1.067 s > 1.0 s association gap, < 1.5 s exit).
    for i in range(3, 35):
        waits.observe([car(1, 120., speed=0., in_region=False)], 'RED', i/FPS, i+1, 1)
    row = only_row(waits)
    assert not row['completed'] and row['tracking_state'] == 'EXIT_PENDING'
    assert row['last_seen'] == pytest.approx(2/FPS)
    # Single dropped frame at 35/FPS (1.10 s since last in-ROI, 0.033 s since last outside sighting).
    waits.observe([], 'RED', 35/FPS, 36, 1)
    assert not row['completed'] and row['last_seen'] == pytest.approx(2/FPS)
    # Returns inside ROI at 36/FPS: stays on the same single row (no :segment-1).
    waits.observe([car(1, 120., speed=0., in_region=True)], 'RED', 36/FPS, 37, 1)
    assert list(waits.records) == ['run:acquisition:0:1']
    assert row['unknown_sec'] == pytest.approx(34/FPS)

    # Vanishing permanently after outside sightings still expires after association_gap_sec from last outside sighting.
    waits_lost = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, exit_persistence_sec=1.5, association_gap_sec=1.0),
                              side=lambda p: -1.)
    for i in range(3):
        waits_lost.observe([car(1, 120., speed=0.)], 'RED', i/FPS, i+1, 1)
    for i in range(3, 35):
        waits_lost.observe([car(1, 120., speed=0., in_region=False)], 'RED', i/FPS, i+1, 1)
    waits_lost.observe([], 'RED', 64/FPS, 65, 1)  # 30/30 = 1.000 s after last outside sighting (not yet > 1.0)
    assert not only_row(waits_lost)['completed']
    waits_lost.observe([], 'RED', 65/FPS, 66, 1)  # 31/30 = 1.033 s > 1.0 s -> tracking_lost
    lost = only_row(waits_lost)
    assert lost['completed'] and 'tracking_lost' in lost['quality_flags']
    assert lost['last_seen'] == pytest.approx(2/FPS)


@pytest.mark.parametrize('fps', [30., 60., 120., 240.])
def test_bottom_clipped_stationary_car_does_not_falsely_cross_at_high_fps(fps):
    """Sol S6: motion history retains enough samples to span MIN_FIT_SPAN_SEC at high FPS
    so bottom-edge occlusion clipping stays open at 120/240 fps just as at 30 fps."""
    stop_y = 270.
    waits = book(side=lambda p: stop_y-p[1])
    pre_frames = int(round(0.15*fps))
    seq = 0
    for _ in range(pre_frames):
        waits.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', seq/fps, seq+1, 1, unassigned=())
        seq += 1
    for _ in range(5):
        waits.observe([car(1, 265., x=150., height=45., speed=0.)], 'RED', seq/fps, seq+1, 1, unassigned=())
        seq += 1
    row = only_row(waits)
    assert not row['completed'] and not row['crossed_stop_line']
    # Genuine forward motion clears the reconstructed rear and completes SERVED.
    for i in range(1, 5):
        waits.observe([car(1, (220.-5.*i)+45., x=150., height=45., speed=2.)], 'GREEN', seq/fps, seq+1, 1, unassigned=())
        seq += 1
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'


def test_exit_timer_and_crossing_continuity_share_max_gap_tolerance():
    """Sol S3 Round 2: with max_gap_sec=0.1, consecutive outside sightings at 0.3 s and
    0.4 s (0.4 - 0.3 == 0.1 + 3e-17) must not advance the exit timer while dropping the anchor."""
    waits = book(max_gap_sec=0.1, exit_persistence_sec=0.3, side=lambda p: 270.-p[1])
    waits.observe([car(1, 280., x=150., height=60., speed=0.)], 'GREEN', 0.0, 1, 1, unassigned=())
    waits.observe([car(1, 280., x=150., height=60., speed=0.)], 'GREEN', 0.1, 2, 1, unassigned=())
    waits.observe([car(1, 265., x=150., height=45., speed=0.)], 'GREEN', 0.2, 3, 1, unassigned=())
    waits.observe([car(1, 260., x=150., height=45., speed=2., in_region=False)], 'GREEN', 0.3, 4, 1, unassigned=())
    row = only_row(waits)
    assert not row['completed'] and row['tracking_state'] == 'EXIT_PENDING'
    waits.observe([car(1, 253., x=150., height=45., speed=2., in_region=False)], 'GREEN', 0.4, 5, 1, unassigned=())
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'


def test_single_tracked_outside_frame_preserves_clip_anchor_and_records_unknown_wait():
    """Sol S7: a single tracked outside frame must preserve crossing/clip continuity across
    consecutive same-ID ROI transitions while still recording the dip as unknown coverage."""
    stop_y = 270.
    waits = book(exit_persistence_sec=0.3, side=lambda p: stop_y-p[1])
    frames = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    frames += [[car(1, 265., x=150., height=45., speed=0.)]]
    end = play(waits, frames, phase='RED')
    # Single tracked outside frame while still clipped.
    waits.observe([car(1, 265., x=150., height=45., speed=0., in_region=False)], 'RED', end/FPS, end+1, 1, unassigned=())
    # Re-enters inside ROI at partial recovery h=51 (y2=271, behind line), then re-clips to h=45.
    waits.observe([car(1, 271., x=150., height=51., speed=0., in_region=True)], 'RED', (end+1)/FPS, end+2, 1, unassigned=())
    waits.observe([car(1, 265., x=150., height=45., speed=0., in_region=True)], 'RED', (end+2)/FPS, end+3, 1, unassigned=())
    row = only_row(waits)
    assert not row['completed'] and not row['crossed_stop_line']
    assert 'observation_gap' in row['quality_flags']
    assert row['unknown_sec'] == pytest.approx(2/FPS)
    assert row['red_wait_sec'] == pytest.approx(6/FPS)  # 5 before dip + 1 after re-entry


def test_unclipped_vehicle_exiting_behind_line_crosses_on_consecutive_outside_frame():
    """Sol S8: an unclipped vehicle that leaves the ROI behind the line and crosses on the
    next consecutive outside frame must complete SERVED rather than EXITED."""
    waits = book(exit_persistence_sec=0.3, side=lambda p: 270.-p[1])
    end = play(waits, [[car(1, 280., x=150., height=60., speed=2.)] for _ in range(5)], phase='GREEN')
    waits.observe([car(1, 275., x=150., height=60., speed=2., in_region=False)], 'GREEN', end/FPS, end+1, 1, unassigned=())
    row = only_row(waits)
    assert not row['completed'] and row['tracking_state'] == 'EXIT_PENDING'
    waits.observe([car(1, 265., x=150., height=60., speed=2., in_region=False)], 'GREEN', (end+1)/FPS, end+2, 1, unassigned=())
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'
    assert row['crossing_evidence']['from_side'] < 0 <= row['crossing_evidence']['to_side']

    # Same sequence with a stationary bottom clip on the second outside frame stays held, not SERVED.
    waits_clip = book(exit_persistence_sec=0.3, side=lambda p: 270.-p[1])
    end2 = play(waits_clip, [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)], phase='RED')
    waits_clip.observe([car(1, 280., x=150., height=60., speed=0., in_region=False)], 'RED', end2/FPS, end2+1, 1, unassigned=())
    waits_clip.observe([car(1, 265., x=150., height=45., speed=0., in_region=False)], 'RED', (end2+1)/FPS, end2+2, 1, unassigned=())
    assert not only_row(waits_clip)['crossed_stop_line']


def test_consecutive_outside_observations_advance_geometric_recovery():
    """Sol S9: two qualifying behind-line recoveries while EXIT_PENDING clear the stale
    pending anchor so a full-height 57 px departure past stop_y=270 confirms SERVED."""
    waits = book(exit_persistence_sec=0.3, side=lambda p: 270.-p[1])
    frames = [[car(1, 280., x=150., height=60., speed=0.)] for _ in range(5)]
    frames += [[car(1, 265., x=150., height=45., speed=0.)]]
    end = play(waits, frames, phase='GREEN')
    waits.observe([car(1, 277., x=150., height=57., speed=0., in_region=False)], 'GREEN', end/FPS, end+1, 1, unassigned=())
    waits.observe([car(1, 277., x=150., height=57., speed=0., in_region=False)], 'GREEN', (end+1)/FPS, end+2, 1, unassigned=())
    assert waits.active[(0, 1)].get('pending_crossing') is None
    waits.observe([car(1, 269., x=150., height=57., speed=2., in_region=False)], 'GREEN', (end+2)/FPS, end+3, 1, unassigned=())
    row = only_row(waits)
    assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'


def test_exit_pending_track_vetoes_occlusion_link_using_latest_outside_box():
    """Sol S10: an EXIT_PENDING track tracked outside > 0.35 s after its last in-ROI sighting
    must still veto an occlusion-tail link near its latest outside box when it vanishes."""
    waits = book(exit_persistence_sec=0.6, side=lambda p: -1.)
    # Track 5 enters ROI at f=0..2, then steps outside ROI at f=3..14 (0.40 s > 0.35 s after in-ROI)
    # sitting near (152, 160, h=60 -> y1=100). Meanwhile Track 1 builds a stationary occlusion tail
    # at (150, 100, h=100->55) at f=2..9, vanishes at f=10..14, and Track 5 vanishes at f=15.
    for i in range(15):
        obs = []
        if i < 3:
            obs.append(car(5, 160., x=250., height=60., speed=0., in_region=True))
        else:
            obs.append(car(5, 160., x=152., height=60., speed=0., in_region=False))
        if 2 <= i < 10:
            frac = (i-2)/7.
            h = 100.-45.*frac
            obs.append(car(1, 100.+h, x=150., height=h, speed=0., in_region=True))
        waits.observe(obs, 'RED', i/FPS, i+1, 1, unassigned=())
    # At f=15 (dt_in_roi for #5 = 13/30 = 0.433 s > 0.35 s, dt_outside for #5 = 1/30 = 0.033 s),
    # newborn #2 appears 2 px from #5's latest outside box and inside #1's occlusion top-gate.
    waits.observe([car(2, 160., x=150., height=60., speed=0., in_region=True)], 'RED', 15/FPS, 16, 1, unassigned=())
    assert _links(waits) == []

    # Sol S10 Round 3: a final heightless outside observation clears history, yet the track's
    # last valid box (within veto_window of its own timestamp) must still veto; once that valid
    # box ages past 0.35 s, the veto expires cleanly even if heightless outside frames continue.
    for extra_heightless, expect_links in ((1, []), (11, [(1, 2)])):
        waits_h = book(exit_persistence_sec=1.0, side=lambda p: -1.)
        valid_end = 14
        tail_start = valid_end+extra_heightless-12
        for i in range(valid_end+extra_heightless):
            obs = []
            if i < 3:
                obs.append(car(5, 160., x=250., height=60., speed=0., in_region=True))
            elif i < valid_end:
                obs.append(car(5, 160., x=152., height=60., speed=0., in_region=False))
            else:
                obs.append(car(5, 160., x=152., height=None, speed=0., in_region=False, bbox=False))
            if tail_start <= i < tail_start+8:
                frac = (i-tail_start)/7.
                h = 100.-45.*frac
                obs.append(car(1, 100.+h, x=150., height=h, speed=0., in_region=True))
            waits_h.observe(obs, 'RED', i/FPS, i+1, 1, unassigned=())
        idx = valid_end+extra_heightless
        waits_h.observe([car(2, 160., x=150., height=60., speed=0., in_region=True)], 'RED', idx/FPS, idx+1, 1, unassigned=())
        assert _links(waits_h) == expect_links, extra_heightless


def test_legacy_crossing_gate_unchanged_when_both_options_off():
    """Sol S3 Round 3: with both fragment_continuation and exit_persistence_sec off and
    max_gap_sec=0.1, an in-ROI sign change from 0.3 to 0.4 s (observation_gap) must not cross."""
    waits = VehicleWaits('run', RunPolicy(stop_persistence_sec=0, max_gap_sec=0.1), side=lambda p: p[1]-100.)
    waits.observe([car(1, 90., speed=2.)], 'GREEN', 0.3, 1, 1)
    waits.observe([car(1, 110., speed=2.)], 'GREEN', 0.4, 2, 1)
    row = only_row(waits)
    assert 'observation_gap' in row['quality_flags']
    assert not row['crossed_stop_line'] and not row['completed']


def test_exit_pending_crossing_during_first_green_counts_in_live_and_archived_n_served(tmp_path):
    """Sol S8 Round 3: a vehicle observed in-ROI only while phase is UNKNOWN (cycles=[])
    that crosses on a consecutive EXIT_PENDING outside frame after GREEN confirms cycle 1
    must count in n_served across live, retained-evicted, and archived store reads."""
    DARK = np.zeros((8, 8, 3), np.uint8)
    e = ObservationEngine('run-s8', RunPolicy(debounce_frames=1, stop_persistence_sec=0, exit_persistence_sec=0.3),
                          {'xmin': 0, 'ymin': 0, 'xmax': 8, 'ymax': 8}, lambda p: p[1]-100., lambda *_: None)
    e.enable_retention(1, tmp_path)
    store = ExperimentStore(tmp_path, e.run_id, {})
    # f=0,1: inside ROI while signal is dark (UNKNOWN -> cycles=[]).
    e.observe_acquisition(DARK, 0.0, 1, [car(1, 90., speed=2., in_region=True)])
    e.observe_acquisition(DARK, 1/FPS, 2, [car(1, 92., speed=2., in_region=True)])
    # f=2: steps outside ROI behind stop line while GREEN starts cycle 1 (EXIT_PENDING).
    e.observe_acquisition(GREEN, 2/FPS, 3, [car(1, 95., speed=2., in_region=False)])
    # f=3: crosses stop line on consecutive outside frame while GREEN cycle 1 is active.
    e.observe_acquisition(GREEN, 3/FPS, 4, [car(1, 105., speed=2., in_region=False)])
    # f=4: advance clock past crossing_at so green_start <= completed_at < end.
    e.observe_acquisition(GREEN, 4/FPS, 5, [])
    row = e.waits.records['run-s8:acquisition:0:1']
    assert row['crossed_stop_line'] and row['crossing_cycle'] == 1 and row['cycles'] == []
    assert e.snapshot()['comparisons'][0]['n_served'] == 1
    # Evict completed vehicle via retention commit + acknowledge, then check live and archived reads.
    state, token = e.prepare_commit(); store.persist(state); e.acknowledge(token)
    assert 'run-s8:acquisition:0:1' not in e.waits.records
    assert e.snapshot()['comparisons'][0]['n_served'] == 1
    e.close()
    state, token = e.prepare_commit(); store.persist(state); e.acknowledge(token)
    archived = store.read()
    assert archived['comparisons'][0]['n_served'] == 1

    # Sol Round 15 (default-off n_served invariance): with both options off (exit_persistence_sec=0)
    # and 1-frame debounce, GREEN(empty) -> YELLOW(car inside y=90) -> GREEN(car outside y=110)
    # keeps cycle 2 n_served == 0 across live, retention eviction, export, and reopened pre-upgrade
    # archives (where exit_persistence_sec is absent); native exit_persistence_sec=0.3 counts 1.
    import json
    YELLOW = np.full((8, 8, 3), (0, 255, 255), np.uint8)
    for exit_sec, expect_served in ((0.0, 0), (0.3, 1)):
        eng = ObservationEngine(f'run-r15-{int(exit_sec*10)}',
                                RunPolicy(debounce_frames=1, stop_persistence_sec=0, exit_persistence_sec=exit_sec),
                                {'xmin': 0, 'ymin': 0, 'xmax': 8, 'ymax': 8}, lambda p: p[1]-100., lambda *_: None)
        eng.enable_retention(1, tmp_path)
        st = ExperimentStore(tmp_path, eng.run_id, {})
        eng.observe_acquisition(GREEN, 0.0, 1, [])
        eng.observe_acquisition(YELLOW, 1/FPS, 2, [car(1, 90., speed=2., in_region=True)])
        eng.observe_acquisition(GREEN, 2/FPS, 3, [car(1, 110., speed=2., in_region=False)])
        eng.observe_acquisition(GREEN, 3/FPS, 4, [])
        c2_live = next(c for c in eng.snapshot()['comparisons'] if c['cycle_id'] == 2)
        assert c2_live['n_served'] == expect_served
        s1, tok1 = eng.prepare_commit(); st.persist(s1); eng.acknowledge(tok1)
        c2_evicted = next(c for c in eng.snapshot()['comparisons'] if c['cycle_id'] == 2)
        assert c2_evicted['n_served'] == expect_served
        eng.close()
        s2, tok2 = eng.prepare_commit(); st.persist(s2); eng.acknowledge(tok2)
        c2_arch = next(c for c in st.read()['comparisons'] if c['cycle_id'] == 2)
        assert c2_arch['n_served'] == expect_served
        if exit_sec == 0.0:
            # Simulate reopening a pre-upgrade archive where exit_persistence_sec was absent from policy.
            ev_path = st.directory/'evidence.json'
            ev_data = json.loads(ev_path.read_text())
            ev_data['state']['policy'].pop('exit_persistence_sec', None)
            ev_path.write_text(json.dumps(ev_data))
            reopened = ExperimentStore.open(st.directory)
            c2_legacy = next(c for c in reopened.read()['comparisons'] if c['cycle_id'] == 2)
            assert c2_legacy['n_served'] == 0


def test_delayed_vehicle_eviction_does_not_leak_archived_crossing_cycle_population(tmp_path):
    """Sol S8 Round 4 & 5: when an open A/B episode starts on the crossing frame (so the
    final interval straddles the retention cutoff and keeps old cycle membership) and delays
    vehicle eviction until a separate acknowledgment AFTER the crossing cycle has been
    archived, retention acknowledgment must not recreate cycle_population for the archived
    cycle, while still preserving upcoming RED-cycle membership."""
    from dataclasses import replace
    from types import SimpleNamespace
    from src.common.types import TrafficSnapshot
    from src.experiment.wait_savings import SchedulePolicy
    from test_vehicle_identity import detection

    roi = {'xmin': 0, 'ymin': 0, 'xmax': 16, 'ymax': 16}
    book = ObservationEngine('paired-s8', RunPolicy(debounce_frames=1, stop_persistence_sec=0),
                             roi, lambda p: p[1]-45.,
                             lambda *_: SimpleNamespace(green_seconds=10, failsafe_active=False),
                             identity_policy=IdentityPolicy(mode='persistent_v1', confirmation_observations=2,
                                                            confirmation_sec=.1, exit_persistence_sec=0),
                             schedule_policy=SchedulePolicy(enabled=True, unchanged_clearance=True),
                             system_roi=roi)
    book.enable_retention(1, tmp_path)
    store = ExperimentStore(tmp_path, 'paired-s8', {})
    frames_rgb = {
        'RED': np.full((16, 16, 3), (0, 0, 255), np.uint8),
        'GREEN': np.full((16, 16, 3), (0, 255, 0), np.uint8),
        'YELLOW': np.full((16, 16, 3), (0, 255, 255), np.uint8),
    }

    def analyze():
        return TrafficSnapshot(v_w_pcu=2, occupancy_pct=10, queue_meters=4), {'inference_calls': 1}

    seq = 0
    def step(t, a, b, ev=()):
        nonlocal seq
        seq += 1
        book.observe_system_frame(t)
        book.observe_a(frames_rgb[a], t, seq, analyze, {'timestamp': t, 'sequence': seq})
        book.observe_acquisition(frames_rgb[b], t, seq, (), identity_evidence=ev)

    # Cycle 1 GREEN starts at t=0.0; vehicle 101 builds confirmation at t=0.0..0.1 behind line
    # and crosses at t=0.2 on the EXACT frame A turns GREEN (opening episode with protected_since=0.2).
    v101_pre = replace(detection(101), point=(50., 40.), ground=(5., 4.))
    v101_post = replace(detection(101), point=(50., 50.), ground=(5., 5.))
    step(0.0, 'RED', 'GREEN', (v101_pre,))
    step(0.1, 'RED', 'GREEN', (v101_pre,))
    step(0.2, 'GREEN', 'GREEN', (v101_post,))
    book.schedule.await_idle()
    vid_101 = next(iter(book.waits.records))
    row_101 = book.waits.records[vid_101]
    assert row_101['crossed_stop_line'] and row_101['crossing_cycle'] == 1 and 1 in row_101['cycles']

    # End B Cycle 1 and complete B Cycle 2 while A's episode stays open (protecting v101 from eviction).
    step(0.3, 'GREEN', 'YELLOW')
    step(0.4, 'GREEN', 'RED')
    step(0.5, 'GREEN', 'GREEN')   # B Cycle 2 starts
    step(0.6, 'GREEN', 'YELLOW')  # B Cycle 2 ends -> Cycle 1 is now outside budget=1
    step(0.7, 'GREEN', 'RED')

    # Commit 1: Cycle 1 is archived and dropped while v101 remains held in memory with cycle 1 still in row['cycles']!
    state1, token1 = book.prepare_commit(); store.persist(state1); book.acknowledge(token1)
    assert 1 not in {c['cycle_id'] for c in book.phase.cycles}
    assert vid_101 in book.waits.records
    assert 1 in book.waits.records[vid_101]['cycles']
    assert 1 not in book.retention.cycle_population

    # Close A's episode so v101 is evicted in a SEPARATE acknowledgment AFTER Cycle 1 archival.
    book.invalidate_a('a_observation_gap', 0.8)
    step(0.9, 'RED', 'RED')

    # Commit 2 (separate acknowledgment AFTER Cycle 1 archival): v101 evicts, and Cycle 1 must NOT be recreated!
    state2, token2 = book.prepare_commit(); store.persist(state2); book.acknowledge(token2)
    assert vid_101 not in book.waits.records
    assert 1 not in book.retention.cycle_population

    # Also verify upcoming RED-cycle membership (cycle_id == phase.cycle_id + 1) IS preserved on eviction.
    upcoming_cycle = book.phase.cycle_id+1
    v202 = replace(detection(202), point=(150., 40.), ground=(15., 4.))
    step(1.0, 'RED', 'RED', (v202,))
    step(1.1, 'RED', 'RED', (v202,))
    step(1.2, 'RED', 'RED', (v202,))
    step(1.3, 'RED', 'RED', (replace(v202, in_region=False),))
    state3, token3 = book.prepare_commit(); store.persist(state3); book.acknowledge(token3)
    assert upcoming_cycle in book.retention.cycle_population
    assert book.retention.cycle_population[upcoming_cycle]['count'] == 1
    book.close()


def test_crossing_at_exact_max_gap_and_sub_tolerance_exit_persistence():
    """Sol Round 29:
    1. A genuine pending crossing confirmation at dt == max_gap_sec (e.g. 0.4 - 0.3 = 0.1 + 2.8e-17)
       must confirm SERVED in-ROI identically to outside-ROI rather than dropping pending_crossing.
    2. A tiny valid exit_persistence_sec (e.g. 1e-9) must enter EXIT_PENDING on the first outside
       sighting and never confirm EXITED on zero elapsed time or split the return into :segment-1."""
    for exit_sec in (0.0, 0.3):
        waits = VehicleWaits(
            'gap-edge',
            RunPolicy(max_gap_sec=0.1, fragment_continuation=True, exit_persistence_sec=exit_sec),
            side=lambda p: 270.-p[1],
        )
        for seq, (t, y, h) in enumerate([(0.0, 280., 60.), (0.1, 280., 60.), (0.2, 280., 60.),
                                         (0.3, 265., 45.), (0.4, 253., 45.)], start=1):
            waits.observe([car(1, y, x=150., height=h, speed=0.)], 'RED', t, seq, 1, unassigned=())
            if seq == 4:
                assert not only_row(waits)['crossed_stop_line']
        row = only_row(waits)
        assert row['crossed_stop_line'] and row['tracking_state'] == 'SERVED'

    waits_tiny = VehicleWaits(
        'tiny-exit',
        RunPolicy(exit_persistence_sec=1e-9),
        side=lambda p: -1.,
    )
    waits_tiny.observe([car(1, 200., x=150., height=60., speed=0., in_region=True)], 'RED', 0.0, 1, 1)
    waits_tiny.observe([car(1, 200., x=150., height=60., speed=0., in_region=False)], 'RED', 1/FPS, 2, 1)
    row_tiny = only_row(waits_tiny)
    assert not row_tiny['completed'] and row_tiny['tracking_state'] == 'EXIT_PENDING'
    waits_tiny.observe([car(1, 200., x=150., height=60., speed=0., in_region=True)], 'RED', 2/FPS, 3, 1)
    assert len(waits_tiny.records) == 1
    assert not only_row(waits_tiny)['completed'] and only_row(waits_tiny)['tracking_state'] != 'EXITED'
    # Two consecutive outside sightings spanning >= 1e-9 DO confirm EXITED at the first outside timestamp.
    waits_tiny.observe([car(1, 200., x=150., height=60., speed=0., in_region=False)], 'RED', 3/FPS, 4, 1)
    assert only_row(waits_tiny)['tracking_state'] == 'EXIT_PENDING'
    waits_tiny.observe([car(1, 200., x=150., height=60., speed=0., in_region=False)], 'RED', 4/FPS, 5, 1)
    assert only_row(waits_tiny)['completed'] and only_row(waits_tiny)['tracking_state'] == 'EXITED'
    assert only_row(waits_tiny)['completed_at'] == pytest.approx(3/FPS)

    # Sol Round 30:
    # 1. Exact-gap crossing through ObservationEngine preserves a single GREEN cycle,
    #    crossing_cycle=1, n_served=1, and full red wait at dt == max_gap_sec (0.4 - 0.3).
    eng = ObservationEngine(
        'gap-engine',
        RunPolicy(max_gap_sec=0.1, fragment_continuation=True, exit_persistence_sec=0.3),
        {'xmin': 0, 'ymin': 0, 'xmax': 8, 'ymax': 8},
        lambda p: 270.-p[1],
        lambda *_: None,
    )
    for seq, (t, y, h) in enumerate([(0.0, 280., 60.), (0.1, 280., 60.), (0.2, 280., 60.),
                                     (0.3, 265., 45.), (0.4, 253., 45.)], start=1):
        eng.observe_acquisition(GREEN, t, seq, [car(1, y, x=150., height=h, speed=0.)], unassigned_boxes=())
    for seq, t in enumerate([0.45, 0.50, 0.55], start=6):
        eng.observe_acquisition(GREEN, t, seq, [], unassigned_boxes=())
    eng_row = next(iter(eng.waits.records.values()))
    assert eng_row['tracking_state'] == 'SERVED' and eng_row['crossing_cycle'] == 1
    comps = eng.snapshot()['comparisons']
    assert len(comps) == 1 and comps[0]['n_served'] == 1

    waits_red = VehicleWaits(
        'gap-red',
        RunPolicy(max_gap_sec=0.1, stop_persistence_sec=0, fragment_continuation=True, exit_persistence_sec=0.3),
        side=lambda p: 270.-p[1],
    )
    for seq, t in enumerate([0.2, 0.3, 0.4], start=1):
        waits_red.observe([car(1, 280., x=150., height=60., speed=0.)], 'RED', t, seq, 1, unassigned=())
    red_row = only_row(waits_red)
    assert red_row['red_wait_sec'] == pytest.approx(0.2, abs=1e-9)
    assert 'observation_gap' not in red_row['quality_flags']

    # 2. Small configured max_gap_sec (1e-9) must not allow handoffs exceeding it (dt=1.5e-9).
    waits_sm = VehicleWaits('small-gap', RunPolicy(max_gap_sec=1e-9, fragment_continuation=True), side=lambda p: -1.)
    for seq, t in enumerate([0.0, 5e-10, 1e-9], start=1):
        waits_sm.observe([car(1, 200., x=150., height=60., speed=0.)], 'RED', t, seq, 1, unassigned=())
    waits_sm.observe([car(2, 200., x=150., height=60., speed=0.)], 'RED', 2.5e-9, 4, 1, unassigned=())
    assert len(waits_sm.records) == 2 and not any(r.get('identity_links') for r in waits_sm.records.values())

    # Sol Round 32: exact association_gap_sec=0.1 at 10 fps (0.30000000000000004 - 0.2) must not
    # complete an EXIT_PENDING visit as LOST or split its return at 0.4 into :segment-1.
    waits_assoc = VehicleWaits(
        'assoc-edge',
        RunPolicy(max_gap_sec=0.1, association_gap_sec=0.1, exit_persistence_sec=0.3),
        side=lambda p: -1.,
    )
    waits_assoc.observe([car(1, 200., x=150., height=60., speed=0., in_region=True)], 'RED', 0.0, 1, 1)
    waits_assoc.observe([car(1, 200., x=150., height=60., speed=0., in_region=True)], 'RED', 0.1, 2, 1)
    waits_assoc.observe([car(1, 200., x=150., height=60., speed=0., in_region=False)], 'RED', 0.2, 3, 1)
    waits_assoc.observe([], 'RED', 0.30000000000000004, 4, 1)
    assert not only_row(waits_assoc)['completed'] and only_row(waits_assoc)['tracking_state'] == 'LOST'
    waits_assoc.observe([car(1, 200., x=150., height=60., speed=0., in_region=True)], 'RED', 0.4, 5, 1)
    assert len(waits_assoc.records) == 1 and not only_row(waits_assoc)['completed']
