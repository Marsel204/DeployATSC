"""Frozen full-prefix safety and separately selected coverage acceptance checks."""
from dataclasses import replace
from pathlib import Path
import json
import os

import numpy as np
import pytest

from src.experiment.calibration import CameraCalibration
from src.experiment.contracts import RunPolicy
from src.experiment.engine import ObservationEngine
from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy
from src.experiment.vehicle_identity import VehicleIdentityRegistry
from test_vehicle_identity import detection

VERSION = os.environ.get('STAGE2D_ASSOCIATION', 'delayed_trajectory_v6')
ROOT = Path(__file__).resolve().parents[3]


def registry(**kwargs):
    return VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1', association=VERSION, **kwargs),
                                   ground_trusted=False)


def replay_prefix(clip):
    data = json.loads((Path(__file__).parent / 'fixtures/flicker_stage2d' /
                       f'{clip}_complete_prefix.json').read_text())
    book = registry()
    engine = None
    if clip == 'queue':
        payload = json.loads((ROOT / 'reviews/youtube-stationary-2026-11-01/interactive/junction-22s-preset.json').read_text())
        camera = CameraCalibration(payload['acquisition'])
        engine = ObservationEngine('stage2', RunPolicy(**payload['policy']), camera.roi, camera.side,
            lambda *_: None, identity_policy=book.policy,
            scale_evidence={'acquisition': {'state': 'unverified'}})
        book = engine.identities
    frames = {}
    bids = []
    original = book._cost

    def traced(entry, item, timestamp):
        cost = original(entry, item, timestamp)
        if not entry.admitted and cost is not None:
            bids.append(dict(frame=book.last_sequence - 1, owner=entry.identity,
                before=entry.detection.native_id, native=item.native_id, cost=cost))
        return cost

    book._cost = traced
    for row in data['frames']:
        items = [IdentityDetection(**value) for value in row['detections']]
        result = (engine.observe_acquisition(np.zeros((10, 10, 3), np.uint8), row['timestamp'],
            row['sequence'], (), identity_evidence=items) if engine else
            book.resolve(items, row['timestamp'], row['sequence']))
        lookup = {id(entry.detection): entry for entry in book.entries.values()}
        values = []
        for item, label in zip(items, result.labels):
            entry = lookup.get(id(item))
            values.append(dict(native=item.native_id, owner=entry.identity if entry else None,
                admitted=label is not None, observations=entry.observations if entry else None,
                born=entry.born if entry else None))
        frames[row['frame']] = values
    return clip, frames, bids


@pytest.fixture(scope='module')
def queue():
    return replay_prefix('queue')[1:]


@pytest.fixture(scope='module')
def semarang():
    return replay_prefix('semarang')[1:]


def at(frames, frame, native):
    matches = [value for value in frames[frame] if value['native'] == native]
    assert len(matches) == 1
    return matches[0]


def test_queue_f56_illegal_rival(queue):
    frames, bids = queue
    suv = at(frames, 52, 87)['owner']
    assert suv is not None
    assert not any(value['frame'] == 56 and value['owner'] == suv and value['native'] == 232
                   for value in bids), 'SUV segment must not enter EV competition at f56'


def test_queue_suv_ev_owners_and_fresh_admission(queue):
    frames, _ = queue
    assert at(frames, 52, 87)['owner'] != at(frames, 57, 232)['owner'], 'f52 SUV evidence must not admit f57 EV'
    assert {87, 232} <= {value['native'] for value in frames[57]}
    assert at(frames, 57, 87)['owner'] != at(frames, 57, 232)['owner']


def test_queue_ev_suv_provisional_owners(queue):
    frames, bids = queue
    owner = at(frames, 75, 232)['owner']
    assert owner is not None
    assert owner != at(frames, 81, 300)['owner'], 'f75 EV tentative must not steal f81 SUV'
    assert not any(value['frame'] == 81 and value['owner'] == owner and value['native'] == 300
                   for value in bids)


def test_queue_f75_f76_rejection(queue):
    frames, _ = queue
    assert at(frames, 75, 232)['owner'] != at(frames, 76, 232)['owner']


def test_new_wrong_f141_f185_white_suv_dark_sedan(queue):
    frames, _ = queue
    before = at(frames, 141, 300)
    assert before['owner'] is not None
    assert before['owner'] != frames[185][24]['owner'], 'Audited white SUV must not link to dark sedan'


@pytest.mark.parametrize('replacement', ['native', 'epoch', 'missing', 'unstable'])
def test_confirmation_state_belongs_to_one_stable_segment(replacement):
    book = registry()
    original = detection()
    book.resolve((original,), 0., 1)
    book.resolve((original,), .1, 2)
    prior = next(iter(book.entries.values()))
    assert prior.observations == 2 and not prior.admitted
    state = (prior.born, prior.observations, prior.first_native_reference, prior.appearance,
             list(prior.history), list(prior.image_history), prior.detection)
    new = {'native': detection(raw=99), 'epoch': replace(original, native_epoch=1),
           'missing': replace(original, native_id=None),
           'unstable': replace(original, native_stable=False)}[replacement]
    result = book.resolve((new,), .2, 3)
    assert not result.vehicles, 'Another alias must not inherit two observations and .2s age'
    fresh = next(entry for entry in book.entries.values() if entry.detection is new)
    assert fresh.identity != prior.identity
    assert (fresh.born, fresh.observations) == (.2, 1)
    assert len(fresh.image_history) == 1 and fresh.image_velocity is None
    assert (prior.born, prior.observations, prior.first_native_reference, prior.appearance,
            list(prior.history), list(prior.image_history), prior.detection) == state


@pytest.mark.parametrize('unstable', [False, True])
def test_anonymous_observations_never_confirm_or_donate_history(unstable):
    book = registry(capacity=4)
    item = replace(detection(), native_stable=not unstable, native_id=4 if unstable else None)
    for sequence in range(1, 6):
        assert not book.resolve((item,), (sequence-1)*.1, sequence).vehicles
        assert len(book.entries) <= 4
        assert all(entry.observations == 1 and not entry.admitted for entry in book.entries.values())
    book.resolve((), 2., 6)
    assert not book.entries
    stable = detection(raw=88)
    assert not book.resolve((stable,), 2.1, 7).vehicles
    fresh = next(iter(book.entries.values()))
    assert fresh.born == 2.1 and fresh.observations == 1 and list(fresh.aliases) == [(0, 88)]
    book.invalidate('source_reset')
    assert not book.entries and book.generation == 1
    assert not book.resolve((stable,), 2.2, 8).vehicles


def test_admitted_unstable_association_remains_available():
    book = registry()
    for sequence in range(1, 4):
        result = book.resolve((detection(),), (sequence-1)*.1, sequence)
    identity = result.vehicles[0].logical_id
    result = book.resolve((replace(detection(), native_stable=False),), .3, 4)
    assert result.vehicles[0].logical_id == identity


def test_stable_replacement_is_independently_born_with_anonymous_still_retained():
    book = registry()
    anonymous = replace(detection(), native_id=None)
    for sequence in range(1, 3):
        book.resolve((anonymous,), (sequence-1)*.1, sequence)
    stable = detection(raw=88, appearance=(.9, .1))
    result = book.resolve((stable,), .2, 3)
    assert not result.vehicles
    fresh = next(entry for entry in book.entries.values() if entry.detection is stable)
    assert (fresh.born, fresh.observations, fresh.first_native_reference) == (.2, 1, (0, 88))
    assert fresh.appearance == stable.appearance and len(fresh.image_history) == 1
    assert all(entry.observations == 1 for entry in book.entries.values())


def test_acceptance_queue_full_prefix_retains_f90_f91(queue):
    frames, _ = queue
    before = at(frames, 90, 300)
    after = at(frames, 91, 300)
    assert before['owner'] is not None
    assert after['owner'] == before['owner'] and after['admitted'], 'Required partial-box continuation f90->91'


def test_acceptance_semarang_retains_all_18_motorcycle_samples(semarang):
    frames, _ = semarang
    samples = [value for values in frames.values() for value in values if value['native'] == 63]
    assert len(samples) == 18
    assert all(value['admitted'] for value in samples), 'All18 native17->63 partial motorcycle samples required'
    owners = {value['owner'] for value in samples}
    prior = [value['owner'] for frame, values in frames.items() if frame < 183
             for value in values if value['native'] == 17 and value['admitted']]
    assert prior and owners == {prior[-1]}


def test_semarang_actual_merged_expansion_remains_blocked(semarang):
    frames, _ = semarang
    assert at(frames, 182, 13)['owner'] != at(frames, 183, 90)['owner']
