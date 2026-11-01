"""Alias/lineage hygiene, including the complete unseeded recorded upstream."""
from dataclasses import replace
from pathlib import Path
import json
import os

import pytest

from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy
from src.experiment.vehicle_identity import Entry, VehicleIdentityRegistry
from test_vehicle_identity import admit, detection

VERSION = os.environ.get('STAGE2C_ASSOCIATION', 'delayed_trajectory_v5')


def registry():
    return VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1', association=VERSION,
                                   confirmation_observations=2, confirmation_sec=.1),
                                   ground_trusted=False)


@pytest.fixture(scope='module')
def upstream():
    data = json.loads((Path(__file__).parent /
        'fixtures/flicker_stage2c/native87_to232_lineage.json').read_text())
    book = VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1', association=VERSION),
                                   ground_trusted=False)
    frames = {}
    calls = []
    original = book._hold

    def traced(entry, item, items, timestamp, sequence):
        outcome = original(entry, item, items, timestamp, sequence)
        calls.append((sequence-1, entry.identity, item.native_id, outcome))
        return outcome

    book._hold = traced
    for row in data['frames']:
        items = [IdentityDetection(**d) for d in row['detections']]
        result = book.resolve(items, row['timestamp'], row['sequence'])
        frames[row['frame']] = {
            d.native_id: (next((e.identity for e in book.entries.values() if e.detection is d), None),
                          result.labels[j]) for j, d in enumerate(items)}
    return frames, calls


def test_full_upstream_native87_to232_must_not_link_f75_to_f76(upstream):
    frames, calls = upstream
    assert any(f == 51 and native == 87 and out == 'reject' for f, _, native, out in calls)
    assert frames[74][232][1] is not None
    before = frames[75][232][0]
    assert before is not None
    assert frames[76][232][0] != before, 'Full upstream lineage must refuse the jump onto the car ahead'


def test_new_native_gets_fresh_hold_after_real_native87_rejection(upstream):
    _, calls = upstream
    earlier = {identity for f, identity, native, out in calls
               if native == 87 and out == 'reject' and f < 75}
    assert any(f == 75 and identity in earlier and native == 232 and out == 'reject'
               for f, identity, native, out in calls), 'An unrelated rejected hold must not skip native232 evaluation'


@pytest.mark.parametrize('new_epoch', [False, True])
def test_rejected_alias_does_not_poison_another_alias_hold(new_epoch):
    book = registry()
    ident = admit(book)
    entry = book.entries[ident]
    rejected = replace(detection(), appearance=(0., 1.))
    assert book._hold(entry, rejected, (rejected,), .2, 3) == 'reject'
    current = replace(detection(), native_epoch=1) if new_epoch else detection(raw=99)
    assert book._hold(entry, current, (current,), .3, 4) == 'hold'


def test_supported_update_starts_a_new_hold_lineage():
    book = registry()
    ident = admit(book)
    entry = book.entries[ident]
    rejected = replace(detection(), appearance=(0., 1.))
    assert book._hold(entry, rejected, (rejected,), .2, 3) == 'reject'
    book.resolve((detection(),), .3, 4)
    assert book._hold(entry, detection(), (), .4, 5) == 'hold'


def test_active_alias_switch_rejection_also_guards_the_proposed_alias():
    book = registry()
    ident = admit(book)
    entry = book.entries[ident]
    assert book._hold(entry, detection(), (), .2, 3) == 'hold'
    assert book._hold(entry, detection(raw=99), (), .3, 4) == 'reject'
    book._update(entry, detection(raw=123), .4, 5)
    tentative = Entry(2, 0, detection(raw=99), .4, .4, 5, observations=1)
    tentative.aliases.append((0, 99))
    book.last_sequence = 6
    assert book._cost(tentative, detection(raw=99, x=70), .5) is None


def test_rejected_native_history_uses_the_existing_alias_memory_bound():
    book = registry()
    ident = admit(book)
    entry = book.entries[ident]
    for raw in range(entry.aliases.maxlen * 2):
        item = replace(detection(raw=raw), appearance=(0., 1.))
        assert book._hold(entry, item, (item,), .2+raw/10, 3+raw) == 'reject'
    assert len(getattr(book, 'rejected_aliases', {}).get(ident, ())) <= entry.aliases.maxlen


@pytest.mark.parametrize('native', [4, 99])
def test_tentative_restart_guard_remembers_each_rejected_alias(native):
    book = registry()
    ident = admit(book)
    entry = book.entries[ident]
    for raw, timestamp, sequence in [(4, .2, 3), (99, .3, 4)]:
        item = replace(detection(raw=raw), appearance=(0., 1.))
        assert book._hold(entry, item, (item,), timestamp, sequence) == 'reject'
    # A different alias can acquire ordinary support without erasing either veto.
    book._update(entry, detection(raw=123), .4, 5)
    tentative = Entry(2, 0, detection(raw=native), .4, .4, 5, observations=1)
    tentative.aliases.append((0, native))
    book.last_sequence = 6
    assert book._cost(tentative, detection(raw=native, x=70), .5) is None


@pytest.mark.parametrize('kind', ['epoch', 'identity', 'expired'])
def test_rejection_does_not_tighten_unrelated_or_expired_tentatives(kind):
    book = registry()
    ident = admit(book)
    entry = book.entries[ident]
    item = replace(detection(), appearance=(0., 1.))
    assert book._hold(entry, item, (item,), .2, 3) == 'reject'
    native = 77 if kind == 'identity' else 4
    epoch = 1 if kind == 'epoch' else 0
    timestamp = 1.2 if kind == 'expired' else .3
    old = replace(detection(raw=native), native_epoch=epoch)
    tentative = Entry(2, 0, old, timestamp-.1, timestamp-.1, 3, observations=1)
    tentative.aliases.append((epoch, native))
    book.last_sequence = 4
    assert book._cost(tentative, replace(old, point=(70., 50.)), timestamp) is not None


@pytest.mark.parametrize('reason', ['retire', 'invalidate', 'ttl'])
def test_rejected_alias_history_clears_with_entry_lifetime(reason):
    book = registry()
    ident = admit(book)
    item = replace(detection(), appearance=(0., 1.))
    book._hold(book.entries[ident], item, (item,), .2, 3)
    if reason == 'retire':
        book.retire(0, ident)
    elif reason == 'invalidate':
        book.invalidate('source_reset')
    else:
        book.resolve((), 11., 4)
    # Behavioural check: a new tentative no longer inherits the retired veto.
    tentative = Entry(2, book.generation, detection(), .2, .2, 3, observations=1)
    tentative.aliases.append((0, 4))
    book.last_sequence = 4
    assert book._cost(tentative, detection(x=70), .3) is not None
