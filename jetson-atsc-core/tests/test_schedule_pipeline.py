from types import SimpleNamespace
import io
import zipfile

import numpy as np
import pytest

from src.common.types import TrafficSnapshot
from src.experiment.contracts import RunPolicy
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.identity_contracts import IdentityPolicy
from src.experiment.wait_savings import SchedulePolicy
from test_vehicle_identity import detection


ROI = {'xmin': 0, 'ymin': 0, 'xmax': 16, 'ymax': 16}


def make_engine():
    return ObservationEngine('paired', RunPolicy(debounce_frames=1, stop_persistence_sec=0),
        ROI, lambda p: -1, lambda *_: SimpleNamespace(green_seconds=10, failsafe_active=False),
        identity_policy=IdentityPolicy(mode='persistent_v1'),
        schedule_policy=SchedulePolicy(enabled=True, unchanged_clearance=True), system_roi=ROI)


def frame(phase):
    return np.full((16, 16, 3), {'RED': (0, 0, 255), 'GREEN': (0, 255, 0), 'YELLOW': (0, 255, 255)}[phase], np.uint8)


def replay(book, store=None, cycles=3):
    calls = []
    def analyze():
        calls.append(1)
        return TrafficSnapshot(v_w_pcu=2, occupancy_pct=10, queue_meters=4), {'inference_calls': 1}
    for seq in range(1, cycles*250+1):
        timestamp = (seq-1)/10
        local = round(timestamp % 25, 6)
        a = 'RED' if local < 1 or local >= 21.5 else 'GREEN' if local < 21 else 'YELLOW'
        b = 'GREEN' if local < .2 or local >= 23 else 'YELLOW' if local < .4 else 'RED'
        book.observe_system_frame(timestamp)
        book.observe_a(frame(a), timestamp, seq, analyze, {'timestamp': timestamp, 'sequence': seq})
        book.observe_acquisition(frame(b), timestamp, seq, (),
            identity_evidence=() if 8 < local < 10 else (detection(4 if local < 8 else 228),))
        if a == 'GREEN' and local == 1:
            book.schedule.await_idle()
        if store is not None and seq % 9 == 0:
            state, token = book.prepare_commit()
            store.persist(state)
            book.acknowledge(token)
    assert calls == [1]*cycles
    book.close()


def test_paired_phases_savings_survive_eviction_and_export(tmp_path):
    oracle = make_engine()
    replay(oracle)
    book = make_engine()
    book.enable_retention(1, tmp_path)
    store = ExperimentStore(tmp_path, 'paired', {})
    replay(book, store)
    state, token = book.prepare_commit()
    store.persist(state)
    book.acknowledge(token)
    saved = store.read()
    expected = oracle.snapshot(evidence=True)
    assert len(saved['schedule_episodes']) == 3
    assert len(saved['vehicle_savings']) == len(expected['vehicle_savings']) == 3
    assert store.page('vehicle_savings')['total'] == 3
    for actual, reference in zip(sorted(saved['vehicle_savings'], key=lambda r: r['savings_id']),
            sorted(expected['vehicle_savings'], key=lambda r: r['savings_id'])):
        assert actual['available'] and actual['potential_saving_sec'] == pytest.approx(10)
        assert actual['estimated_schedule_red_stop_sec'] == pytest.approx(reference['estimated_schedule_red_stop_sec'])
    with zipfile.ZipFile(io.BytesIO(store.export())) as bundle:
        assert 'potential_saving_sec' in bundle.read('vehicle_savings.csv').decode()
        assert 'schedule_episodes.csv' in bundle.namelist()
    assert len(book.waits.records) <= 1


def test_a_green_without_fresh_b_red_cannot_claim_a_valid_pair():
    book = make_engine()
    book.observe_a(frame('RED'), 0, 1, lambda: None, {'timestamp': 0, 'sequence': 1})
    book.observe_a(frame('GREEN'), .1, 2, lambda: (TrafficSnapshot(queue_meters=1), {}), {'timestamp': .1, 'sequence': 2})
    book.schedule.await_idle()
    book.close()
    row = book.snapshot()['schedule_episodes'][0]
    assert row['b_episode_id'] is None
    assert row['status'] == 'unavailable'
