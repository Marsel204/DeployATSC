import io
import zipfile

import pytest

from src.experiment.export import ExperimentStore
from test_persistent_identity_pipeline import engine, tick
from test_vehicle_identity import detection


def replay(book, store=None):
    for seq in range(1, 131):
        t = (seq-1)/10
        tick(book, seq, t, () if 4 < t < 7 else (detection(4 if t <= 4 else 228),))
        if store is not None and seq % 3 == 0:
            state, token = book.prepare_commit()
            store.persist(state)
            book.acknowledge(token)
            book.acknowledge(token)
    book.close()


def test_retention_pending_gap_reconstruction_export_and_summaries_agree(tmp_path):
    oracle = engine()
    replay(oracle)
    book = engine()
    book.enable_retention(1, tmp_path)
    store = ExperimentStore(tmp_path, book.run_id, {'vehicle_identity': {'mode': 'persistent_v1'}})
    replay(book, store)
    state, token = book.prepare_commit()
    store.persist(state)
    book.acknowledge(token)
    read = store.read()
    expected = oracle.snapshot(evidence=True)
    assert len(read['vehicles']) == len(expected['vehicles']) == 1
    for key in ('red_wait_sec', 'estimated_red_wait_sec', 'display_red_wait_sec', 'unknown_sec'):
        assert read['vehicles'][0][key] == pytest.approx(expected['vehicles'][0][key])
    assert read['vehicles'][0]['estimated_red_wait_sec'] > 2.9
    assert read['summary']['best_estimate_mean_red_wait_sec'] == pytest.approx(expected['summary']['best_estimate_mean_red_wait_sec'])
    assert book.snapshot()['summary']['best_estimate_mean_red_wait_sec'] == pytest.approx(read['summary']['best_estimate_mean_red_wait_sec'])
    with zipfile.ZipFile(io.BytesIO(store.export())) as bundle:
        assert 'estimated_red_wait_sec' in bundle.read('vehicles.csv').decode()
    measured = store.quantiles()
    estimated = store.quantiles(metric='best_estimate')
    assert estimated['median_red_wait_sec'] > measured['median_red_wait_sec']
    assert estimated['metric'] == 'best_estimate'
    assert store.quantiles()['median_red_wait_sec'] == measured['median_red_wait_sec']


def test_unclean_recovery_cancels_pending_estimate_and_never_resumes_identity(tmp_path):
    book = engine()
    for seq in range(1, 31):
        tick(book, seq, (seq-1)/10, (detection(),) if seq < 20 else ())
    state = book.snapshot(evidence=True)
    assert state['vehicles'][0]['pending_estimated_red_wait_sec'] > 0
    store = ExperimentStore(tmp_path, book.run_id, {})
    store.persist(state)
    before = state['vehicles'][0]['red_wait_sec']
    store.recover()
    row = store.read()['vehicles'][0]
    assert row['pending_estimated_red_wait_sec'] == 0
    assert row['red_wait_sec'] == before
    assert row['display_red_wait_sec'] == row['red_wait_sec']+row['estimated_red_wait_sec']
    assert row['unresolved_coverage_sec']>=.9
    assert row['pending_gap'] is None
    book.close()
