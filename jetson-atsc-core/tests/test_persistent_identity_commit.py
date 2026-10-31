import pytest

from src.experiment.export import ExperimentStore
from test_persistent_identity_pipeline import engine,tick
from test_vehicle_identity import detection


def test_commit_inflight_never_discards_a_later_coalesced_interval(tmp_path):
    book=engine();book.enable_retention(1,tmp_path)
    store=ExperimentStore(tmp_path,book.run_id,{})
    try:
        for seq in range(1,12):tick(book,seq,(seq-1)/10,(detection(),))
        state,token=book.prepare_commit()
        tick(book,12,1.1,(detection(),))
        store.persist(state);book.acknowledge(token)
        state,token=book.prepare_commit();store.persist(state);book.acknowledge(token)
        row=store.read()['vehicles'][0]
        observed=sum(i['end']-i['start'] for i in row['intervals'] if i['valid'] and i['stopped'] and i['phase']=='RED')
        assert observed==pytest.approx(row['red_wait_sec'])
    finally:book.close()
