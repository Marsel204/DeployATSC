from dataclasses import replace

import numpy as np

from src.experiment.contracts import RunPolicy
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.identity_contracts import IdentityPolicy
from test_vehicle_identity import detection


def make_book():
    return ObservationEngine('departure',RunPolicy(debounce_frames=1,stop_persistence_sec=0),
        {'xmin':0,'ymin':0,'xmax':8,'ymax':8},lambda p:p[1]-45,lambda *_:None,
        identity_policy=IdentityPolicy(mode='persistent_v1',confirmation_observations=2,
            confirmation_sec=.1,exit_persistence_sec=0))


def observe(book,seq,item):
    return book.observe_acquisition(np.full((8,8,3),(0,0,255),np.uint8),(seq-1)/10,seq,(),identity_evidence=(item,))


def test_crossed_identity_keeps_one_count_after_ordinary_archive_eviction(tmp_path):
    book=make_book();book.enable_retention(1,tmp_path)
    store=ExperimentStore(tmp_path,book.run_id,{})
    for seq in range(1,11):observe(book,seq,replace(detection(),point=(50,40),ground=(5,4)))
    for seq in range(11,21):
        observe(book,seq,replace(detection(),point=(50,50),ground=(5,5)))
        state,token=book.prepare_commit();store.persist(state);book.acknowledge(token)
    book.close()
    state,token=book.prepare_commit();store.persist(state);book.acknowledge(token)
    rows=store.read()['vehicles']
    assert len(rows)==1 and rows[0]['crossed_stop_line']


def test_departure_and_return_are_separate_visits_with_fresh_logical_id():
    book=make_book()
    first=replace(detection(),point=(50,40),ground=(5,4))
    for seq in range(1,11):result=observe(book,seq,first)
    identity=result.vehicles[0].logical_id
    observe(book,11,replace(first,in_region=False))
    for seq in range(12,16):result=observe(book,seq,replace(first,native_id=88))
    assert result.vehicles[0].logical_id!=identity
    assert len(book.waits.records)==2
    book.close()
