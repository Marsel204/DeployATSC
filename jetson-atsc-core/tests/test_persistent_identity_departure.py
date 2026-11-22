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


def test_persistent_exit_pending_preserves_genuine_stop_line_crossings():
    """Sol S1: VehicleWaits must never consume PersistentVehicleWaits's own exit_candidate
    when native exit_persistence_sec is off (or overridden to 0 in persistent_v1)."""
    for run_exit_sec in (0.0, 0.5):
        # Case A: vehicle steps outside ROI behind the line (persistent exit_pending=True),
        # then crosses the stop line on the very next consecutive frame while still exit_pending.
        book_a=ObservationEngine('dep-a',RunPolicy(debounce_frames=1,stop_persistence_sec=0,exit_persistence_sec=run_exit_sec),
            {'xmin':0,'ymin':0,'xmax':8,'ymax':8},lambda p:p[1]-45,lambda *_:None,
            identity_policy=IdentityPolicy(mode='persistent_v1',confirmation_observations=2,
                confirmation_sec=.1,exit_persistence_sec=.3))
        base=replace(detection(),point=(50,40),ground=(5,4))
        for seq in range(1,6):observe(book_a,seq,base)
        r6=observe(book_a,6,replace(base,point=(50,42),ground=(5,4.2),in_region=False))
        assert r6.vehicles[0].exit_pending
        observe(book_a,7,replace(base,point=(50,50),ground=(5,5),in_region=False))
        row_a,=book_a.waits.records.values()
        assert row_a['crossed_stop_line'] and row_a['tracking_state']=='SERVED'
        assert 'observation_gap' not in row_a['quality_flags']
        book_a.close()

        # Case B: vehicle stays outside behind the line through persistent exit_pending frames,
        # then crosses the stop line on the exact frame the persistent exit timer expires.
        book_b=ObservationEngine('dep-b',RunPolicy(debounce_frames=1,stop_persistence_sec=0,exit_persistence_sec=run_exit_sec),
            {'xmin':0,'ymin':0,'xmax':8,'ymax':8},lambda p:p[1]-45,lambda *_:None,
            identity_policy=IdentityPolicy(mode='persistent_v1',confirmation_observations=2,
                confirmation_sec=.1,exit_persistence_sec=.3))
        for seq in range(1,6):observe(book_b,seq,base)
        for seq in (6,7,8):
            res=observe(book_b,seq,replace(base,point=(50,42),ground=(5,4.2),in_region=False))
            assert res.vehicles[0].exit_pending
        observe(book_b,9,replace(base,point=(50,50),ground=(5,5),in_region=False))
        row_b,=book_b.waits.records.values()
        assert row_b['crossed_stop_line'] and row_b['tracking_state']=='SERVED'
        book_b.close()
