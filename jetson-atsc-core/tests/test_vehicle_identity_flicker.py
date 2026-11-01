"""Recorded counterexamples; short local windows, never the full evidence stream."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import json
import os

import pytest

from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy
from src.experiment.vehicle_identity import Entry, VehicleIdentityRegistry

CANDIDATE = os.environ.get('FLICKER_ASSOCIATION', 'delayed_trajectory_v3')

FIXTURES = Path(__file__).parent/'fixtures/flicker'


def load_case(name, association):
    data = json.loads((FIXTURES/f'{name}.json').read_text())
    # This switch runs precisely the same assertions on unmodified bc3c403,
    # before the opt-in policy field exists; no missing-API failures qualify.
    kw = {} if os.environ.get('STAGE1_BASELINE') else {'association': association}
    book = VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1', **kw), ground_trusted=False)
    for row in data['entries']:
        row['detection'] = IdentityDetection(**row['detection'])
        if row['trusted_detection'] is not None:
            row['trusted_detection'] = IdentityDetection(**row['trusted_detection'])
        for key, length in [('history',8),('image_history',8),('aliases',16)]:
            row[key] = deque((tuple(x) for x in row[key]), maxlen=length)
        if row['aliases']:row['aliases'] = deque(map(tuple,row['aliases']),maxlen=16)
        entry = Entry(**row)
        book.entries[entry.identity] = entry
    book.next_id = data['next_id']
    book.last_time = data['last_time']
    book.last_sequence = data['last_sequence']
    results = []
    for row in data['frames']:
        ds = [IdentityDetection(**d) for d in row['detections']]
        result = book.resolve(ds, row['timestamp'], row['sequence'])
        labels = [v for d,v in zip(ds,result.labels) if d.native_id==data['native']]
        results.append((row,result,labels))
    return data,book,results



def test_turn_reattaches_without_crediting_held_frames():
    association=CANDIDATE
    data,book,results = load_case('turn',association)
    labels = results[-1][2]
    assert any(v and v.logical_id==data['target'] for v in labels), 'V9 turn must reattach by f50'
    held = [r for row,r,labels in results if row['frame']>=45 and not any(labels)]
    assert held, 'Failed-gate evidence must be held, never credited immediately'
    original=next(e for e in data['entries'] if e['identity']==data['target'])['observations']
    credited=sum(bool(v and v.logical_id==data['target']) for row,r,labels in results for v in labels)
    assert book.entries[data['target']].observations==original+credited


@pytest.mark.parametrize('name',['switch','merged'])

def test_same_native_switch_and_merged_box_never_inherit_timer(name):
    association=CANDIDATE
    data,book,results = load_case(name,association)
    boundary = 51 if name=='switch' else 63
    assert all(not v or v.logical_id!=data['target'] for row,r,labels in results if row['frame']>=boundary for v in labels)


@pytest.mark.parametrize('name',['ghost359','ghost386','ghost403','ghost429'])

def test_stale_ghost_cannot_block_valid_consecutive_native(name):
    association=CANDIDATE
    data,book,results = load_case(name,association)
    assert any(v and v.logical_id==data['target'] for v in results[0][2]), 'Consecutive gated track must beat stale V28 geometry'



def test_cheaper_recent_competitor_stays_ambiguous():
    association=CANDIDATE
    data,book,results = load_case('ambiguous',association)
    assert not any(v and v.logical_id==data['target'] for v in results[0][2])


from test_vehicle_identity_moving_queue import CASES

@pytest.mark.parametrize('case',CASES,ids=lambda c:c['baseline_id'])
def test_original_wrong_transfer_guards_for_every_arm(case):
    book=VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1',association=CANDIDATE),ground_trusted=False)
    for row in case['history']:
        book.resolve((IdentityDetection(**row['detection']),),row['timestamp'],row['sequence'])
    entry=min(book.entries.values(),key=lambda e:e.identity)
    assert entry.admitted
    assert book._cost(entry,IdentityDetection(**case['candidate']),case['timestamp']) is None


def test_candidate_provenance_and_source_reset():
    association=CANDIDATE
    from test_vehicle_identity import detection
    from src.experiment.contracts import RunPolicy
    from src.experiment.engine import ObservationEngine
    engine=ObservationEngine('candidate',RunPolicy(),{'xmin':0,'ymin':0,'xmax':100,'ymax':100},lambda p:-1,lambda *_:None,
        identity_policy=IdentityPolicy(mode='persistent_v1',association=association,confirmation_observations=2,confirmation_sec=.1))
    import numpy as np
    engine.observe_acquisition(np.zeros((10,10,3),np.uint8),0.,1,(),identity_evidence=[detection()])
    result=engine.observe_acquisition(np.zeros((10,10,3),np.uint8),.1,2,(),identity_evidence=[detection()])
    assert result.vehicles[0].association==association
    for sequence in range(3,8):
        engine.observe_acquisition(np.zeros((10,10,3),np.uint8),(sequence-1)/10,sequence,(),identity_evidence=[detection()])
    rows=engine.waits.rows(include_intervals=True)
    assert rows and rows[0]['association_policy']==association
    intervals=[i for row in rows for i in row['intervals']]
    assert intervals
    assert all(i['identity_association']==association for i in intervals)
    book=engine.identities
    book.holds[1]={'samples':deque(maxlen=book.window)}
    book.invalidate('source_boundary')
    assert not book.holds and not book.entries and not book.decisions
    assert book.generation==1


@pytest.mark.parametrize('version',['p2','p1_k4','p1_k8','p3','p4','p2_p3_p4','p2_p1_k4_p4','p2_p1_k8_p4','unreviewed'])
def test_unknown_or_removed_candidate_version_rejected(version):
    with pytest.raises(ValueError,match='association'):
        IdentityPolicy(mode='persistent_v1',association=version)


def candidate(**kwargs):
    return VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1',association=CANDIDATE,
        confirmation_observations=2,confirmation_sec=.1,**kwargs),ground_trusted=False)


@pytest.mark.parametrize('native',['same','new','epoch'])
@pytest.mark.parametrize('reliable',[False,True])
def test_normal_continuation_evidence_requirements(native,reliable):
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book)
    item=replace(detection(raw=4 if native!='new' else 228),native_epoch=1 if native=='epoch' else 0,appearance_reliable=reliable)
    result=book.resolve((item,),.2,3)
    assert any(v.logical_id==identity for v in result.vehicles)==(native=='same' or reliable)
    assert not book.holds


@pytest.mark.parametrize('time,sequence',[(.2,2),(.2,1),(.1,3)])
def test_duplicate_packets_preserve_candidate_state(time,sequence):
    from test_vehicle_identity import admit,detection
    import copy
    book=candidate();admit(book);before=copy.deepcopy(vars(book))
    assert book.resolve((detection(),),time,sequence).duplicate
    assert vars(book)==before


@pytest.mark.parametrize('time',[float('nan'),float('inf'),-float('inf')])
def test_candidate_rejects_nonfinite_timestamps(time):
    from test_vehicle_identity import detection
    book=candidate()
    with pytest.raises(ValueError,match='finite'):book.resolve((detection(),),time,1)
    assert not book.entries and book.last_sequence is None


@pytest.mark.parametrize('reason',['camera_displacement','source_boundary','capture_gap'])
def test_source_invalidation_clears_pending_trajectory(reason):
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book);entry=book.entries[identity]
    assert book._hold(entry,detection(),(detection(),),.2,3)=='hold'
    book.invalidate(reason)
    assert not book.entries and not book.holds and not book.decisions
    assert book.generation==1 and book.last_time is None


@pytest.mark.parametrize('crossed',[False,True])
@pytest.mark.parametrize('rejected',[False,True])
def test_retirement_discards_active_or_rejected_hold(crossed,rejected):
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book);entry=book.entries[identity]
    book._hold(entry,detection(),(),.2,3);book.holds[identity]['rejected']=rejected
    book.retire(entry.generation,identity,crossed=crossed)
    assert identity not in book.holds
    assert entry.lifecycle==('SERVED' if crossed else 'EXITED')
    assert not entry.image_history and entry.image_velocity is None


@pytest.mark.parametrize('contradiction',['width','height','appearance','reversal','rival','alias'])
def test_hold_rejects_independent_switch_evidence(contradiction):
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book);entry=book.entries[identity];item=detection();rivals=()
    if contradiction=='width':item=replace(item,bbox=(30,10,70,50))
    if contradiction=='height':item=replace(item,bbox=(40,0,60,60))
    if contradiction=='appearance':item=replace(item,appearance=(0.,1.))
    if contradiction=='reversal':
        entry.image_history=deque([(0.,(40,50)),(.1,(50,50))],maxlen=8);item=detection(x=40)
    if contradiction=='rival':item=detection(x=70);rivals=(detection(raw=99),)
    if contradiction=='alias':
        assert book._hold(entry,item,(item,),.15,3)=='hold'
        item=detection(raw=99)
    assert book._hold(entry,item,(item,*rivals),.2,4)=='reject'
    assert book.holds[identity]['rejected'] and entry.observations==2


@pytest.mark.parametrize('sequence_step',[1,2])
def test_unreliable_hold_expires_by_sequence_slots(sequence_step):
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book);entry=book.entries[identity]
    item=replace(detection(),appearance_reliable=False)
    outcomes=[]
    for seq in range(3,12,sequence_step):
        decision=book._hold(entry,item,(item,),seq/10,seq);outcomes.append(decision)
        if decision=='reject':break
    assert outcomes[-1]=='reject' and len(book.holds[identity]['samples'])<=8
    assert entry.observations==2


def test_four_strong_held_points_can_release_early():
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book);entry=book.entries[identity];item=detection()
    outcomes=[book._hold(entry,item,(item,),seq/10,seq) for seq in range(3,7)]
    assert outcomes==['hold','hold','hold','accept']
    assert entry.observations==2


def test_same_native_failure_beyond_one_second_does_not_start_hold():
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book)
    result=book.resolve((detection(x=500),),1.100001,20)
    assert not book.holds and not book.decisions
    assert all(v.logical_id!=identity for v in result.vehicles)


@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('ratio',[1.249,1.251])
def test_shape_contradiction_keeps_registered_threshold(axis,ratio):
    from test_vehicle_identity import detection
    from src.experiment.vehicle_identity_candidates import shape_jump
    old=detection();box=list(old.bbox);box[axis+2]=box[axis]+(box[axis+2]-box[axis])*ratio
    assert shape_jump(old,replace(old,bbox=tuple(box)))==(ratio>1.25)


def test_successful_normal_frames_never_run_hold_logic():
    from test_vehicle_identity import admit,detection
    book=candidate();identity=admit(book)
    def forbidden(*args,**kwargs):raise AssertionError('Normal frame evaluated a hold')
    book._hold=forbidden
    for seq in range(3,20):
        result=book.resolve((detection(),),(seq-1)/10,seq)
        assert result.vehicles[0].logical_id==identity
    assert not book.holds and not book.decisions
