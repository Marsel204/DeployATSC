"""Frozen Stage 2b counterexamples, checked before implementing v4."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import json,os
import pytest
from src.experiment.identity_contracts import IdentityDetection,IdentityPolicy
from src.experiment.vehicle_identity import Entry,VehicleIdentityRegistry

VERSION=os.environ.get('STAGE2B_ASSOCIATION','delayed_trajectory_v4')

def book():
    return VehicleIdentityRegistry(IdentityPolicy(mode='persistent_v1',association=VERSION),ground_trusted=False)

def recorded(frame):
    data=json.loads((Path(__file__).parent/f'fixtures/flicker_stage2b/case{frame}.json').read_text())
    reg=book()
    for row in data['entries']:
        row['detection']=IdentityDetection(**row['detection'])
        if row['trusted_detection'] is not None:row['trusted_detection']=IdentityDetection(**row['trusted_detection'])
        for k,n in [('history',8),('image_history',8),('aliases',16)]:row[k]=deque(map(tuple,row[k]),maxlen=n)
        reg.entries[row['identity']]=Entry(**row)
    reg.next_id=data['next_id'];reg.last_time=data['last_time'];reg.last_sequence=data['last_sequence']
    for k,h in data['holds'].items():
        h['samples']=deque(h['samples'],maxlen=8);h['alias']=tuple(h['alias']);reg.holds[int(k)]=h
    return reg,data['frames']

def run_frame(reg,row):
    ds=[IdentityDetection(**d) for d in row['detections']]
    result=reg.resolve(ds,row['timestamp'],row['sequence'])
    identities={d.native_id:next((e.identity for e in reg.entries.values() if e.detection is d),None) for d in ds}
    return result,identities

def test_rejected_native_restart_must_not_link_f75_to_f76():
    reg,rows=recorded(75)
    _,before=run_frame(reg,rows[0]);assert before[232] is not None
    _,after=run_frame(reg,rows[1])
    assert after[232]!=before[232], 'Rejected native restart must not use a loose tentative gate to follow the car ahead'

def test_cross_native_merged_takeover_f361_must_be_refused():
    reg,rows=recorded(361);old=next(e.identity for e in reg.entries.values() if e.detection.native_id==101 and e.seen==12.)
    result,_=run_frame(reg,rows[0]);j=next(i for i,d in enumerate(rows[0]['detections']) if d['native_id']==1368)
    assert not result.labels[j] or result.labels[j].logical_id!=old, '2.22x wider merged detection must not inherit the small car timer'

def test_rejected_ghosts_allow_independent_f181_to_f182_continuation():
    reg,rows=recorded(169);outputs={r['frame']:run_frame(reg,r) for r in rows}
    result181,before=outputs[181];result182,after=outputs[182]
    assert before[648] is not None and after[648]==before[648], 'Rejected ghosts must not suppress the independent maroon car forever'
    assert any(v.detection.native_id==648 for v in result182.vehicles)
    assert after[648]!=39, 'Maroon car must not inherit the other sedan identity'

# Scope and thresholds use synthetic geometry, independently of physical labels.
def admitted():
    from test_vehicle_identity import detection
    reg=book();d=detection();ent=Entry(1,0,d,0.,0.,1,observations=10,admitted=True,appearance=d.appearance)
    ent.aliases.append((d.native_epoch,d.native_id));ent.image_history.extend([(-.1,d.point),(0.,d.point)])
    ent.image_velocity=(0.,0.);reg.entries[1]=ent;reg.next_id=2
    return reg,ent,d

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('ratio',[1.249,1.251])
@pytest.mark.parametrize('cross',[False,True])
def test_shape_guard_applies_only_to_cross_native(axis,ratio,cross):
    reg,ent,d=admitted();box=list(d.bbox);box[axis+2]=box[axis]+(box[axis+2]-box[axis])*ratio
    item=replace(d,bbox=tuple(box),native_id=99 if cross else d.native_id)
    result=reg.resolve((item,),.1,2)
    assert bool(result.labels[0] and result.labels[0].logical_id==1)==(not (cross and ratio>1.25))

@pytest.mark.parametrize('release',[False,True])
@pytest.mark.parametrize('fraction',[.699,.701])
def test_independent_containment_guard_boundary(release,fraction):
    reg,ent,d=admitted();item=d if release else replace(d,native_id=99)
    # Independent rival rectangle extends outside the proposal; fraction covered.
    x0,y0,x1,y1=d.bbox;w=x1-x0;h=y1-y0
    rival=replace(d,native_id=88,bbox=(x1-w*fraction,y0,x1+w*(1-fraction),y1),point=(x1+w*(.5-fraction),y1))
    if release:
        outcomes=[reg._hold(ent,item,(item,rival),seq/10,seq+1) for seq in range(1,5)]
        assert outcomes[-1]==('reject' if fraction>=.7 else 'accept')
    else:
        result=reg.resolve((item,rival),.1,2)
        assert bool(result.labels[0] and result.labels[0].logical_id==1)==(fraction<.7)
