from copy import deepcopy
import random
from types import SimpleNamespace
import numpy as np
import pytest

from src.experiment.engine import ObservationEngine
from src.experiment.contracts import RunPolicy,VehicleObservation
from critical_path_engine_oracle import ObservationEngine as Oracle
from critical_path_retention_oracle import Retention as OracleRetention


def engines(budget=8):
    policy=RunPolicy(debounce_frames=1,stop_persistence_sec=.1,association_gap_sec=.3)
    result=[cls('fixed',policy,{'xmin':0,'ymin':0,'xmax':20,'ymax':20},lambda p:p[1]-12,lambda s,c:None) for cls in (ObservationEngine,Oracle)]
    for engine in result:engine.enable_retention(budget)
    result[1].retention=OracleRetention(budget);result[1].waits.archived=result[1].retention.identity
    return result


def brute(engine):
    return sum(len(r['intervals'])+int(r['completed']) for r in engine.waits.records.values())+max(0,len(engine.phase.events)-engine.retention.budget)+max(0,len(engine.gaps)-engine.retention.budget)


def test_random_observation_snapshot_commit_and_acknowledge_equality():
    new,old=engines();rng=random.Random(29102026)
    for seq in range(1,401):
        t=seq/10;phase=rng.choice([(0,0,255),(0,255,0),(0,255,255)]) if seq%11==0 else (0,0,255)
        observations=[VehicleObservation(i,'car',rng.choice([0.,1.]),(i,rng.choice([3,9,16])),velocity_valid=rng.choice([True,False]),in_region=rng.choice([True,True,False]),epoch=seq//100) for i in range(8) if rng.random()<.8]
        for e in (new,old):
            e.observe_system_frame(t);e.observe_acquisition(np.full((20,20,3),phase,np.uint8),t,seq,observations)
            if seq%37==0:e.invalidate('random_gap')
        assert new.waits.work==brute(new)-max(0,len(new.phase.events)-new.retention.budget)-max(0,len(new.gaps)-new.retention.budget)
        assert new.retention_due()==(brute(new)>=new.retention.budget)
        for evidence in (False,True):assert new.snapshot(1,7,evidence)==old.snapshot(1,7,evidence)
        if seq%19==0:
            a,ta=new.prepare_commit();b,tb=old.prepare_commit();assert (a,ta)==(b,tb)
            assert new.prepare_commit()==old.prepare_commit()
            for e,token in ((new,ta),(old,tb)):e.acknowledge(token);e.acknowledge(token)
            assert new.waits.work==sum(len(r['intervals'])+int(r['completed']) for r in new.waits.records.values())
            assert new.snapshot(evidence=True)==old.snapshot(evidence=True)
    for e in (new,old):e.close()
    assert new.waits.work==sum(len(r['intervals'])+int(r['completed']) for r in new.waits.records.values())
    assert new.snapshot(evidence=True)==old.snapshot(evidence=True)


def test_snapshot_mutation_isolation_and_aggregate_cache(monkeypatch):
    engine,_=engines()
    engine.observe_acquisition(np.full((20,20,3),(0,0,255),np.uint8),0,1,[VehicleObservation(1,'car',0,(1,1),velocity_valid=True)])
    calls=[];summary=engine.retention.summary
    monkeypatch.setattr(engine.retention,'summary',lambda *a:calls.append(1) or summary(*a))
    first=engine.snapshot(evidence=True);saved=deepcopy(first)
    first['vehicles'][0]['cycles'].append('bad');first['summary']['quantiles']['state']='bad';first['phase_events'].clear();first['policy']['movement_id']='bad'
    assert engine.snapshot(evidence=True)==saved
    assert len(calls)==1
    state,token=engine.prepare_commit();engine.acknowledge(token)
    engine.snapshot(evidence=True)
    assert len(calls)>=2


def test_retention_due_never_scans_records():
    engine,_=engines()
    for i in range(2000):engine.waits._new(VehicleObservation(i,'car',0,(1,1)),0,1)
    class NoScan(dict):
        def values(self):raise AssertionError('retention_due scanned wait records')
    engine.waits.records=NoScan(engine.waits.records)
    assert not engine.retention_due()


def test_specialized_clone_falls_back_for_non_json_values():
    from src.common.json_data import clone
    custom=SimpleNamespace(values=[1,2])
    value={'plain':[None,True,1,1.5,'x',{'tuple':(1,[2])}],'custom':custom}
    result=clone(value);assert result==value
    result['plain'][5]['tuple'][1].append(3);result['custom'].values.append(3)
    assert value['plain'][5]['tuple'][1]==[2] and custom.values==[1,2]


def test_clone_non_json_mapping_keys_and_cycles_use_deepcopy_semantics():
    from src.common.json_data import clone
    class Key:pass
    key=Key();key.items=[1];value={key:[2]}
    result=clone(value);other=next(iter(result));other.items.append(3)
    assert key.items==[1] and result[other]==[2]
    cycle=[];cycle.append(cycle);copied=clone(cycle)
    assert copied is copied[0] and copied is not cycle
