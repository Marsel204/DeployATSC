from copy import deepcopy
import threading
from types import SimpleNamespace
import numpy as np
import pytest
from src.common.types import TrafficSnapshot
from src.experiment.contracts import RunPolicy,VehicleObservation
from src.experiment.engine import ObservationEngine
from src.experiment.export import ExperimentStore
from src.experiment.summary import summarize


def make_engine():
    return ObservationEngine('run',RunPolicy(debounce_frames=1,max_gap_sec=2,stop_persistence_sec=.3),
                             {'xmin':0,'ymin':0,'xmax':2,'ymax':2},lambda p:p[0],lambda *_:SimpleNamespace(green_seconds=25,failsafe_active=False))


def test_o1_seal_terminalizes_prediction_whose_green_cleared_pending():
    engine=make_engine();entered=threading.Event();release=threading.Event()
    def evaluate(*_):
        entered.set();assert release.wait(5)
        return SimpleNamespace(green_seconds=25,failsafe_active=False)
    engine.predictions.evaluate=evaluate
    try:
        engine.update_system(TrafficSnapshot(timestamp=0,v_w_pcu=1,queue_meters=1,occupancy_pct=1))
        engine.predict('blocked');assert entered.wait(2)
        engine.predictions.on_green(1,1)
        assert engine.predictions.pending is None
        engine.close('operator_stop')
        sealed=engine.snapshot(evidence=True)
        row=sealed['predictions'][0]
        assert row['status']=='unmatched', 'sealed in-flight row must be terminal even after pending slot cleared'
        assert row['invalidation_reason']=='operator_stop'
        assert not row['eligible'] and not row['qualified']
        release.set();engine.predictions.await_idle()
        assert engine.snapshot(evidence=True)==sealed
    finally:release.set();engine.close()


def test_ack_retention_and_long_lived_track_reconstruction(tmp_path):
    engine=make_engine();oracle=make_engine();store=ExperimentStore(tmp_path,'run',{})
    assert hasattr(engine,'enable_retention'), 'engine must produce and acknowledge archive batches'
    engine.enable_retention(3)
    frame=np.full((2,2,3),(0,0,255),np.uint8)
    try:
        for i in range(80):
            t=i*.2
            observation=VehicleObservation(1,'car',0 if i%7<4 else 1,(-1,0),velocity_valid=True)
            for e in (engine,oracle):
                e.observe_acquisition(frame,t,i,[observation])
                # Many completed cycles while one vehicle remains in the scene.
                if i%2==1:
                    e.phase.cycles.append({'cycle_id':i,'green_start':t-.1,'green_end':t,'complete':True,'actual_green_sec':.1})
            if i%4==3:
                payload,token=engine.prepare_commit()
                retained=len(engine.waits.records[ next(iter(engine.waits.records)) ]['intervals'])
                assert retained>=len(payload['vehicles'][0]['intervals'])
                store.persist(payload);engine.acknowledge(token)
                assert len(engine.phase.cycles)<=3
                assert len(next(iter(engine.waits.records.values()))['intervals'])<8
        for e in (engine,oracle):e.close()
        payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        actual=store.read();expected=oracle.snapshot(0,100,True)
        assert actual['vehicles']==expected['vehicles']
        assert actual['summary']==expected['summary']
        assert actual['comparisons']==expected['comparisons']
    finally:engine.close();oracle.close()


def test_failed_commit_cannot_evict_and_ack_is_idempotent(tmp_path,monkeypatch):
    engine=make_engine();store=ExperimentStore(tmp_path,'run',{})
    assert hasattr(engine,'enable_retention')
    engine.enable_retention(2)
    try:
        engine.waits.observe([VehicleObservation(1,'car',1,(-1,0),velocity_valid=True)],'RED',0,0,1)
        engine.waits.observe([],'RED',2,1,1)
        engine.revision+=1
        payload,token=engine.prepare_commit();before=deepcopy(engine.waits.records)
        with monkeypatch.context() as patch:
            patch.setattr(store,'_write',lambda *_:(_ for _ in ()).throw(OSError('full')))
            with pytest.raises(OSError):store.persist(payload)
        assert engine.waits.records==before
        store.persist(payload);engine.acknowledge(token);engine.acknowledge(token)
        assert not engine.waits.records and not engine.waits.active
        assert len(store.read()['vehicles'])==1
    finally:engine.close()


def test_crossed_native_track_remains_one_vehicle_after_ack(tmp_path):
    engine=make_engine();store=ExperimentStore(tmp_path,'run',{})
    engine.enable_retention(2)
    try:
        for i,x in enumerate((-1,1,2,3)):
            engine.waits.observe([VehicleObservation(7,'car',1,(x,0),velocity_valid=True)],'GREEN',i*.2,i,1)
            engine.revision+=1
            payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        assert len(store.read()['vehicles'])==1
        assert engine.snapshot()['summary']['observed_vehicle_count']==1
    finally:engine.close()


def test_retained_checkpoint_cycle_maps_bound_for_long_lived_track(tmp_path):
    engine=make_engine();engine.enable_retention(2);store=ExperimentStore(tmp_path,'run',{})
    try:
        for i in range(100):
            engine.waits.observe([VehicleObservation(7,'car',0,(-1,0),velocity_valid=True)],'RED',i*.2,i,i+1)
            engine.phase.cycle_id=i+1
            engine.phase.cycles.append({'cycle_id':i+1,'green_start':i*.2,'green_end':i*.2+.1,'complete':True,'actual_green_sec':.1})
            engine.revision+=1
            payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
            row=next(iter(engine.waits.records.values()))
            assert len(row['cycles'])<=3 and len(row['wait_by_cycle'])<=3
        assert len(store.read()['vehicles'][0]['cycles'])==100
    finally:engine.close()


def test_incremental_aggregates_match_fractional_full_history_oracle(tmp_path):
    engine=make_engine();oracle=make_engine();engine.enable_retention(2);store=ExperimentStore(tmp_path,'run',{})
    for e in (engine,oracle):e.waits.policy=RunPolicy(stop_persistence_sec=0,max_gap_sec=2,debounce_frames=1)
    try:
        for i in range(37):
            for e in (engine,oracle):
                base=i*.2
                e.waits.observe([VehicleObservation(i,'car',0,(-1,0),velocity_valid=True)],'RED',base,i*3,1)
                e.waits.observe([VehicleObservation(i,'car',0,(-1,0),velocity_valid=True)],'RED',base+1/30,i*3+1,1)
                e.waits.observe([VehicleObservation(i,'car',1,(1,0),velocity_valid=True)],'GREEN',base+2/30,i*3+2,1)
                e.revision+=1
            # zero persistence enables fractional stopped accounting in this fixture
            # (rows may remain partial; both populations must match exactly).
            payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        actual=engine.snapshot()['summary'];expected=oracle.snapshot()['summary']
        for key,value in expected.items():
            if key not in ('observed_median_red_wait_sec','observed_p95_red_wait_sec'):
                assert actual[key]==value,(key,actual[key],value)
        assert store.read()['summary']==expected
    finally:engine.close();oracle.close()


@pytest.mark.parametrize('history',[20,200,2000])
def test_scale_equal_active_population_has_bounded_live_work(tmp_path,history):
    import json,os,resource,time,statistics
    from pathlib import Path
    engine=make_engine();engine.enable_retention(4);store=ExperimentStore(tmp_path,'run',{})
    latencies=[];snapshots=[];bounds=[]
    try:
        for i in range(history):
            for step,(color,x) in enumerate([((0,0,255),-1),((0,255,0),-1),((0,255,255),1)]):
                sequence=i*3+step;t=sequence*.2
                frame=np.full((2,2,3),color,np.uint8)
                engine.observe_acquisition(frame,t,sequence,[
                    VehicleObservation(0,'car',0,(-2,0),velocity_valid=True),
                    VehicleObservation(i+1,'car',1,(x,0),velocity_valid=True)])
                if step==0:
                    engine.update_system(TrafficSnapshot(timestamp=t,v_w_pcu=1,queue_meters=1,occupancy_pct=1))
                    engine.predict(f'capture-{i}');engine.predictions.await_idle()
            payload,token=engine.prepare_commit();started=time.perf_counter()
            store.persist(payload);engine.acknowledge(token)
            latencies.append((time.perf_counter()-started)*1000)
            started=time.perf_counter();view=engine.snapshot();snapshots.append((time.perf_counter()-started)*1000)
            if i>8:
                bounds.append((len(engine.waits.records),len(engine.waits.active),len(engine.phase.cycles),len(engine.phase.events),
                    len(engine.predictions.records),sum(len(r['intervals']) for r in engine.waits.records.values()),
                    sum(len(r['cycles'])+len(r['wait_by_cycle']) for r in engine.waits.records.values()),
                    engine.retention.visited_records,(store.directory/'evidence.json').stat().st_size))
        maxima=[max(b[k] for b in bounds) for k in range(len(bounds[0]))]
        assert maxima[0]<=1 and maxima[1]<=3 and maxima[2]<=4 and maxima[3]<=4
        assert maxima[5]<=3 and maxima[6]<=12 and maxima[7]<=18 and maxima[8]<30000
        evidence=store.read()
        assert len(evidence['vehicles'])==history+1
        assert len(evidence['comparisons'])==history
        assert len(evidence['predictions'])==history
        assert evidence['summary']['timing_eligible_prediction_count']==history
        assert maxima[4]<=8
        assert evidence['vehicles'][0]['cycles']==list(range(1,history+1))
        destination=os.environ.get('ATSC_STORAGE_SCALE_EVIDENCE')
        if destination:
            Path(destination).mkdir(exist_ok=True)
            result={'finalized_history':history,'active_population':1,'retention_budget':4,
                'maxima':dict(zip(['records','active_with_tombstones','cycles','events','predictions','intervals','cycle_map_entries','visited_records','checkpoint_bytes'],maxima)),
                'persist_ms':{'median':statistics.median(latencies),'p95':sorted(latencies)[int(len(latencies)*.95)]},
                'snapshot_ms':{'median':statistics.median(snapshots),'p95':sorted(snapshots)[int(len(snapshots)*.95)]},
                'rss_max_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'archive_bytes':(store.directory/'archive.jsonl').stat().st_size}
            (Path(destination)/f'{history}.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:engine.close()


def test_archival_does_not_round_samples_before_exact_mean(tmp_path):
    engine=make_engine();oracle=make_engine();engine.enable_retention(1);store=ExperimentStore(tmp_path,'run',{})
    for e in (engine,oracle):e.waits.policy=RunPolicy(stop_persistence_sec=0,max_gap_sec=2)
    try:
        for i,duration in enumerate((.420571580830845,.25891675029296335)):
            for e in (engine,oracle):
                e.waits.observe([VehicleObservation(i,'car',0,(-1,0),velocity_valid=True)],'RED',i*2,i*2,1)
                e.waits.observe([VehicleObservation(i,'car',1,(1,0),velocity_valid=True)],'GREEN',i*2+duration,i*2+1,1)
                e.revision+=1
            payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        expected=oracle.snapshot()['summary']['observed_mean_red_wait_sec']
        assert engine.snapshot()['summary']['observed_mean_red_wait_sec']==expected
        assert store.read()['summary']['observed_mean_red_wait_sec']==expected
        assert store.quantiles()['median_red_wait_sec']==expected
    finally:engine.close();oracle.close()


def test_served_population_remains_correct_after_ack_during_running_green(tmp_path):
    engine=make_engine();engine.enable_retention(2);store=ExperimentStore(tmp_path,'run',{})
    try:
        for sequence,(color,x) in enumerate([((0,0,255),-1),((0,255,0),-1),((0,255,0),1)]):
            engine.observe_acquisition(np.full((2,2,3),color,np.uint8),sequence*.2,sequence,
                                       [VehicleObservation(7,'car',1,(x,0),velocity_valid=True)])
        payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        engine.observe_acquisition(np.full((2,2,3),(0,255,0),np.uint8),.6,3,[])
        assert engine.snapshot()['comparisons'][0]['n_served']==1
    finally:engine.close()


def test_finalized_tombstone_drops_hysteresis_dependency_after_ack(tmp_path):
    engine=make_engine();engine.enable_retention(2);store=ExperimentStore(tmp_path,'run',{})
    try:
        for sequence,(speed,x) in enumerate([(0,-1),(0,-1),(0,-1),(1,1)]):
            engine.observe_acquisition(np.full((2,2,3),(0,0,255),np.uint8),sequence*.2,sequence,
                                       [VehicleObservation(7,'car',speed,(x,0),velocity_valid=True)])
        payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        assert all(a['stop_candidate'] is None and a['move_candidate'] is None for a in engine.waits.active.values())
        engine.revision+=1
        payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        assert len(store.read()['vehicles'])==1
    finally:engine.close()


def test_compaction_gaps_zeros_partial_populations_and_backdating_match_oracle(tmp_path):
    engine=make_engine();oracle=make_engine();engine.enable_retention(3);store=ExperimentStore(tmp_path,'run',{})
    timestamp=0.
    try:
        for sequence in range(120):
            timestamp+=.9 if sequence in (17,58,95) else .1
            color=[(0,0,255),(0,255,0),(0,255,255),(0,0,0)][(sequence//7)%4]
            frame=np.full((2,2,3),color,np.uint8)
            phase=sequence%8
            rows=[VehicleObservation(1,'car',0 if sequence%13<7 else 1,(-1,0),velocity_valid=sequence%19!=0),
                  VehicleObservation(10+sequence//8,'car',0 if phase<4 else 1,(-1 if phase<6 else 1,0),velocity_valid=True)]
            for e in (engine,oracle):e.observe_acquisition(frame,timestamp,sequence,rows)
            if sequence%3==2:
                payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
                actual=engine.snapshot()['summary'];expected=oracle.snapshot()['summary']
                for key in expected:
                    if key not in ('observed_median_red_wait_sec','observed_p95_red_wait_sec'):
                        assert actual[key]==expected[key],(sequence,key,actual[key],expected[key])
        for e in (engine,oracle):e.close()
        payload,token=engine.prepare_commit();store.persist(payload);engine.acknowledge(token)
        import json
        actual=store.read();expected=json.loads(json.dumps(oracle.snapshot(0,1000,True)))
        assert actual['vehicles']==expected['vehicles']
        assert actual['comparisons']==expected['comparisons']
        assert actual['summary']==expected['summary']
    finally:engine.close();oracle.close()
