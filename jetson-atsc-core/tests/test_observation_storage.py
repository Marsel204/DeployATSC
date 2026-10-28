"""Durable generation tests use temporary files and no native/model services."""
import json
from copy import deepcopy
import pytest
from src.experiment.export import ExperimentStore


def state(revision, name='a'):
    return {'run_id':'run','snapshot_revision':revision,'closed':False,
            'vehicles':[], 'predictions':[], 'comparisons':[],
            '_archive':[{'kind':'vehicles','id':name,'value':{'vehicle_id':name,'red_wait_sec':revision,'intervals':[]}}]}


def test_committed_generation_reader_and_stale_rejection(tmp_path):
    store=ExperimentStore(tmp_path,'run',{})
    assert hasattr(store,'read'), 'authoritative committed generation reader required'
    store.persist(state(1))
    pinned=store.pin()
    store.persist(state(2,'b'))
    assert [r['vehicle_id'] for r in store.read(pinned)['vehicles']]==['a']
    assert [r['vehicle_id'] for r in store.read()['vehicles']]==['a','b']
    with pytest.raises(ValueError,match='stale'):store.persist(state(1))
    assert store.read()['snapshot_revision']==2


@pytest.mark.parametrize('point',['archive_write','archive_fsync','checkpoint_write','checkpoint_fsync','checkpoint_replace','directory_fsync'])
@pytest.mark.parametrize('after',[False,True])
def test_failure_retry_is_one_complete_generation(tmp_path,monkeypatch,point,after):
    store=ExperimentStore(tmp_path,'run',{})
    assert hasattr(store,'_stage'), 'transactional failure-injection boundary required'
    store.persist(state(1))
    original=store._stage
    error=OSError(point)
    def fault(name,operation):
        if name!=point:return original(name,operation)
        if after:operation()
        raise error
    monkeypatch.setattr(store,'_stage',fault)
    with pytest.raises(OSError) as caught:store.persist(state(2,'b'))
    assert caught.value is error
    recovered=ExperimentStore.open(store.directory)
    evidence=recovered.read()
    assert (evidence['snapshot_revision'],[r['vehicle_id'] for r in evidence['vehicles']]) in [(1,['a']),(2,['a','b'])]
    monkeypatch.setattr(store,'_stage',original)
    store.persist(state(2,'b'))
    assert [r['vehicle_id'] for r in store.read()['vehicles']]==['a','b']
    assert store.read()['snapshot_revision']==2
    assert len((store.directory/'archive.jsonl').read_text().splitlines())==2


def test_legacy_v1_and_phase_a_v2_remain_read_only(tmp_path):
    for version in (1,2):
        folder=tmp_path/str(version);folder.mkdir()
        manifest={'schema_version':version,'run_id':str(version)}
        evidence={'vehicles':[{'vehicle_id':'old','red_wait_sec':20}], 'closed':True}
        (folder/'manifest.json').write_text(json.dumps(manifest))
        (folder/'evidence.json').write_text(json.dumps(evidence))
        before={p.name:p.read_bytes() for p in folder.iterdir()}
        assert hasattr(ExperimentStore,'open'), 'versioned reader required'
        store=ExperimentStore.open(folder)
        assert store.read()['vehicles']==evidence['vehicles']
        store.export()
        assert before=={p.name:p.read_bytes() for p in folder.iterdir()}


def test_partial_tail_cleanup_cannot_replace_original_error(tmp_path,monkeypatch):
    store=ExperimentStore(tmp_path,'run',{})
    assert hasattr(store,'_stage')
    store.persist(state(1))
    original=store._stage;error=OSError('primary failure')
    def fault(name,operation):
        if name=='archive_write':
            with (store.directory/'archive.jsonl').open('ab') as stream:stream.write(b'{partial')
            raise error
        return original(name,operation)
    monkeypatch.setattr(store,'_stage',fault)
    with pytest.raises(OSError) as caught:store.persist(state(2,'b'))
    assert caught.value is error
    assert ExperimentStore.open(store.directory).read()['snapshot_revision']==1
    monkeypatch.setattr(store,'_stage',original)
    store.persist(state(2,'b'))
    assert len(store.read()['vehicles'])==2


def test_real_session_storage_failure_suspends_and_retries_retained_payload(video_factory,tmp_path,monkeypatch):
    from src.ui.observation_session import ObservationSession
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        durable=session.lifecycle()['durable_revision']
        assert durable is not None, 'actual session must acknowledge a durable generation'
        original=session.store.persist
        calls=[]
        def fail(payload):calls.append(deepcopy(payload));raise OSError('disk full')
        monkeypatch.setattr(session.store,'persist',fail)
        session.engine.observe_system_frame(session.engine.time)  # Camera 1 admits preview frames only
        assert not session.persist()
        pending=deepcopy(session.pending_commit[0])
        for _ in range(4):assert not session.persist()
        assert all(c==pending for c in calls)
        before=session.engine.snapshot(evidence=True)
        assert not session.engine.observe_system_frame(session.engine.time)
        assert session.engine.snapshot(evidence=True)==before
        with pytest.raises(ValueError,match='storage|ended|stopping'):session.predict('new')
        assert session.lifecycle()['durable_revision']==durable
        monkeypatch.setattr(session.store,'persist',original)
        assert session.retry_storage()
        session.stop()
        assert session.lifecycle()['durable_finalized']
        assert session.store.recovery_status()['status']=='finalized'
        assert session.pending_commit is None
    finally:session.stop()


def test_session_telemetry_remains_available_during_archive_fsync(video_factory,tmp_path,monkeypatch):
    import threading
    from src.ui.observation_session import ObservationSession
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    entered=threading.Event();release=threading.Event();done=threading.Event()
    assert hasattr(session.store,'_stage')
    original=session.store._stage
    def block(name,operation):
        if name=='archive_fsync':entered.set();assert release.wait(5)
        return original(name,operation)
    monkeypatch.setattr(session.store,'_stage',block)
    # A new generation is required to exercise IO rather than idempotent retry.
    session.engine.observe_system_frame(session.engine.time)
    writer=threading.Thread(target=session.persist);writer.start()
    try:
        assert entered.wait(2)
        reader=threading.Thread(target=lambda:(session.telemetry(),done.set()));reader.start()
        assert done.wait(1),'coherent telemetry must not wait on disk fsync'
        reader.join(2)
    finally:release.set();writer.join(2);session.stop()


def test_historical_api_lookup_pages_export_and_recovery(video_factory,tmp_path,monkeypatch):
    import io,zipfile
    from functools import partial
    from fastapi.testclient import TestClient
    from src.ui.app import app
    import src.ui.observation_api as api
    from src.ui.observation_session import ObservationManager,ObservationSession
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    manager=ObservationManager(tmp_path/'runs',session_factory=partial(ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
    monkeypatch.setattr(api,'observations',manager)
    with TestClient(app) as client:
        payload=run_payload(video_factory)
        response=client.post('/api/observation/start',json=payload);assert response.status_code==200
        run=manager.current
        run.engine.retention.budget=2
        first=None
        for index in range(7):
            capture={'run_id':run.run_id,'request_id':f'click-{index}'}
            result=client.post('/api/observation/predict',json=capture);assert result.status_code==200,result.text
            run.engine.predictions.await_idle()
            result=client.post(f'/api/observation/predictions/click-{index}/cancel',params={'run_id':run.run_id})
            assert result.status_code==200
            if index==0:first=result.json()
        assert len(run.engine.predictions.records)<=2
        page=client.get('/api/observation/history/predictions',params={'run_id':run.run_id,'offset':0,'limit':3})
        assert page.status_code==200, 'history must be connected to the actual observation router'
        data=page.json();assert data['total']==7 and data['next_offset']==3
        generation=data['generation']
        assert client.get('/api/observation/history/predictions',params={'run_id':run.run_id,'generation':generation-1}).status_code==409
        retry=client.post('/api/observation/predict',json={'run_id':run.run_id,'request_id':'click-0'})
        assert retry.status_code==200 and retry.json()==first
        run.stop()
        quantiles=client.get('/api/observation/quantiles',params={'run_id':run.run_id}).json()
        assert quantiles['state']=='exact' and quantiles['generation']==run.durable_generation
        bundle=client.get('/api/observation/export',params={'run_id':run.run_id})
        assert bundle.status_code==200
        with zipfile.ZipFile(io.BytesIO(bundle.content)) as archive:
            evidence=json.loads(archive.read('evidence.json'))
            assert len(evidence['predictions'])==7
            assert evidence['predictions'][0]==first
            assert len(archive.read('predictions.csv').decode().splitlines())==8
        recovery=client.get('/api/observation/runs').json()
        assert recovery['runs'][0]['status']=='finalized'
        # A fresh manager has no active session; persisted history remains usable.
        monkeypatch.setattr(api,'observations',ObservationManager(tmp_path/'runs'))
        assert client.get('/api/observation/history/predictions',params={'run_id':run.run_id,'offset':3,'limit':4}).json()['total']==7
        assert client.get('/api/observation/export',params={'run_id':run.run_id}).status_code==200
        assert client.get('/api/observation/predictions/click-0',params={'run_id':run.run_id}).json()==first
        assert client.post('/api/observation/predict',json={'run_id':run.run_id,'request_id':'click-0'}).json()==first
        assert len(client.get('/api/observation/evidence',params={'run_id':run.run_id}).json()['predictions'])==7


def test_capture_write_failure_is_reported_and_recovery_stops_run(video_factory,tmp_path,monkeypatch):
    from src.ui.observation_session import ObservationSession
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    original=session.store.persist
    try:
        monkeypatch.setattr(session.store,'persist',lambda *_:(_ for _ in ()).throw(OSError('disk full')))
        with pytest.raises(OSError,match='disk full'):session.predict('capture')
        with pytest.raises(OSError,match='disk full'):session.predict('capture')
        monkeypatch.setattr(session.store,'persist',original)
        assert session.retry_storage()
        session.stop()
        assert session.state=='stopped' and session.finalized
        assert len(session.store.read()['predictions'])==1
    finally:session.stop()


def test_restart_open_generation_recovers_without_rewriting_legacy(tmp_path):
    store=ExperimentStore(tmp_path,'run',{})
    payload=state(1);payload.pop('_archive')
    payload['predictions']=[{'request_id':'blocked','status':'predicting','eligible':False,'qualified':False}]
    store.persist(payload)
    reopened=ExperimentStore.open(store.directory)
    assert reopened.recovery_status()['status']=='unclean'
    assert hasattr(reopened,'recover')
    assert reopened.recover()['status']=='finalized'
    result=reopened.read()
    assert result['predictions'][0]['status']=='unmatched'
    assert result['predictions'][0]['invalidation_reason']=='unclean_recovery'
    assert result['closed'] and result['lifecycle']['recovered_interrupted']


@pytest.mark.parametrize('mutation',['completion','closure'])
def test_concurrent_old_snapshot_cannot_regress_new_generation(tmp_path,mutation):
    import threading
    from test_observation_history import make_engine
    from src.common.types import TrafficSnapshot
    engine=make_engine();store=ExperimentStore(tmp_path,'run',{})
    entered=threading.Event();release=threading.Event();outcomes=[]
    try:
        completed=threading.Event();finish=threading.Event()
        original=engine.predictions.evaluate
        def gated(*args):
            completed.set();assert finish.wait(3);return original(*args)
        engine.predictions.evaluate=gated
        engine.update_system(TrafficSnapshot(timestamp=0,v_w_pcu=1,queue_meters=1,occupancy_pct=1))
        engine.predict('r');assert completed.wait(2)
        old=engine.snapshot(0,100,True);finish.set()
        engine.predictions.await_idle()
        if mutation=='closure':engine.close()
        newer=engine.snapshot(0,100,True)
        def old_write():
            entered.set();assert release.wait(3)
            try:store.persist(old)
            except ValueError as error:outcomes.append(str(error))
        writer=threading.Thread(target=old_write);writer.start();assert entered.wait(2)
        store.persist(newer);release.set();writer.join(3)
        assert outcomes and 'stale' in outcomes[0]
        assert store.read()['snapshot_revision']==newer['snapshot_revision']
        assert store.read()['predictions']==newer['predictions']
    finally:release.set();engine.close()


def test_cleanup_retries_open_payload_then_commits_actual_final_generation(video_factory,tmp_path,monkeypatch):
    from src.ui.observation_session import ObservationSession
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    original=session.store.persist
    try:
        session.engine.observe_system_frame(session.engine.time)
        monkeypatch.setattr(session.store,'persist',lambda *_:(_ for _ in ()).throw(OSError('disk full')))
        assert not session.persist()
        assert not session.pending_commit[2]
        monkeypatch.setattr(session.store,'persist',original)
        session.stop()
        evidence=session.store.read()
        assert session.finalized and evidence['closed'] and evidence['lifecycle']['durable_finalized']
        assert evidence['durability']['durable_revision']==session.durable_revision
    finally:session.stop()


def test_historical_rows_order_is_independent_of_finalization_order(tmp_path):
    store=ExperimentStore(tmp_path,'run',{})
    payload=state(1);payload['_archive']=[
        {'kind':'comparisons','id':2,'value':{'cycle_id':2}},
        {'kind':'predictions','id':'b','value':{'request_id':'b','clicked_at':2}}]
    store.persist(payload)
    payload=state(2);payload['_archive']=[
        {'kind':'comparisons','id':1,'value':{'cycle_id':1}},
        {'kind':'predictions','id':'a','value':{'request_id':'a','clicked_at':1}}]
    store.persist(payload)
    assert [row['cycle_id'] for row in store.page('comparisons')['rows']]==[1,2]
    assert [row['request_id'] for row in store.page('predictions')['rows']]==['a','b']


def test_recovery_cannot_mutate_another_managers_active_generation(video_factory,tmp_path,monkeypatch):
    from functools import partial
    from src.ui.observation_session import ObservationSession,ObservationManager,ObservationConflict
    from src.ui.runtime_ownership import RuntimeOwnership
    import src.ui.observation_session as module
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    shared=RuntimeOwnership();monkeypatch.setattr(module,'runtime_ownership',shared)
    manager=ObservationManager(tmp_path/'runs',session_factory=partial(ObservationSession,detector_factory=FakeDetector,anfis_factory=FakeANFIS))
    session=manager.start(run_payload(video_factory),before_handoff=lambda:None)
    other=ObservationManager(tmp_path/'runs')
    try:
        before=session.store.pin()
        with pytest.raises(ObservationConflict):other.recover_interrupted(session.run_id)
        assert session.store.pin()==before
        assert not session.engine.closed
    finally:session.stop()


def test_repeated_partial_appends_and_cleanup_failure_preserve_primary_error(tmp_path,monkeypatch):
    from pathlib import Path
    store=ExperimentStore(tmp_path,'run',{});store.persist(state(1))
    original=store._stage;primary=OSError('primary fsync failure')
    def fault(name,operation):
        if name=='checkpoint_fsync':raise primary
        return original(name,operation)
    with monkeypatch.context() as patch:
        patch.setattr(store,'_stage',fault)
        patch.setattr(Path,'unlink',lambda *a,**k:(_ for _ in ()).throw(OSError('cleanup failure')))
        for _ in range(4):
            with pytest.raises(OSError) as caught:store.persist(state(2,'b'))
            assert caught.value is primary
            assert store.read()['snapshot_revision']==1
    store.persist(state(2,'b'))
    assert len(store.read()['vehicles'])==2
    assert len((store.directory/'archive.jsonl').read_text().splitlines())==2



def test_routine_session_telemetry_never_reconstructs_archive(video_factory,tmp_path,monkeypatch):
    from src.ui.observation_session import ObservationSession
    from src.experiment.contracts import VehicleObservation
    from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        start_time=(session.engine.waits.last_time or 0)+2
        start_sequence=(session.engine.waits.last_sequence or 0)+1
        for i in range(9):
            session.engine.waits.observe([VehicleObservation(i,'car',1,(-1,0),velocity_valid=True)],'RED',start_time+i*2,start_sequence+i*2,1)
            session.engine.waits.observe([],'UNKNOWN',start_time+i*2+1.5,start_sequence+i*2+1,None)
            session.engine.revision+=1
            assert session.persist()
        monkeypatch.setattr(session.store,'read',lambda *a,**k:(_ for _ in ()).throw(AssertionError('poll scanned archive')))
        for _ in range(40):
            state=session.telemetry()
            assert state['summary']['observed_vehicle_count']==9
            assert state['summary']['quantiles']['state']=='unavailable'
    finally:session.stop()


def test_archived_request_lookup_streams_without_full_history_materialization(tmp_path,monkeypatch):
    store=ExperimentStore(tmp_path,'run',{})
    payload=state(1);payload['_archive']=[{'kind':'predictions','id':'old','value':{'request_id':'old','status':'cancelled'}}]
    store.persist(payload)
    monkeypatch.setattr(store,'read',lambda *_:(_ for _ in ()).throw(AssertionError('request lookup reconstructed all history')))
    assert store.lookup('old')=={'request_id':'old','status':'cancelled'}
    assert store.lookup('missing') is None


def test_unrecoverable_new_protocol_run_blocks_replacement_admission(tmp_path):
    from src.ui.observation_session import ObservationManager
    store=ExperimentStore(tmp_path,'run',{});store.persist(state(1))
    (store.directory/'archive.jsonl').write_bytes(b'')
    made=[]
    class Session:
        run_id='replacement';resources_released=False
        def __init__(self,*_):made.append(self)
        def stop(self,**_):self.resources_released=True
    manager=ObservationManager(tmp_path,session_factory=Session,preflight=lambda *_:None)
    try:
        with pytest.raises(ValueError,match='incomplete'):manager.start({},before_handoff=lambda:None)
        assert made==[]
    finally:
        if manager.current:manager.current.stop()


@pytest.mark.parametrize('budget',[0,-1,1.5,True])
def test_retention_budget_is_validated_before_runtime_handoff(video_factory,tmp_path,budget):
    from src.ui.observation_session import ObservationManager
    from test_observation_sessions import run_payload
    data=run_payload(video_factory);data['retention_budget']=budget
    stopped=[]
    class Owner:
        run_id='healthy';resources_released=False
        def stop(self,**_):stopped.append(True)
    manager=ObservationManager(tmp_path/'runs');manager.current=Owner()
    with pytest.raises(ValueError,match='Retention budget'):manager.start(data,before_handoff=lambda:stopped.append('legacy'))
    assert not stopped and manager.current.run_id=='healthy'
