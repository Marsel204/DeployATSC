"""Write-behind durability uses the existing store and persist protocol."""
import io,json,threading,time,zipfile
import pytest
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload


class Manual(ObservationSession):
    def _worker(self):self.initializing=False;self.ready.set()


def make(video_factory,tmp_path,mode='background'):
    data=run_payload(video_factory);data['persist_mode']=mode
    s=Manual(data,tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS,sysfs_root=tmp_path/'sys')
    s.persist();return s


def wait_for(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while not predicate() and time.monotonic()<deadline:time.sleep(.001)
    assert predicate()


def idle(s):
    with s.persist_condition:
        assert s.persist_condition.wait_for(lambda:not s.persist_pending and not s.persist_inflight,5)


def dirty(s):
    with s.engine.lock:s.engine.revision+=1


def test_mode_validation_and_api_default(video_factory,tmp_path):
    from src.ui.observation_api import StartObservation
    data=run_payload(video_factory)
    assert StartObservation(**data).persist_mode=='inline'
    for mode in ('inline','background'):
        data['persist_mode']=mode;assert StartObservation(**data).persist_mode==mode
        ObservationSession.preflight(data,tmp_path,FakeDetector,FakeANFIS)
    for mode in ('queue',None,1):
        data['persist_mode']=mode
        with pytest.raises((TypeError,ValueError)):StartObservation(**data)
        with pytest.raises(ValueError,match='Persist mode'):ObservationSession.preflight(data,tmp_path,FakeDetector,FakeANFIS)


def test_coalesces_one_commit_after_inflight_and_preserves_order(video_factory,tmp_path):
    s=make(video_factory,tmp_path);entered=threading.Event();release=threading.Event();calls=[];original=s.store.persist
    def slow(state):
        calls.append((state['snapshot_revision'],threading.current_thread().name))
        if len(calls)==1:entered.set();assert release.wait(5)
        return original(state)
    s.store.persist=slow
    try:
        s.applied_state='running';dirty(s);s.engine.time=7.5;s._request_persist()
        assert entered.wait(2) and s.last_persisted==7.5
        for _ in range(20):dirty(s);s._request_persist()
        assert len(calls)==1
        release.set();idle(s)
        assert len(calls)==2 and calls[0][0]<calls[1][0]
        assert all(name=='Observation-persist-'+s.run_id for _,name in calls)
        telemetry=s.telemetry()
        assert telemetry['persist_background_requests']==21 and telemetry['persist_coalesced_requests']==19
        assert telemetry['persist_background_commits']==2 and telemetry['persist_worker_errors']==0
        assert s.store.read()['snapshot_revision']==s.engine.revision
    finally:release.set();s.stop()
    assert s.finalized and not s.persist_thread.is_alive()


def test_trigger_sites_do_not_run_store_on_measurement_thread_or_skip_camera2(video_factory,tmp_path,monkeypatch):
    s=make(video_factory,tmp_path);entered=threading.Event();release=threading.Event();original=s.store.persist
    def slow(state):entered.set();assert release.wait(5);return original(state)
    s.store.persist=slow
    try:
        s.applied_state='running';monkeypatch.setattr(s.engine,'retention_due',lambda:True)
        s._process_batch();assert entered.wait(2)
        for _ in range(12):s._process_batch()
        assert s.engine.processed['acquisition']==13
        assert s.runtime['acquisition']['frames']==13
        assert s.engine.phase.last_sequence==13
    finally:release.set();s.stop()


@pytest.mark.parametrize('state',['paused','ended','stopped'])
def test_nonrunning_and_inline_mode_preserve_inline_persist(video_factory,tmp_path,state):
    s=make(video_factory,tmp_path);calls=[];original=s.persist
    def record(*a,**kw):calls.append((threading.current_thread(),kw));return original(*a,**kw)
    s.persist=record
    try:
        s.applied_state=state;dirty(s);s._request_persist()
        assert calls==[(threading.current_thread(),{'wait':False})]
        s.applied_state='running';s.persist_mode='inline';dirty(s);s._request_persist()
        assert calls[-1]==(threading.current_thread(),{'wait':False}) and len(calls)==2
        assert s.telemetry()['persist_background_requests']==0
    finally:s.stop()


def test_step_mode_is_inline_deterministic(video_factory,tmp_path,monkeypatch):
    s=make(video_factory,tmp_path);names=[];original=s.store.persist
    def record(state):names.append(threading.current_thread().name);return original(state)
    s.store.persist=record
    try:
        monkeypatch.setattr(s.engine,'retention_due',lambda:True)
        for _ in range(5):s.steps=1;s._process_batch()
        assert s.engine.processed['acquisition']==5 and names
        assert set(names)=={threading.current_thread().name} and s.telemetry()['persist_background_requests']==0
    finally:s.stop()


def test_storage_failure_seals_stops_admission_retains_payload_and_retries(video_factory,tmp_path):
    s=make(video_factory,tmp_path);original=s.store.persist;calls=0
    def failing(state):
        nonlocal calls
        calls+=1
        if calls>=2:raise OSError('disk full on second background commit')
        return original(state)
    s.store.persist=failing
    try:
        s.applied_state='running';s._process_batch();s._request_persist();idle(s)
        s._process_batch();s._request_persist();idle(s)
        assert s.storage_suspended and s.engine.closed and 'disk full' in s.storage_error
        assert s.pending_commit is not None
        admitted=s.engine.processed.copy();sequence=s.pair.acquisition.sequence;frames=s.runtime['acquisition']['frames']
        for _ in range(3):s._process_batch()
        assert s.engine.processed==admitted and s.pair.acquisition.sequence==sequence and s.runtime['acquisition']['frames']==frames
        with pytest.raises(ValueError,match='suspended'):s.predict('blocked')
        s.stop();assert s.resources_released and not s.finalized and s.pending_commit is not None
        s.store.persist=original;assert s.retry_storage();s.stop()
        assert s.finalized and s.pending_commit is None
        saved=s.store.read();assert saved['closed'] and saved['lifecycle']['durable_finalized']
        assert saved['processed_frames']==admitted
    finally:s.store.persist=original;s.stop()


def test_stop_and_finalize_wait_for_inflight_commit(video_factory,tmp_path):
    s=make(video_factory,tmp_path);entered=threading.Event();release=threading.Event();finished=threading.Event();original=s.store.persist;finals=[]
    def slow(state):
        if not entered.is_set():entered.set();assert release.wait(5)
        finals.append(state['lifecycle'].get('finalization_requested',False));return original(state)
    s.store.persist=slow
    try:
        s.applied_state='running';dirty(s);s._request_persist();assert entered.wait(2)
        def stop():s.stop();finished.set()
        thread=threading.Thread(target=stop);thread.start()
        wait_for(lambda:s.stopping.is_set());assert not finished.wait(.05) and not s.finalized
        assert s.persist_thread.is_alive() and not s.resources_released
        release.set();thread.join(5);assert finished.is_set() and s.finalized
        assert finals==[False,True] and not s.persist_thread.is_alive()
    finally:release.set();s.stop()


def test_export_serializes_behind_background_commit(video_factory,tmp_path):
    s=make(video_factory,tmp_path);entered=threading.Event();release=threading.Event();done=threading.Event();original=s.store.persist;exports=[]
    def slow(state):entered.set();assert release.wait(5);return original(state)
    s.store.persist=slow
    thread=None
    try:
        s.applied_state='running';s._process_batch();s._request_persist();assert entered.wait(2)
        s._process_batch()
        def export():exports.append(s.export());done.set()
        thread=threading.Thread(target=export);thread.start();assert not done.wait(.05)
        release.set();thread.join(5);assert done.is_set()
        with zipfile.ZipFile(io.BytesIO(exports[0])) as bundle:
            evidence=json.loads(bundle.read('evidence.json'))
        assert evidence['processed_frames']['acquisition']==2
    finally:
        release.set()
        if thread:thread.join(5)
        s.stop()


def test_unexpected_worker_exception_contained_and_next_request_runs(video_factory,tmp_path):
    s=make(video_factory,tmp_path);original=s.persist;calls=[]
    def broken(*a,**kw):
        calls.append(kw)
        if len(calls)==1:raise RuntimeError('unexpected')
        return original(*a,**kw)
    s.persist=broken
    try:
        s.applied_state='running';dirty(s);s._request_persist();idle(s)
        assert s.telemetry()['persist_worker_errors']==1 and s.persist_thread.is_alive()
        dirty(s);s._request_persist();idle(s)
        assert s.telemetry()['persist_background_commits']==1 and not s.storage_suspended
        assert calls==[{'wait':True},{'wait':True}]
    finally:s.stop()


def test_fifty_start_stop_cycles_leave_no_threads(video_factory,tmp_path):
    data=run_payload(video_factory);data['persist_mode']='background'
    before={t.ident for t in threading.enumerate()}
    for _ in range(50):
        s=ObservationSession(data,tmp_path/'cycles',detector_factory=FakeDetector,anfis_factory=FakeANFIS,sysfs_root=tmp_path/'sys')
        s.stop();assert s.finalized and not s.persist_thread.is_alive()
    assert not [t.name for t in threading.enumerate() if t.ident not in before and t.name.startswith('Observation')]


def test_sigkill_mid_background_commit_recovers_with_existing_manager(video_factory,tmp_path):
    import os,signal,subprocess,sys
    from pathlib import Path
    from src.ui.observation_session import ObservationManager
    from src.experiment.export import ExperimentStore
    data=run_payload(video_factory);data['persist_mode']='background';data['retention_budget']=1
    payload=tmp_path/'payload.json';payload.write_text(json.dumps(data));ready=tmp_path/'ready.json';root=tmp_path/'crashed'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    log=(tmp_path/'child.log').open('w')
    process=subprocess.Popen([sys.executable,str(Path(__file__).with_name('persist_crash_runner.py')),str(payload),str(root),str(ready)],env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
        wait_for(lambda:ready.exists() or process.poll() is not None,15)
        assert ready.exists(),(tmp_path/'child.log').read_text()
        checkpoint=json.loads(ready.read_text());process.kill();process.wait(5)
        assert process.returncode==-signal.SIGKILL
        store=ExperimentStore.open(root/checkpoint['run_id']);assert store.recovery_status()['status']=='unclean'
        before=store.read();assert before['processed_frames']==checkpoint['processed']
        assert before['predictions'][0]['status']=='predicting'
        manager=ObservationManager(root,tmp_path/'profiles');result=manager.recover_interrupted(checkpoint['run_id'])
        assert result['status']=='finalized' and result['durable_finalized']
        saved=ExperimentStore.open(root/checkpoint['run_id']).read()
        assert saved['closed'] and saved['lifecycle']['recovered_interrupted']
        assert saved['processed_frames']==before['processed_frames']
        assert saved['predictions'][0]['status']=='unmatched'
        assert saved['predictions'][0]['invalidation_reason']=='unclean_recovery'
        assert saved['summary']==dict(before['summary'],unknown_qualification_prediction_count=0)
    finally:
        if process.poll() is None:process.kill();process.wait(5)
        log.close()


def test_background_storage_error_preserves_credential_redaction(video_factory,tmp_path):
    from src.common.source_identity import public_source
    s=make(video_factory,tmp_path);original=s.store.persist
    source='rtsp://operator:persist-secret-pass@camera.local/private'
    public_source(source)
    def fail(state):raise OSError('storage rejected '+source+' persist-secret-pass')
    try:
        s.store.persist=fail;s.applied_state='running';dirty(s);s._request_persist();idle(s)
        serialized=json.dumps(s.telemetry())
        assert s.storage_suspended and s.engine.closed
        assert 'persist-secret-pass' not in serialized and 'operator@' not in serialized and source not in serialized
        assert s.telemetry()['lifecycle']['storage_suspended']
    finally:s.store.persist=original;s.stop()


def test_unfenced_background_saved_domain_and_predictions_equal_inline(tmp_path,monkeypatch):
    """Real retained evidence with no per-request worker fence; CPU fixtures only."""
    from test_observation_storage_corrections import scripted_payload,ScriptedDetector,neutral
    from test_critical_path_golden import normalize
    monkeypatch.setattr(time,'perf_counter',lambda:100.)
    data=scripted_payload(tmp_path,budget=1);results={}
    for mode in ('inline','background'):
        data['persist_mode']=mode
        s=Manual(data,tmp_path/mode,detector_factory=ScriptedDetector,anfis_factory=FakeANFIS,sysfs_root=tmp_path/'sys')
        try:
            s.state=s.applied_state='running'
            for frame in range(140):
                s._process_batch()
                if frame in (15,95):
                    s.predict('click-'+str(frame));s.engine.predictions.await_idle();s.persist()
            assert s.engine.processed['acquisition']==140 and not s.storage_suspended
            s.state=s.applied_state='ended';s._cleanup();assert s.finalized
            saved=s.store.read()
            keys=('vehicles','comparisons','summary','phase_events','observation_gaps','processed_frames','predictions')
            results[mode]=normalize(neutral({key:saved[key] for key in keys},s.run_id))
            assert saved['vehicles'] and len(saved['predictions'])==2
        finally:s.stop()
    assert results['inline']==results['background']
