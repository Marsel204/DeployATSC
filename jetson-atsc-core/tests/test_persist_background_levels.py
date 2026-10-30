"""Level-triggered retention and periodic durability must not queue duplicate I/O."""
import json,sys,threading,time
from collections import deque
import pytest
from test_persist_background import Manual,make,idle
from test_observation_sessions import FakeANFIS


def gap(s):
    # Real retained evidence and acknowledgment, without unrelated tracker changes.
    with s.engine.lock:
        s.engine.gaps.append({'timestamp':s.engine.time,'reason':'retention-test'})
        s.engine.revision+=1


def slow_store(s):
    entered=threading.Event();release=threading.Event();calls=[];original=s.store.persist
    def persist(state):
        calls.append(state['snapshot_revision']);entered.set();assert release.wait(5)
        time.sleep(.05)
        return original(state)
    s.store.persist=persist
    return entered,release,calls


def test_pending_request_is_coalesced_before_worker_takes_it(video_factory,tmp_path):
    s=make(video_factory,tmp_path)
    try:
        s.applied_state='running';gap(s)
        with s.persist_condition:
            s._request_persist();s.engine.time=5.;s._request_persist()
            assert s.persist_pending and not s.persist_inflight
            assert s.last_persisted==5.
            assert s._persist_counters()==dict(persist_background_requests=1,persist_background_commits=0,
                persist_coalesced_requests=1,persist_worker_errors=0)
        idle(s);assert s._persist_counters()['persist_background_commits']==1
    finally:s.stop()


def test_slow_store_commits_once_per_retention_due_episode(video_factory,tmp_path):
    s=make(video_factory,tmp_path);s.applied_state='running';s._process_batch();s.engine.retention.budget=1
    entered,release,calls=slow_store(s)
    try:
        for episode in (1,2):
            entered.clear();release.clear();gap(s);gap(s)
            assert s.engine.retention_due()
            s._process_batch();assert entered.wait(2)
            for _ in range(12):
                assert s.engine.retention_due();s._process_batch()
            assert len(calls)==episode
            release.set();idle(s)
            assert not s.engine.retention_due()
            s._process_batch();idle(s)
            assert len(calls)==episode  # Physical store calls, not worker successes.
    finally:release.set();s.stop()


@pytest.mark.parametrize('still_due',[False,True])
def test_inflight_coalesces_then_next_frame_rechecks_acknowledged_level(video_factory,tmp_path,still_due):
    s=make(video_factory,tmp_path);s.applied_state='running';s._process_batch();s.engine.retention.budget=1
    entered,release,calls=slow_store(s)
    try:
        gap(s);gap(s);s._request_persist();assert entered.wait(2)
        if still_due:gap(s)
        for _ in range(4):s._process_batch()
        assert s.persist_inflight and not s.persist_pending
        release.set();idle(s)
        assert len(calls)==1 and s.engine.retention_due()==still_due
        # Only a new measurement-thread due check may request the next commit.
        s._process_batch();idle(s)
        assert len(calls)==(2 if still_due else 1)
        assert not s.engine.retention_due()
        assert len(s.store.read()['observation_gaps'])==(3 if still_due else 2)
    finally:release.set();s.stop()


def test_periodic_five_source_seconds_coalesces_with_inflight_commit(video_factory,tmp_path):
    s=make(video_factory,tmp_path);entered,release,calls=slow_store(s)
    try:
        s.applied_state='running';s.engine.time=5.;s._process_batch();assert entered.wait(2)
        assert not s.engine.retention_due() and s.last_persisted==5.
        for timestamp in (10.,15.):
            s.engine.time=timestamp;s._process_batch()
            assert s.last_persisted==timestamp and len(calls)==1
        release.set();idle(s);s._process_batch();idle(s)
        assert len(calls)==1 and not s.engine.retention_due()
        s.engine.time=20.;s._process_batch();idle(s)
        assert len(calls)==2 and s.last_persisted==20.
    finally:release.set();s.stop()


def deep_size(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,dict):return size+sum(deep_size(k,seen)+deep_size(v,seen) for k,v in value.items())
    if isinstance(value,(list,tuple,set,frozenset,deque)):return size+sum(deep_size(v,seen) for v in value)
    return size


def test_ten_source_second_store_stall_retained_evidence_growth(tmp_path,record_property):
    from test_observation_storage_corrections import scripted_payload,ScriptedDetector
    from src.perception.detector_tracker import TrackedVehicle
    class Detector(ScriptedDetector):
        def track(self,frames,timestamp=None):
            rows=[TrackedVehicle(i,0,'car',[6+i*6,10,10+i*6,20],.9,timestamp,obs_count=6) for i in range(8)]
            for v in rows:v.velocity_valid=True;v.velocity_mps=0.
            return [rows]
    data=scripted_payload(tmp_path,budget=1);data['persist_mode']='background'
    s=Manual(data,tmp_path/'runs',detector_factory=Detector,anfis_factory=FakeANFIS,sysfs_root=tmp_path/'sys')
    entered,release,calls=slow_store(s);samples=[]
    def sample():
        with s.engine.lock:
            roots=(s.engine.waits.records,s.engine.waits.active,s.engine.phase.events,s.engine.phase.cycles,
                s.engine.phase.green,s.engine.gaps,s.engine.predictions.records,s.engine.retention.pending,s.pending_commit)
            samples.append({'source_seconds':s.engine.time,'bytes':deep_size(roots),
                'intervals':sum(len(r['intervals']) for r in s.engine.waits.records.values())})
    try:
        s.applied_state='running';s._process_batch();s._process_batch();assert entered.wait(2);sample()
        for frame in range(100):
            s._process_batch()
            if (frame+1)%25==0:sample()
        assert len(calls)==1 and s.persist_inflight and not s.storage_suspended
        elapsed=samples[-1]['source_seconds']-samples[0]['source_seconds']
        growth=samples[-1]['bytes']-samples[0]['bytes'];rate=growth/elapsed
        result={'samples':samples,'elapsed_source_seconds':elapsed,'growth_bytes':growth,'bytes_per_source_second':rate,
            'scope':'deep size: retained rows/active state/events/cycles/gaps/predictions plus both pending snapshots; unique objects counted once'}
        record_property('stall_growth',json.dumps(result));print('STALL_GROWTH '+json.dumps(result))
        assert elapsed==pytest.approx(10.) and samples[-1]['intervals']>samples[0]['intervals']
        assert rate<=2_000_000,result
    finally:release.set();s.stop()
