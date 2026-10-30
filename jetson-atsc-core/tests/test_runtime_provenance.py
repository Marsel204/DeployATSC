import json
from src.common import runtime_tuning as tuning
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload
from test_runtime_tuning import cpu


def test_host_manifest_advice_sampling_and_isolation(video_factory,tmp_path,monkeypatch):
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    sysfs=tmp_path/'sys';cpu(sysfs,minimum=729600)
    monkeypatch.delenv('OPENBLAS_NUM_THREADS',raising=False)
    tuning.apply_blas_policy({},sysfs)
    session=Manual(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS,sysfs_root=sysfs)
    try:
        host=session.telemetry()['execution']['host']
        assert host['clocks_pinned'] is False
        assert host['cpu_policies']==[{'cpus':[0],'governor':'schedutil','min_khz':729600,'max_khz':1510400,'cur_khz':1510400}]
        assert host['runtime_tuning']=={'policy':'auto','openblas_num_threads':None,'reason':'clocks_unpinned'}
        assert host['openblas_num_threads'] is None
        manifest=json.loads((session.store.directory/'manifest.json').read_text())
        assert manifest['host']==host and manifest['execution']['host']==host
        advice=session.telemetry()['performance_advice']
        assert len(advice)==1 and 'governor schedutil, min 729.6 MHz' in advice[0] and 'sudo jetson_clocks --restore' in advice[0]
        host['cpu_policies'][0]['min_khz']=-1
        cpu(sysfs)
        assert session.telemetry()['execution']['host']['clocks_pinned'] is False
        session._sample_execution_provenance(1)
        assert session.telemetry()['execution']['host']['clocks_pinned'] is True
        assert 'OpenBLAS threads are not capped' in session.telemetry()['performance_advice'][0]
        monkeypatch.setenv('OPENBLAS_NUM_THREADS','1');session._sample_execution_provenance(2)
        assert session.telemetry()['performance_advice']==[]
        monkeypatch.setattr(tuning,'read_cpu_policies',lambda *a:(_ for _ in ()).throw(AssertionError('telemetry must use cache')))
        session.telemetry();session.telemetry()
    finally:session.stop()


def test_unknown_sysfs_no_advice(video_factory,tmp_path):
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS,sysfs_root=tmp_path/'missing')
    try:
        host=session.execution_provenance()['host']
        assert host['clocks_pinned'] is None and host['cpu_policies']==[]
        assert session.telemetry()['performance_advice']==[]
    finally:session.stop()
