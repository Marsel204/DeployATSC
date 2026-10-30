"""Real launcher startup and sustained BLAS provenance, without hardware inference."""
import json,os,subprocess,sys
from pathlib import Path
import pytest
from test_observation_sessions import run_payload


@pytest.mark.parametrize('launchers',[['launch_ui.py'],['launch_demo.py','launch_ui.py']],ids=['ui','demo'])
@pytest.mark.parametrize('clocks,overrides,requested,reason,applied,effective',[
    ('pinned',{},'auto','clocks_pinned','1','1'),
    ('unpinned',{},'auto','clocks_unpinned',None,None),
    ('unknown',{},'auto','clocks_unknown',None,None),
    ('pinned',{'OPENBLAS_NUM_THREADS':'4'},'auto','user_openblas_preserved',None,'4'),
    ('pinned',{'ATSC_BLAS_THREADS':'3'},'3','explicit_threads','3','3'),
    ('pinned',{'ATSC_BLAS_THREADS':'3','OPENBLAS_NUM_THREADS':'4'},'3','user_openblas_preserved',None,'4'),
],ids=['pinned','unpinned','unknown','user-openblas','explicit','both-overrides'])
def test_launcher_keeps_first_policy_in_telemetry(video_factory,tmp_path,launchers,clocks,overrides,requested,reason,applied,effective):
    core=Path(__file__).resolve().parents[1]
    env=dict(os.environ,YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
             PYTHONPATH=os.pathsep.join([str(core),str(core/'tests')]))
    for name in ('OPENBLAS_NUM_THREADS','ATSC_BLAS_THREADS'):env.pop(name,None)
    env.update(overrides)
    spec={'launchers':launchers,'clocks':clocks,'requested':requested,'reason':reason,'applied':applied,
          'effective':effective,'payload':run_payload(video_factory),'storage':str(tmp_path/'runs')}
    script='''
import ast,json,os,sys
from pathlib import Path
from src.common import runtime_tuning as tuning
spec=json.loads(sys.argv[1]);reads=[]
minimum=1510400 if spec['clocks']=='pinned' else 729600
policies=() if spec['clocks']=='unknown' else tuple(
    tuning.CpuPolicy(cpus,'schedutil',minimum,1510400,1510400) for cpus in ((0,1,2,3),(4,5)))
def read(*args,**kwargs):reads.append(None);return policies
tuning.read_cpu_policies=read
first=None
for launcher in spec['launchers']:
    path=Path('scripts')/launcher;prefix=[]
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,ast.Import) and any(alias.name=='argparse' for alias in node.names):break
        prefix.append(node)
    assert not any(name in sys.modules for name in ('numpy','torch','cv2'))
    exec(compile(ast.Module(body=prefix,type_ignores=[]),str(path),'exec'),
         {'__file__':str(path.resolve()),'__name__':'scripts.'+path.stem})
    if first is None:first=tuning.last_decision()
    decision=tuning.last_decision()
    print(json.dumps({'launcher':launcher,'reason':decision.reason,'applied':decision.applied,
                      'effective':os.environ.get('OPENBLAS_NUM_THREADS')}),flush=True)
    assert decision is first,'Second launcher replaced the first policy decision'
    assert (decision.requested,decision.reason,decision.applied)==(spec['requested'],spec['reason'],spec['applied'])
    assert os.environ.get('OPENBLAS_NUM_THREADS')==spec['effective']
assert len(reads)==1,'Launcher policy must evaluate CPU facts exactly once'

from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeDetector,FakeANFIS
class Manual(ObservationSession):
    def _worker(self):self.ready.set()
session=Manual(spec['payload'],spec['storage'],detector_factory=FakeDetector,anfis_factory=FakeANFIS)
expected={'policy':spec['requested'],'reason':spec['reason'],'openblas_num_threads':spec['applied']}
try:
    for elapsed in (0,1,6):
        session._sample_execution_provenance(elapsed)
        host=session.telemetry()['execution']['host']
        assert host['runtime_tuning']==expected,host
        assert host['openblas_num_threads']==spec['effective']
        assert tuning.last_decision() is first
finally:session.stop()
manifest=json.loads((session.store.directory/'manifest.json').read_text())
assert manifest['host']['runtime_tuning']==manifest['execution']['host']['runtime_tuning']==expected
assert tuning.last_decision() is first
'''
    run=subprocess.run([sys.executable,'-c',script,json.dumps(spec)],cwd=core,env=env,capture_output=True,text=True,timeout=60)
    assert run.returncode==0,run.stdout+'\n'+run.stderr
