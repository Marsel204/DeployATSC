"""Startup-only BLAS policy; sysfs is always synthetic here."""
import ast,dataclasses,json,os,subprocess,sys
from pathlib import Path
import pytest
from src.common import runtime_tuning as tuning


def cpu(root,index=0,governor='schedutil',minimum=1510400,maximum=1510400,current=1510400,online=True):
    base=root/f'devices/system/cpu/cpu{index}';base.mkdir(parents=True,exist_ok=True)
    (base/'online').write_text('1' if online else '0')
    policy=base/'cpufreq';policy.mkdir(exist_ok=True)
    for key,value in [('scaling_governor',governor),('scaling_min_freq',minimum),('scaling_max_freq',maximum),('scaling_cur_freq',current)]:
        (policy/key).write_text(str(value))
    return policy


@pytest.mark.parametrize('governor,minimum,cap',[('schedutil',1510400,'1'),('schedutil',729600,None),('performance',729600,'1')])
def test_auto_pinning(tmp_path,governor,minimum,cap):
    cpu(tmp_path,governor=governor,minimum=minimum);env={}
    decision=tuning.apply_blas_policy(env,tmp_path)
    assert env.get('OPENBLAS_NUM_THREADS')==cap and decision.applied==cap
    assert tuning.last_decision() is decision and decision.requested=='auto'
    assert decision.policies[0].cur_khz==1510400
    with pytest.raises(dataclasses.FrozenInstanceError):decision.reason='bad'


def test_mixed_offline_and_shared_policies(tmp_path):
    policy=cpu(tmp_path);cpu(tmp_path,1,minimum=729600)
    assert tuning.apply_blas_policy({},tmp_path).applied is None
    (tmp_path/'devices/system/cpu/cpu1/online').write_text('0')
    assert tuning.apply_blas_policy({},tmp_path).applied=='1'
    base=tmp_path/'devices/system/cpu/cpu2';base.mkdir();(base/'cpufreq').symlink_to(policy,target_is_directory=True)
    decision=tuning.apply_blas_policy({},tmp_path)
    assert len(decision.policies)==1 and decision.policies[0].cpus==(0,2)


@pytest.mark.parametrize('mode,expected',[(None,'1'),('auto','1'),('default',None),('3','3'),('0',None),('-1',None),('bad',None)])
def test_modes_and_user_override(tmp_path,mode,expected):
    cpu(tmp_path);env={} if mode is None else {'ATSC_BLAS_THREADS':mode}
    assert tuning.apply_blas_policy(env,tmp_path).applied==expected
    env['OPENBLAS_NUM_THREADS']='7';tuning.apply_blas_policy(env,tmp_path)
    assert env['OPENBLAS_NUM_THREADS']=='7'


def test_missing_invalid_and_permission_denied_sysfs(tmp_path,monkeypatch):
    assert tuning.apply_blas_policy({},tmp_path).applied is None
    p=cpu(tmp_path);(p/'scaling_min_freq').write_text('invalid')
    assert tuning.apply_blas_policy({},tmp_path).applied is None
    (p/'scaling_min_freq').write_text('1510400')
    original=Path.read_text
    def denied(path,*a,**kw):
        if path.name=='scaling_governor':raise PermissionError('fixture')
        return original(path,*a,**kw)
    monkeypatch.setattr(Path,'read_text',denied)
    assert tuning.apply_blas_policy({},tmp_path).applied is None


def test_startup_policy_precedes_numpy_in_fresh_process(tmp_path):
    cpu(tmp_path);env=dict(os.environ,ATSC_BLAS_THREADS='auto',CUDA_VISIBLE_DEVICES='',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    env.pop('OPENBLAS_NUM_THREADS',None)
    script='''
import sys,json
from src.common.runtime_tuning import apply_blas_policy
assert not any(n in sys.modules for n in ('numpy','torch','cv2'))
apply_blas_policy(sysfs_root=sys.argv[1])
import os,numpy as np
assert os.environ['OPENBLAS_NUM_THREADS']=='1'
np.dot(np.ones((8,8)),np.ones((8,8)))
try:
 from threadpoolctl import threadpool_info
except ImportError:print(json.dumps({'environment_only':True}))
else:
 pools=[p for p in threadpool_info() if p['internal_api']=='openblas']
 assert all(p['num_threads']==1 for p in pools),pools
 print(json.dumps({'pools':pools}))
'''
    run=subprocess.run([sys.executable,'-c',script,str(tmp_path)],cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stderr
    assert json.loads(run.stdout)


@pytest.mark.parametrize('launcher',['launch_ui.py','launch_demo.py'])
def test_launcher_applies_policy_before_other_imports(launcher):
    tree=ast.parse((Path(__file__).resolve().parents[1]/'scripts'/launcher).read_text())
    imports=[node for node in ast.walk(tree) if isinstance(node,(ast.Import,ast.ImportFrom)) and getattr(node,'module',None)!='__future__']
    first=min(imports,key=lambda n:n.lineno)
    assert isinstance(first,ast.ImportFrom) and first.module=='src.common.runtime_tuning'
    call=next(n for n in tree.body if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and getattr(n.value.func,'id',None)=='apply_blas_policy')
    assert all(n.lineno>call.lineno for n in imports if n is not first)
