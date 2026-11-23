"""Golden comparison against c683cdb, exported without changing the checkout."""
import io,json,os,subprocess,sys,tarfile
from pathlib import Path
import pytest
import cv2
import numpy as np


# The baseline predates tracking_profile, RunPolicy.fragment_continuation, and
# RunPolicy.exit_persistence_sec. Only additive profile metadata and off-default
# RunPolicy fields are dropped, from policy dicts and parsed JSON CSV cells; any
# non-default policy value (such as exit_persistence_sec=0.05) still differs.
def without_additive_defaults(value, in_policy=False):
    if isinstance(value, dict):
        return {
            k: without_additive_defaults(v, in_policy=(k == 'policy'))
            for k, v in value.items()
            if k != 'tracking_profile'
            and not (in_policy and k == 'fragment_continuation' and v is False)
            and not (in_policy and k == 'exit_persistence_sec' and type(v) in (int, float) and v == 0)
        }
    if isinstance(value, list):
        return [without_additive_defaults(v) for v in value]
    return value


def normalize(value):
    return _normalize(value)


def _normalize(value, in_policy=False):
    # Additive profile metadata is checked in test_tracking_profile_sessions;
    # the historical tree predates it. All existing behavior remains compared.
    if isinstance(value, dict):
        return {
            key: _normalize(item, in_policy=(key == 'policy'))
            for key, item in value.items()
            if key not in (
                'run_id', 'commit_sequence', 'source_hashes', 'package_versions',
                'persist_background_requests', 'persist_background_commits',
                'persist_coalesced_requests', 'persist_worker_errors', 'tracking_profile',
            )
            and not key.endswith('_utc')
            and not (in_policy and key == 'fragment_continuation' and item is False)
            and not (in_policy and key == 'exit_persistence_sec' and type(item) in (int, float) and item == 0)
        }
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, str) and any(
        marker in value for marker in ('""tracking_profile""', '""fragment_continuation""', '""exit_persistence_sec""')
    ):
        # CSV provenance embeds JSON; parse each JSON cell and drop only additive defaults.
        import csv
        rows = list(csv.reader(io.StringIO(value)))
        changed = False
        for row in rows:
            for i, cell in enumerate(row):
                if not any(k in cell for k in ('"tracking_profile":', '"fragment_continuation":', '"exit_persistence_sec":')):
                    continue
                try:
                    parsed = json.loads(cell)
                except json.JSONDecodeError:
                    continue
                updated = without_additive_defaults(parsed)
                if updated != parsed:
                    row[i] = json.dumps(updated, allow_nan=False)
                    changed = True
        if changed:
            stream = io.StringIO(newline='')
            csv.writer(stream).writerows(rows)
            return stream.getvalue()
    return value


@pytest.mark.parametrize("persist_mode,blas",[("inline","default"),("inline","auto"),("background","auto")])
def test_scripted_session_saved_exports_telemetry_and_jpeg_bytes_equal_baseline(tmp_path,persist_mode,blas):
    core=Path(__file__).resolve().parents[1];baseline=tmp_path/'baseline';baseline.mkdir()
    archive=subprocess.check_output(['git','archive','c683cdbb2ed937d1b93542c24683b089d2c91451','jetson-atsc-core/src'],cwd=core.parent)
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for member in bundle.getmembers():
            if not member.isfile():continue
            target=baseline/Path(member.name).relative_to('jetson-atsc-core');target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(bundle.extractfile(member).read())
    clip=tmp_path/'fixed.mp4';writer=cv2.VideoWriter(str(clip),cv2.VideoWriter_fourcc(*'mp4v'),15,(320,240));assert writer.isOpened()
    for seq in range(90):
        color=(0,0,255) if seq<30 or seq>=75 else (0,255,0) if seq<60 else (0,255,255)
        frame=np.full((240,320,3),color,np.uint8);cv2.putText(frame,str(seq),(100,40),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);writer.write(frame)
    writer.release();outputs=[]
    for name,tree,mode in [('baseline',baseline,persist_mode),('candidate',core,persist_mode)]+([('inline',core,'inline')] if persist_mode=='background' else []):
        output=tmp_path/(name+'.json');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',PYTHONDONTWRITEBYTECODE='1')
        subprocess.run([sys.executable,str(core/'tests/critical_path_golden_runner.py'),str(tree),str(core),str(clip),str(output),str(tmp_path/(name+'-runs')),mode,blas],cwd=tree,env=env,check=True,capture_output=True,text=True,timeout=120)
        outputs.append(normalize(json.loads(output.read_text())))
    assert all(result==outputs[0] for result in outputs[1:])


def test_normalize_csv_removes_only_additive_profile_metadata():
    import csv
    def table(value):
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['provenance','observed_wait']);writer.writerow([json.dumps(value),'3.500'])
        return stream.getvalue()
    before={'execution':{'detector':{}},'phase':'RED','wait':3.5}
    after={'execution':{'detector':{'tracking_profile':{'name':'low_score'}}},'phase':'RED','wait':3.5}
    assert normalize(table(after))==normalize(table(before))
    after['wait']=4.5
    assert normalize(table(after))!=normalize(table(before))
    assert normalize(table(before))==table(before)


def test_normalize_csv_removes_only_exact_zero_run_policy_defaults():
    import csv
    def table(value):
        stream=io.StringIO(newline='');writer=csv.writer(stream)
        writer.writerow(['provenance','observed_wait']);writer.writerow([json.dumps(value),'3.500'])
        return stream.getvalue()
    base={'policy':{'max_gap_sec':1},'vehicle_identity':{'exit_persistence_sec':0.0}}
    with_defaults={'policy':{'max_gap_sec':1,'fragment_continuation':False,'exit_persistence_sec':0.0},
                   'vehicle_identity':{'exit_persistence_sec':0.0}}
    with_prefix={'policy':{'max_gap_sec':1,'exit_persistence_sec':0.05},
                 'vehicle_identity':{'exit_persistence_sec':0.0}}
    with_other={'policy':{'max_gap_sec':15},
                'vehicle_identity':{'exit_persistence_sec':0.0}}
    with_enabled={'policy':{'max_gap_sec':1,'fragment_continuation':True},
                  'vehicle_identity':{'exit_persistence_sec':0.0}}
    assert normalize(table(with_defaults))==normalize(table(base))
    assert normalize(table(with_prefix))!=normalize(table(base))
    assert normalize(table(with_prefix))!=normalize(table(with_other))
    assert normalize(table(with_enabled))!=normalize(table(base))
    # Non-policy exit_persistence_sec=0.0 is preserved intact.
    assert '"vehicle_identity": {"exit_persistence_sec": 0.0}' in list(csv.reader(io.StringIO(normalize(table(with_defaults)))))[1][0]
