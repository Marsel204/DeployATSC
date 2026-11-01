"""Golden comparison against c683cdb, exported without changing the checkout."""
import io,json,os,subprocess,sys,tarfile
from pathlib import Path
import pytest
import cv2
import numpy as np


def without_tracking_profile(value):
    if isinstance(value,dict):return {k:without_tracking_profile(v) for k,v in value.items() if k!='tracking_profile'}
    if isinstance(value,list):return [without_tracking_profile(v) for v in value]
    return value


def normalize(value):
    # Additive profile metadata is checked in test_tracking_profile_sessions;
    # the historical tree predates it. All existing behavior remains compared.
    if isinstance(value,dict):return {key:normalize(item) for key,item in value.items() if key not in ('run_id','commit_sequence','source_hashes','package_versions','persist_background_requests','persist_background_commits','persist_coalesced_requests','persist_worker_errors','tracking_profile') and not key.endswith('_utc')}
    if isinstance(value,list):return [normalize(item) for item in value]
    if isinstance(value,str) and '""tracking_profile""' in value:
        # CSV provenance embeds JSON; keep every other cell/value and the format.
        import csv
        rows=list(csv.reader(io.StringIO(value)))
        for row in rows:
            for i,cell in enumerate(row):
                if '"tracking_profile":' not in cell:continue
                row[i]=json.dumps(without_tracking_profile(json.loads(cell)),allow_nan=False)
        stream=io.StringIO(newline='');csv.writer(stream).writerows(rows)
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
