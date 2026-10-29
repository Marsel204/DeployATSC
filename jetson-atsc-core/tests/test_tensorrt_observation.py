"""Backend integrity and preview/measurement isolation, without native GPU resources."""
from functools import partial
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import numpy as np
import pytest


def models(tmp_path):
    root=tmp_path/'models';root.mkdir()
    (root/'Final.pt').write_bytes(b'original weights')
    (root/'Final_fp16.engine').write_bytes(b'tested engine')
    metadata={'source_sha256':hashlib.sha256(b'original weights').hexdigest(),
              'engine_sha256':hashlib.sha256(b'tested engine').hexdigest(),
              'tensorrt_version':'test', 'compute_capability':[8,7],
              'batch':1,'imgsz':640,'precision':'fp16'}
    (root/'Final_fp16.engine.json').write_text(json.dumps(metadata))
    return root


def selector(monkeypatch):
    from src.perception import observation_model as module
    monkeypatch.setattr(module,'runtime_error',lambda metadata,device:None)
    monkeypatch.delenv('ATSC_DETECTOR_BACKEND',raising=False)
    return module


def test_valid_engine_selected_and_original_remains_available(tmp_path,monkeypatch):
    module=selector(monkeypatch);root=models(tmp_path)
    selected=module.select_model(root)
    assert selected['path']==str(root/'Final_fp16.engine')
    assert selected['backend']=='tensorrt'
    assert module.select_model(root,backend='pytorch')['path']==str(root/'Final.pt')
    assert module.select_model(root,device='cpu')['backend']=='pytorch'


@pytest.mark.parametrize('file',['Final.pt','Final_fp16.engine'])
def test_stale_or_modified_engine_binding_is_rejected(tmp_path,monkeypatch,file):
    module=selector(monkeypatch);root=models(tmp_path)
    (root/file).write_bytes(b'different')
    assert module.select_model(root)['backend']=='pytorch'
    with pytest.raises(ValueError,match='hash'):module.select_model(root,backend='tensorrt')


def test_incompatible_runtime_and_missing_engine_have_explicit_fallback(tmp_path,monkeypatch):
    module=selector(monkeypatch);root=models(tmp_path)
    monkeypatch.setattr(module,'runtime_error',lambda metadata,device:'different TensorRT runtime')
    choice=module.select_model(root)
    assert choice['backend']=='pytorch' and 'TensorRT' in choice['reason']
    with pytest.raises(ValueError,match='runtime'):module.select_model(root,backend='tensorrt')
    (root/'Final_fp16.engine').unlink()
    assert module.select_model(root)['backend']=='pytorch'
    with pytest.raises(ValueError):module.select_model(root,backend='tensorrt')


def test_partial_native_factory_selects_engine_but_injected_factory_stays_isolated(tmp_path,monkeypatch):
    module=selector(monkeypatch);root=models(tmp_path)
    from src.perception.detector_tracker import DetectorTracker
    assert module.model_for_factory(root,partial(DetectorTracker,device='cuda:0'))['backend']=='tensorrt'
    assert module.model_for_factory(root,partial(DetectorTracker,device='cpu'))['backend']=='pytorch'
    assert module.model_for_factory(root,lambda *a,**kw:None)['backend']=='injected'


def test_single_frame_engine_guard_and_device_selection(monkeypatch):
    from src.perception.detector_tracker import DetectorTracker
    monkeypatch.setattr(DetectorTracker,'_load_model',lambda self:None)
    tracker=DetectorTracker('test.engine',device='cuda:0')
    calls=[]
    tracker.model=SimpleNamespace(track=lambda **kw:calls.append(kw) or [])
    frame=np.zeros((20,20,3),np.uint8)
    with pytest.raises(ValueError,match='one frame'):tracker.track([frame,frame])
    assert not calls
    tracker.track([frame])
    assert calls[0]['device']=='cuda:0'


def test_throttled_camera1_preview_does_not_throttle_capture_frame(monkeypatch,request):
    import src.ui.observation_session as module
    from src.ui.observation_session import ObservationSession
    session=ObservationSession.__new__(ObservationSession)
    session.lock=threading.RLock();session.applied_state='running'
    session.jpeg_frames={};session.clean_frames={};session.clean_jpeg_cache={};session.frame_times={}
    session.preview_width=640;session.preview_published_at={};session.storage_suspended=True
    session.runtime={'system':{'preview_frames':0}}
    session.engine=SimpleNamespace(observe_system_frame=lambda *a,**kw:True)
    session._render=lambda role,frame,tracks:frame.copy()
    session.run_id='throttled-preview';session._start_previews();request.addfinalizer(session._stop_previews)
    clock=[0.];monkeypatch.setattr(module.time,'monotonic',lambda:clock[0])
    encodes=[];original=module.cv2.imencode
    monkeypatch.setattr(module.cv2,'imencode',lambda *a,**kw:encodes.append(a[1].shape) or original(*a,**kw))
    for seq in range(1,7):
        clock[0]=seq/1000
        packet=SimpleNamespace(role='system',frame=np.full((540,960,3),seq,np.uint8),
            sequence=seq,timestamp=seq/30,source_timestamp=seq/30,timestamp_origin='presentation_time')
        session._observe_camera1(packet,packet.frame,0)
    with session.preview_condition:assert session.preview_condition.wait_for(lambda:session.preview_completed.get('system',0)>=1,timeout=2)
    assert len(encodes)==1
    assert encodes[0]==(360,640,3)
    assert session.camera1_latest['sequence']==6
    assert session.camera1_latest['frame'][0,0,0]==6
    assert session.runtime['system']['preview_frames']==6
    # Calibration continues to read the latest full-size clean image.
    clean=module.cv2.imdecode(np.frombuffer(session.jpeg('system',clean=True),np.uint8),1)
    assert clean.shape==(540,960,3) and abs(int(clean[0,0,0])-6)<=1


def test_real_session_binds_engine_and_weights_without_background_camera1_inference(tmp_path,monkeypatch,video_factory):
    import src.ui.observation_session as module
    import src.perception.detector_tracker as detector_module
    from src.perception.detector_tracker import DetectorTracker
    from src.ui.observation_session import ObservationSession
    from test_observation_sessions import FakeANFIS,run_payload
    selector(monkeypatch)
    root=models(tmp_path)
    (root/'anfis_sugeno.onnx').write_bytes(b'fake anfis')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(detector_module,'_still_predictor_type',lambda:object)
    made=[]
    def load(self):
        calls=[]
        self.model=SimpleNamespace(names={0:'motorcycle',1:'car',2:'bus',3:'truck'},
            predictor=SimpleNamespace(device='synthetic:cuda',model=SimpleNamespace(fp16=False),args=SimpleNamespace(half=False)),
            track=lambda **kw:calls.append('track') or [SimpleNamespace(boxes=None)],
            predict=lambda **kw:calls.append('predict') or [SimpleNamespace(boxes=None)])
        made.append(calls)
    monkeypatch.setattr(DetectorTracker,'_load_model',load)
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',anfis_factory=FakeANFIS)
    try:
        assert made==[[],['track']]
        manifest=json.loads((session.store.directory/'manifest.json').read_text())
        assert manifest['model_hashes']['Final_fp16.engine']==hashlib.sha256(b'tested engine').hexdigest()
        assert manifest['model_hashes']['Final.pt']==hashlib.sha256(b'original weights').hexdigest()
        assert manifest['detector_selection']['backend']=='tensorrt'
        precision=manifest['execution']['detectors']['acquisition']
        assert precision['effective_precision']=='fp32' and precision['precision_scope']=='input_tensor'
        assert precision['engine_build_precision']=='fp16_enabled'
        session.predict('one-frame');session.engine.predictions.await_idle()
        assert made==[['predict'],['track']]
        assert session.telemetry()['predictions'][0]['camera_analysis']['inference_calls']==1
    finally:session.stop()
