from copy import deepcopy
from types import SimpleNamespace
import time

from src.ui.observation_session import ObservationSession
from critical_path_session_oracle import OriginalSessionMethods
from test_observation_sessions import FakeDetector,FakeANFIS,run_payload


def test_telemetry_equals_original_and_is_mutation_isolated(video_factory,tmp_path,monkeypatch):
    settings=run_payload(video_factory);settings['system']['name']='telemetry-static-name'
    session=ObservationSession(settings,tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        for role in ('system','acquisition'):assert session.jpeg(role)
        with monkeypatch.context() as clock:
            clock.setattr(time,'monotonic',lambda:session.created_monotonic+1)
            old=OriginalSessionMethods.telemetry(session,evidence=True);new=session.telemetry(evidence=True)
            old.update({key:new[key] for key in ("preview_encoded_frames","preview_coalesced_frames","preview_errors")})
            assert new==old
            saved=deepcopy(new)
            new['camera_calibrations']['system']['anchors'][0][0]=-999;new['policy']['movement_id']='bad'
            new['execution']['detectors']['system']['model']['path']='bad'
            new['runtime']['system']['frames']=-1;new['vehicles'].append({'bad':True})
            assert session.telemetry(evidence=True)==saved
        from src.common.source_identity import public_source
        reference=session.calibrations['system'].payload['name']
        public_source('rtsp://u:'+reference+'@test/secret')
        with monkeypatch.context() as clock:
            clock.setattr(time,'monotonic',lambda:session.created_monotonic+1)
            current=session.telemetry();original=OriginalSessionMethods.telemetry(session)
            original.update({key:current[key] for key in ("preview_encoded_frames","preview_coalesced_frames","preview_errors")})
            assert current==original
    finally:session.stop()


def test_static_public_view_cache_rediscovers_and_invalidates():
    from src.common.source_identity import PublicDataCache,public_data,public_source
    cache=PublicDataCache();static={'name':'unique-view-secret','nested':[1,2]}
    value={'static':static,'name':'whole-view-secret','source':'rtsp://u:whole-view-secret@h/x'}
    assert cache.render(value,('static',))==public_data(value)
    result=cache.render(value,('static',));result['static']['nested'].append(3)
    assert cache.render(value,('static',))['static']['nested']==[1,2]
    public_source('rtsp://u:unique-view-secret@h/x')
    assert cache.render(value,('static',))==public_data(value)


def test_execution_sampling_bound_and_control_resample(video_factory,tmp_path):
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session=Manual(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        detector=session.detectors['acquisition']
        backend=SimpleNamespace(device='cpu:a',fp16=False)
        detector.model=SimpleNamespace(predictor=SimpleNamespace(model=backend,args=SimpleNamespace(conf=.28)))
        session._execution_frames=0;session._execution_resample=True
        for i in range(1,4):session._sample_execution_provenance(i/15)
        assert session.execution_provenance()['detectors']['acquisition']['effective_device']=='cpu:a'
        backend.device='cpu:b';backend.fp16=True;detector.model.predictor.args.conf=.5
        for i in range(4,79):session._sample_execution_provenance(i/15)
        execution=session.execution_provenance()['detectors']['acquisition']
        assert execution['effective_device']=='cpu:b' and execution['effective_precision']=='fp16'
        assert execution['effective_settings']['conf']==.5
        backend.device='cpu:c';session.control('play');session._apply_controls();session._sample_execution_provenance(79/15)
        assert session.execution_provenance()['detectors']['acquisition']['effective_device']=='cpu:c'
    finally:session.stop()


def test_static_telemetry_refreshes_on_calibration_revision_and_selection(video_factory,tmp_path,monkeypatch):
    from src.experiment.calibration import CameraCalibration
    session=ObservationSession(run_payload(video_factory),tmp_path/'runs',detector_factory=FakeDetector,anfis_factory=FakeANFIS)
    try:
        for role in ('system','acquisition'):session.jpeg(role)
        session.telemetry()
        data=deepcopy(session.calibrations['system'].payload);data['name']='revised calibration'
        session.calibrations['system']=CameraCalibration(data)
        session.model_selection['backend']='injected-revised'
        with monkeypatch.context() as clock:
            clock.setattr(time,'monotonic',lambda:session.created_monotonic+1)
            value=session.telemetry();expected=OriginalSessionMethods.telemetry(session)
            expected.update({key:value[key] for key in ('preview_encoded_frames','preview_coalesced_frames','preview_errors')})
            assert value==expected
    finally:session.stop()
