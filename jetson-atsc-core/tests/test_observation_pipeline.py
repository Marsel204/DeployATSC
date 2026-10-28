import csv,io,json,zipfile,time
import cv2,numpy as np
import pytest
from src.perception.detector_tracker import TrackedVehicle
from src.control.signal_interface import SignalInterface
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeANFIS


class KnownVehicleDetector:
    def __init__(self,*args,**kwargs):self.epoch=0
    def track(self,frames,timestamp=None):
        stopped=10<=timestamp<34
        bottom=20 if timestamp<40 else 35
        vehicle=TrackedVehicle(7,1,"car",[15,bottom-10,25,bottom],1,timestamp)
        vehicle.velocity_mps=0 if stopped else 2
        vehicle.velocity_valid=True
        return [[vehicle]]
    def update_velocities(self,*args,**kwargs):pass
    def reset_tracking(self):self.epoch+=1


def make_clip(path,fps,signal=False):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(64,48))
    assert writer.isOpened()
    for index in range(fps*60):
        t=index/fps
        color=((0,0,255) if t<30 else (0,255,0) if t<55 else (0,255,255)) if signal else (40,40,40)
        writer.write(np.full((48,64,3),color,np.uint8))
    writer.release()


def exported_csv_rows(archive,table):
    text=archive.read(table+'.csv').decode()
    previous=csv.field_size_limit()
    try:
        # Detailed interval arrays are intentionally retained in nested CSV JSON.
        csv.field_size_limit(max(previous,len(text.encode())))
        return list(csv.DictReader(io.StringIO(text)))
    finally:csv.field_size_limit(previous)


def test_paired_video_pipeline_timing_waits_exports_and_no_hardware(tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Observation invoked cabinet hardware')
    for method in ('__init__','from_config','start_watchdog','dispatch_green_actuation','_send_heartbeat'):
        monkeypatch.setattr(SignalInterface,method,forbidden)
    system=tmp_path/'system.mp4';acquisition=tmp_path/'acquisition.mp4'
    make_clip(system,25);make_clip(acquisition,30,True)
    def calibration(role,source):
        return {'role':role,'source':str(source),'image_size':[64,48],
                'anchors':[[3,3],[60,3],[60,44],[3,44]],'polygon':[[3,3],[60,3],[60,44],[3,44]],
                'scale_evidence':{'method':'measured_dimensions','reference':'synthetic 8 by 40 metre fixture'},'binding':{'reviewed':True},
                'road_width_m':8,'road_length_m':40,'signal_roi':{'xmin':0,'ymin':0,'xmax':64,'ymax':48},
                'stop_line':[[3,30],[60,30]],'crossing_direction':1}
    session=ObservationSession({'system':calibration('system',system),'acquisition':calibration('acquisition',acquisition),
        'step_mode':True,'policy':{'debounce_frames':3}},tmp_path/'runs',detector_factory=KnownVehicleDetector,anfis_factory=FakeANFIS)
    captured=False
    while not session.pair.ended:
        control=session.control("step")
        deadline=time.monotonic()+2
        while time.monotonic()<deadline:
            with session.lock:
                done=session.applied_control_seq>=control["control_seq"] and not session.steps and not session.packets
            if done or session.pair.ended:break
            time.sleep(.001)
        assert done or session.pair.ended, "Worker did not drain the replay step"
        if not captured and session.engine.time>=29:
            session.predict('before-green');session.engine.predictions.await_idle();captured=True
    session.stop()
    data=session.telemetry()
    cycle=data['comparisons'][0];vehicle=data['vehicles'][0]
    assert cycle['actual_green_sec']==pytest.approx(25,abs=.04)
    assert cycle['recommended_green_sec']==35 and cycle['difference_sec']==pytest.approx(10,abs=.04)
    assert cycle['eligible']
    # On-demand Camera 1: one still image has no motion evidence, so its spatial queue
    # proxy keeps the timing-eligible comparison diagnostic (never qualified).
    assert not cycle['qualified'] and cycle['qualification']['policy']=='motion-phase-scale-v2'
    assert cycle['qualification']['state']=='diagnostic'
    assert cycle['qualification']['reasons']==['Q_measurement_single_frame_spatial_proxy']
    assert data['summary']['qualified_cycle_comparisons']==0 and data['summary']['valid_cycle_comparisons']==1
    assert vehicle['red_wait_sec']==pytest.approx(20,abs=.04)
    assert vehicle['total_stopped_sec']==pytest.approx(24,abs=.04)
    assert vehicle['crossed_stop_line'] and cycle['n_served']==1
    assert data['los']['grade'] is None
    assert not session.pair.system.cap.isOpened() and not session.pair.acquisition.cap.isOpened()
    with zipfile.ZipFile(io.BytesIO(session.export())) as archive:
        evidence=json.loads(archive.read('evidence.json'))
        assert evidence['schema_version']==2
        assert json.loads(archive.read('manifest.json'))['schema_version']==2
        assert evidence['snapshot_revision']==data['snapshot_revision']
        assert not evidence['predictions'][0]['snapshot']['quality']['Q']['complete']
        assert evidence['predictions'][0]['snapshot']['quality']['Q']['state']=='single_frame_spatial_proxy'
        assert evidence['predictions'][0]['metric_method']=='camera1_single_frame_spatial_proxy'
        assert evidence['predictions'][0]['camera_analysis']['inference_calls']==1
        assert evidence['comparisons'][0]['eligible'] and not evidence['comparisons'][0]['qualified']
        assert evidence['vehicles'][0]['intervals'] and evidence['phase_events']
        assert evidence['predictions'][0]['snapshot']['timestamp']<cycle['green_start']
        prediction=evidence['predictions'][0];comparison=evidence['comparisons'][0];track=evidence['vehicles'][0]
        assert prediction['request_id']==comparison['prediction']['request_id']=='before-green'
        assert prediction['cycle_id']==comparison['cycle_id']==track['crossing_cycle']
        assert track['vehicle_id'].startswith(session.run_id+':acquisition:')
        assert evidence['camera_calibrations']['system']['source']==str(system)
        assert evidence['camera_calibrations']['acquisition']['source']==str(acquisition)
        assert prediction['qualification']['policy']=='motion-phase-scale-v2'
        assert prediction['provenance']['clock_provenance']['system']['last_processed']['mapped_timestamp']==prediction['snapshot']['timestamp']
        for table,identity in [('predictions','request_id'),('comparisons','cycle_id'),('vehicles','vehicle_id')]:
            rows=exported_csv_rows(archive,table)
            assert [row[identity] for row in rows]==[str(row[identity]) for row in evidence[table]]


@pytest.mark.parametrize('valid,stable',[(False,True),(True,False),(True,True)])
def test_real_session_wires_strict_quality_and_run_gap_policy(video_factory,tmp_path,valid,stable):
    from test_observation_sessions import run_payload
    class Detector(KnownVehicleDetector):
        def track(self,frames,timestamp=None):
            tracks=super().track(frames,timestamp)
            for v in tracks[0]:
                v.velocity_valid=valid;v.stable_id=stable
                v.bottom_center=(160,120);v.bbox=[150,110,170,120]
            return tracks
    payload=run_payload(video_factory);payload['policy']['max_gap_sec']=.25
    session=ObservationSession(payload,tmp_path/'runs',detector_factory=Detector,anfis_factory=FakeANFIS)
    try:
        data=session.telemetry()
        assert all(d.max_gap_sec==.25 for d in session.detectors.values())
        assert session.metrics.strict_motion
        assert data['metrics'] is None  # Camera 1 metrics exist only after an explicit capture
        session.predict('quality-click');session.engine.predictions.await_idle()
        record=session.engine.predictions.records['quality-click']
        quality=record['snapshot']['quality']['Q']
        # Detector-supplied motion flags are never inherited by a single still image.
        assert quality['state']=='single_frame_spatial_proxy' and quality['complete'] is False
        assert quality['motion_valid_vehicle_count']==0 and record['snapshot']['queue_meters']>0
        assert session.telemetry()['metrics']['quality']['Q']==quality
    finally:session.stop()
    # Durability/export must retain adverse quality rather than inventing zero Q
    # or promoting a diagnostic capture when the session is stopped.
    with zipfile.ZipFile(io.BytesIO(session.export())) as archive:
        evidence=json.loads(archive.read('evidence.json'))
        saved=evidence['predictions'][0]
        assert saved['request_id']=='quality-click'
        assert saved['snapshot']['quality']['Q']==quality
        assert saved['snapshot']['queue_meters']==record['snapshot']['queue_meters']
        assert not saved['qualified'] and saved['qualification']['state']=='diagnostic' # this existing fixture has unreviewed scale
        assert saved['qualification']['scale']=='unverified'
        assert saved['qualification']['policy']=='motion-phase-scale-v2'
        rows=exported_csv_rows(archive,'predictions')
        assert rows[0]['request_id']==saved['request_id']
        assert json.loads(rows[0]['snapshot'])==saved['snapshot']


@pytest.mark.parametrize('phase',['UNKNOWN','GREEN'])
def test_export_preserves_measurement_crossing_attribution(tmp_path,phase):
    from src.experiment.contracts import VehicleObservation
    from src.experiment.export import ExperimentStore
    from test_observation_engine import make_engine
    engine=make_engine()
    try:
        colors={'GREEN':(0,255,0),'YELLOW':(0,255,255),'UNKNOWN':(0,0,0)}
        for seq,(t,state,y) in enumerate([(0,'GREEN',10),(.1,phase,25),(.2,'GREEN',30),(.3,'YELLOW',30)],1):
            engine.observe_acquisition(np.full((20,20,3),colors[state],np.uint8),t,seq,
                [VehicleObservation(7,'car',2,(10,y),velocity_valid=True)])
        engine.close()
        store=ExperimentStore(tmp_path,engine.run_id,{'operation':'observation'})
        store.persist(engine.snapshot())
        with zipfile.ZipFile(io.BytesIO(store.export())) as archive:
            evidence=json.loads(archive.read('evidence.json'))
            track=evidence['vehicles'][0]
            assert track['crossing_phase']==phase
            assert track['crossing_cycle']==(None if phase=='UNKNOWN' else 1)
            assert sum(c['n_served'] for c in evidence['comparisons'])==(0 if phase=='UNKNOWN' else 1)
            assert ('crossing_phase_unknown' in track['quality_flags'])==(phase=='UNKNOWN')
            rows=exported_csv_rows(archive,'vehicles')
            assert rows[0]['vehicle_id']==track['vehicle_id']
            assert rows[0]['crossing_cycle']==('' if phase=='UNKNOWN' else '1')
            assert rows[0]['crossing_phase']==phase
    finally:engine.close()
