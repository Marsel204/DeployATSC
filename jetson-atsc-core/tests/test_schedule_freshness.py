import threading
import time
from types import SimpleNamespace

from src.experiment.sources import FramePacket
from src.ui.observation_session import ObservationSession
from test_schedule_pipeline import make_engine,frame
from test_vehicle_identity import detection


def test_stale_a_frame_is_preview_only_and_never_starts_neural_analysis():
    book=make_engine()
    session=ObservationSession.__new__(ObservationSession)
    session.engine=book;session.lock=threading.RLock();session.policy=book.policy
    session.source_mode='live';session.pair=SimpleNamespace(origin=time.monotonic()-5)
    session.runtime={'system':{'preview_frames':0}}
    session.calibrations={'system':SimpleNamespace(revision='test')}
    session._publish_frame=lambda *args:None
    def forbidden(*args):raise AssertionError('Stale A image must not be analysed')
    session._analyze_schedule_frame=forbidden
    try:
        packet=FramePacket('system',1,1.,1.,frame('GREEN'))
        session._observe_camera1(packet,packet.frame,0)
        assert not book.schedule.records
        assert str(book.system_phase.measurement_phase)=='UNKNOWN'
        assert session.runtime['system']['preview_frames']==1
    finally:book.close()


def test_a_only_discontinuity_does_not_reset_b_vehicle_timer():
    book=make_engine()
    try:
        for seq in range(1,11):book.observe_acquisition(frame('RED'),(seq-1)/10,seq,(),identity_evidence=(detection(),))
        old=next(iter(book.waits.records.values()));identity=old['vehicle_id'];wait=old['red_wait_sec']
        book.invalidate_a('a_source_stale')
        for seq in range(11,16):book.observe_acquisition(frame('RED'),(seq-1)/10,seq,(),identity_evidence=(detection(),))
        assert len(book.waits.records)==1
        assert next(iter(book.waits.records.values()))['vehicle_id']==identity
        assert next(iter(book.waits.records.values()))['red_wait_sec']>wait
    finally:book.close()
