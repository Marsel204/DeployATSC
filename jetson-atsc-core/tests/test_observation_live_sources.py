from collections import deque
import threading
import numpy as np
from src.perception.frame_grabber import FrameGrabber


def test_live_packet_sequence_is_consumed_once():
    source=FrameGrabber.__new__(FrameGrabber)
    source._lock=threading.Lock()
    source._observation_queue=deque([{"sequence":1,"timestamp":10.,"frame":np.zeros((20,30,3),np.uint8)}])
    first=source.read_observation()
    assert first["sequence"]==1
    assert source.read_observation() is None


class BlockingCapture:
    def __init__(self,opened=True):
        self.opened=opened;self.entered=threading.Event();self.unblock=threading.Event();self.releases=[]
    def isOpened(self):return self.opened
    def set(self,*args):pass
    def read(self):
        self.entered.set();assert self.unblock.wait(5)
        return False,None
    def release(self):self.releases.append(threading.current_thread().name);self.opened=False


def make_grabber(monkeypatch,cap):
    monkeypatch.setattr(FrameGrabber,'_build_gstreamer_pipeline',lambda self:None)
    monkeypatch.setattr('src.perception.frame_grabber.cv2.VideoCapture',lambda *a:cap)
    return FrameGrabber('fake-device')


def test_blocked_read_releases_only_from_capture_owner_after_exit(monkeypatch):
    cap=BlockingCapture();grabber=make_grabber(monkeypatch,cap).start()
    try:
        assert cap.entered.wait(1)
        grabber.stop(timeout=.01)
        assert cap.releases==[]
        assert not grabber.resources_released
        assert grabber.observation_health()['release_pending']
    finally:
        cap.unblock.set();grabber.stop()
    assert cap.releases==['Grabber-Camera']
    assert grabber.resources_released
    grabber.stop();assert len(cap.releases)==1


def test_unstarted_and_failed_start_captures_release_once(monkeypatch):
    import pytest
    for opened in (True,False):
        cap=BlockingCapture(opened);grabber=make_grabber(monkeypatch,cap)
        grabber.stop();grabber.stop()
        assert len(cap.releases)==1
        assert grabber.resources_released
    cap=BlockingCapture();grabber=make_grabber(monkeypatch,cap)
    monkeypatch.setattr(threading.Thread,'start',lambda self:(_ for _ in ()).throw(RuntimeError('thread failed')))
    with pytest.raises(RuntimeError,match='thread failed'):grabber.start()
    grabber.stop();assert len(cap.releases)==1


def test_live_pair_start_failure_closes_both_unstarted_captures(monkeypatch):
    import pytest
    import src.experiment.sources as sources
    made=[]
    class Grabber:
        def __init__(self,*a,**kw):self.stops=0;made.append(self)
        def is_opened(self):return True
        def start(self):raise RuntimeError('cannot start')
        def stop(self,**kw):self.stops+=1
    monkeypatch.setattr(sources,'FrameGrabber',Grabber)
    with pytest.raises(RuntimeError,match='cannot start'):sources.LivePair('a','b')
    assert [x.stops for x in made]==[1,1]


def test_failed_fallback_does_not_release_gstreamer_capture_twice(monkeypatch):
    import pytest
    cap=BlockingCapture(opened=False);calls=[]
    def open_capture(*args):
        calls.append(True)
        if len(calls)==1:return cap
        raise RuntimeError('fallback initialization failed')
    monkeypatch.setattr('src.perception.frame_grabber.cv2.VideoCapture',open_capture)
    with pytest.raises(RuntimeError,match='fallback initialization failed'):FrameGrabber('fake-device')
    assert len(cap.releases)==1


def test_file_source_metadata_failure_releases_capture_once(tmp_path,monkeypatch):
    import pytest
    from src.experiment.sources import FileSource
    source=tmp_path/'fake.mp4';source.write_bytes(b'fake')
    cap=BlockingCapture()
    def bad_metadata(*a):raise RuntimeError('metadata failed')
    cap.get=bad_metadata
    monkeypatch.setattr('src.experiment.sources.cv2.VideoCapture',lambda *a:cap)
    with pytest.raises(RuntimeError,match='metadata failed'):FileSource(source,'system')
    assert len(cap.releases)==1


def test_partial_live_pair_failure_health_preserves_last_known_owner(monkeypatch):
    import pytest
    import src.experiment.sources as sources
    class Grabber:
        def __init__(self,*a,**kw):self.resources_released=False;self.stops=0
        def is_opened(self):return False
        def stop(self,**kw):self.stops+=1;self.resources_released=True
        def observation_health(self):return {'state':'released','resources_released':self.resources_released}
    monkeypatch.setattr(sources,'FrameGrabber',Grabber)
    with pytest.raises(ValueError,match='system camera') as failure:sources.LivePair('a','b')
    pair=failure.value.resource_owner
    assert pair.system.stops==1 and pair.resources_released
    assert pair.health()['system']['resources_released']
    assert pair.health()['acquisition']['state']=='not_started'


# Phase B corrections: one failed release never skips the other source; failures stay retryable.
class FailingGrabber:
    def __init__(self,*a,**kw):self.stops=0;self.failures=0;self.resources_released=False;self.opened=True
    def is_opened(self):return self.opened
    def start(self):return self
    def stop(self,**kw):
        if self.resources_released:return
        if self.failures:self.failures-=1;raise OSError('native capture release failed')
        self.stops+=1;self.resources_released=True


def test_replay_pair_close_attempts_both_sources_and_is_retryable(video_factory):
    import pytest
    from src.experiment.sources import ReplayPair
    pair=ReplayPair(video_factory('a.mp4'),video_factory('b.mp4'))
    failing=[True];released=[]
    class Cap:
        def __init__(self,role,cap):self.role=role;self.cap=cap
        def release(self):
            if failing[0] and self.role=='system':raise OSError('native capture release failed')
            released.append(self.role);self.cap.release()
    pair.system.cap=Cap('system',pair.system.cap);pair.acquisition.cap=Cap('acquisition',pair.acquisition.cap)
    with pytest.raises(OSError,match='release failed'):pair.close()
    assert released==['acquisition'] and pair.acquisition.resources_released and not pair.resources_released
    failing[0]=False;pair.close();pair.close()
    assert released==['acquisition','system'] and pair.resources_released


def test_live_pair_close_and_startup_failure_attempt_every_source(monkeypatch):
    import pytest
    import src.experiment.sources as sources
    made=[]
    def grabber(*a,**kw):made.append(FailingGrabber());return made[-1]
    monkeypatch.setattr(sources,'FrameGrabber',grabber)
    pair=sources.LivePair('a','b');made[0].failures=1
    with pytest.raises(OSError,match='release failed'):pair.close()
    assert [g.stops for g in made]==[0,1] and not pair.resources_released and pair.ended
    pair.close();assert [g.stops for g in made]==[1,1] and pair.resources_released
    made.clear()
    def unopened_acquisition(*a,**kw):
        made.append(FailingGrabber());made[-1].opened=len(made)==1;made[0].failures=1;return made[-1]
    monkeypatch.setattr(sources,'FrameGrabber',unopened_acquisition)
    with pytest.raises(ValueError,match='acquisition camera') as failure:sources.LivePair('a','b')
    owner=failure.value.resource_owner
    assert [g.stops for g in made]==[0,1] and not owner.resources_released
    owner.close();assert owner.resources_released and [g.stops for g in made]==[1,1]


def test_capture_release_failure_after_reader_exit_is_retried_by_stop(monkeypatch):
    import pytest
    hooked=[];monkeypatch.setattr(threading,'excepthook',hooked.append)
    cap=BlockingCapture();cap.unblock.set();failing=[True];release=cap.release
    def failing_release():
        if failing[0]:raise OSError('native capture release failed')
        release()
    cap.release=failing_release
    grabber=make_grabber(monkeypatch,cap).start()
    assert cap.entered.wait(1)
    with pytest.raises(OSError,match='release failed'):grabber.stop(timeout=1)
    assert not grabber.resources_released and not hooked
    failing[0]=False;grabber.stop();grabber.stop()
    assert grabber.resources_released and len(cap.releases)==1
