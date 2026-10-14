import pytest
from src.experiment.sources import ReplayPair


def test_different_fps_and_offset_consume_each_frame_once(video_factory):
    system=video_factory("system.mp4",fps=25)
    acquisition=video_factory("acquisition.mp4",size=(160,120),fps=30)
    pair=ReplayPair(str(system),str(acquisition),offset_sec=.2)
    seen={"system":[],"acquisition":[]}
    last_system_time=0
    while not pair.ended:
        for packet in pair.step():
            if packet.role=="system":last_system_time=packet.timestamp
            else:assert 0<=packet.timestamp<=last_system_time+1e-6
            seen[packet.role].append(packet.sequence)
    assert len(seen["system"])==50
    assert seen["system"]==sorted(set(seen["system"]))
    assert seen["acquisition"]==sorted(set(seen["acquisition"]))
    assert seen["acquisition"][0]==7
    pair.close();assert not pair.system.cap.isOpened()


def test_early_acquisition_eof_keeps_system_replay(video_factory):
    system=video_factory("system.mp4",fps=10,seconds=2)
    acquisition=video_factory("acquisition.mp4",fps=10,seconds=.5)
    pair=ReplayPair(str(system),str(acquisition))
    system_count=0
    while not pair.ended:
        system_count+=sum(p.role=="system" for p in pair.step())
    assert system_count==20 and pair.acquisition.ended
    pair.close()


def test_startup_failure_is_atomic(video_factory,monkeypatch):
    import src.experiment.sources as sources
    original=sources.FileSource
    opened=[]
    class TrackedSource(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);opened.append(self)
    monkeypatch.setattr(sources,"FileSource",TrackedSource)
    with pytest.raises(ValueError):ReplayPair(str(video_factory()),"missing-file.mp4")
    assert len(opened)==1 and not opened[0].cap.isOpened()
