from dataclasses import replace
from types import SimpleNamespace
import threading

from src.common.types import TrafficSnapshot
from src.experiment.schedule_episodes import ScheduleEpisodes
from src.experiment.wait_savings import SchedulePolicy


def event(phase, timestamp, previous):
    return {'phase': phase, 'timestamp': timestamp, 'confirmed_at': timestamp, 'from': previous}


def test_one_call_per_a_green_b_binding_and_actual_a_duration():
    calls = []
    book = ScheduleEpisodes('run', SchedulePolicy(enabled=True, unchanged_clearance=True),
        lambda *_: SimpleNamespace(green_seconds=10, failsafe_active=False), lambda: 2.)
    def analyze():
        calls.append(1)
        return TrafficSnapshot(timestamp=1., v_w_pcu=1, occupancy_pct=10, queue_meters=4), {'inference_calls': 1}
    try:
        book.observe_b('RED', 0., 1, event('RED', 0., 'YELLOW'))
        book.observe_b('RED', .9, 1, None)
        book.observe_a(event('GREEN', 1., 'RED'), {'timestamp': 1., 'sequence': 1}, analyze)
        book.observe_a(None, {'timestamp': 1.1, 'sequence': 2}, analyze)
        book.await_idle()
        book.observe_a(event('YELLOW', 21., 'GREEN'), {'timestamp': 21., 'sequence': 3}, analyze)
        book.observe_b('GREEN', 23., 1, event('GREEN', 23., 'RED'))
        book.finalize([], [])
        row = book.view()['episodes'][0]
        assert calls == [1]
        assert row['a_end']-row['a_start'] == 20
        assert row['b_release'] == 23 and row['b_cycle_id'] == 1
        assert row['sealed'] and row['recommended_green_sec'] == 10
    finally:
        book.close('test_end')


def test_sealing_during_inference_blocks_late_publication_and_keeps_resource_owner():
    entered = threading.Event()
    release = threading.Event()
    book = ScheduleEpisodes('run', SchedulePolicy(enabled=True, unchanged_clearance=True), lambda *_: None, lambda: 1.)
    def analyze():
        entered.set()
        assert release.wait(5)
        raise RuntimeError('late native result')
    book.observe_a(event('GREEN', 1., 'RED'), {'timestamp': 1., 'sequence': 1}, analyze)
    assert entered.wait(2)
    book.close('operator_stop')
    assert not book.resources_released
    release.set()
    book.await_idle()
    assert book.resources_released
    row = book.view()['episodes'][0]
    assert row['recommended_green_sec'] is None and row['status'] == 'unavailable'
    assert row['camera_analysis']['inference_calls']==1


def test_b_phase_failure_during_native_call_cannot_be_promoted_back_to_ready():
    entered=threading.Event();release=threading.Event()
    book=ScheduleEpisodes('run',SchedulePolicy(enabled=True,unchanged_clearance=True),
        lambda *_:SimpleNamespace(green_seconds=10,failsafe_active=False),lambda:2.)
    def analyze():
        entered.set();assert release.wait(5)
        return TrafficSnapshot(queue_meters=1),{'inference_calls':1}
    try:
        book.observe_b('RED',.9,1,event('RED',.9,'YELLOW'))
        book.observe_a(event('GREEN',1.,'RED'),{'timestamp':1.,'sequence':1},analyze)
        assert entered.wait(2)
        book.observe_b('UNKNOWN',1.1,1,event('UNKNOWN',1.1,'RED'))
        release.set();book.await_idle();book.finalize([],[])
        row=book.view()['episodes'][0]
        assert row['status']=='unavailable' and row['sealed']
        assert 'b_phase_discontinuity' in row['reasons']
    finally:
        release.set();book.close('test_end')
