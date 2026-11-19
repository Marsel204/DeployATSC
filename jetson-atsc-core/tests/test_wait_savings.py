from copy import deepcopy

import pytest

from src.experiment.wait_savings import SchedulePolicy, vehicle_savings


def episode(**kwargs):
    return dict(episode_id='A1', b_episode_id='B1', b_cycle_id=1, a_start=0., a_end=10.,
        b_red_start=0., b_release=12., recommended_green_sec=5., completed_at=1.,
        captured_at=0., a_start_complete=True, b_start_complete=True,
        unchanged_clearance=True, reasons=[], **kwargs)


def vehicle(start=0., estimated=False):
    interval = {'start': start, 'end': 12., 'phase': 'RED', 'cycle_id': 1,
        'valid': not estimated, 'stopped': True}
    if estimated:
        interval['gap_evidence'] = {'status': 'accepted', 'red_slices': [
            {'start': start, 'end': 12., 'cycle_id': 1}]}
    return {'vehicle_id': 'V1', 'red_wait_sec': 0. if estimated else 12.-start,
        'estimated_red_wait_sec': 12.-start if estimated else 0.,
        'wait_by_cycle': {'1': {'red_wait_sec': 0. if estimated else 12.-start,
            'estimated_red_wait_sec': 12.-start if estimated else 0.}}, 'intervals': [interval]}


def test_owner_schedule_and_late_arrival_are_per_vehicle_intersections():
    early = vehicle_savings(vehicle(), episode())
    late = vehicle_savings(vehicle(9), episode())
    assert early['available'] and early['potential_saving_sec'] == pytest.approx(5)
    assert early['estimated_schedule_red_stop_sec'] == pytest.approx(7)
    assert late['potential_saving_sec'] == pytest.approx(3)
    assert late['estimated_schedule_red_stop_sec'] == 0


def test_estimated_wait_stays_separate_from_confirmed_saving():
    result = vehicle_savings(vehicle(9, estimated=True), episode())
    assert result['confirmed_potential_saving_sec'] == 0
    assert result['estimated_potential_saving_sec'] == pytest.approx(3)


def test_equal_duration_has_zero_saving_and_late_completion_is_retrospective():
    item = episode()
    item['recommended_green_sec'] = 10.
    assert vehicle_savings(vehicle(), item)['potential_saving_sec'] == 0
    item['recommended_green_sec'] = 5.
    item['completed_at'] = 8.
    result = vehicle_savings(vehicle(), item)
    assert result['available'] and not result['online_eligible']


def test_recommendation_deadline_is_a_proposed_end_before_clearance():
    item=episode();item['completed_at']=6.
    # A should end at 5; B would release at 7 after two seconds clearance.
    result=vehicle_savings(vehicle(),item)
    assert result['available'] and not result['online_eligible']
    assert result['action_deadline']==5.


@pytest.mark.parametrize('field,value', [('recommended_green_sec', 15),
    ('a_start_complete', False), ('b_start_complete', False),
    ('unchanged_clearance', False), ('b_release', None)])
def test_unsupported_schedule_is_unavailable(field, value):
    item = episode()
    item[field] = value
    result = vehicle_savings(vehicle(), item)
    assert not result['available'] and result['potential_saving_sec'] is None


def test_schedule_policy_requires_explicit_controller_assumption():
    with pytest.raises(ValueError):
        SchedulePolicy(enabled=True)
    with pytest.raises(ValueError):
        SchedulePolicy(enabled=True, unchanged_clearance=True, a_movement_id='same', b_movement_id='same')


from src.experiment.wait_savings import collection_eligibility


def tracked_row(*spans, crossing_at=None):
    intervals = []
    for start, end in spans:
        t = start
        while t < end-1e-9:
            intervals.append({'start': t, 'end': min(end, t+.1), 'tracked': True})
            t += .1
    for (_, a), (b, _) in zip(spans, spans[1:]):
        intervals.append({'start': a, 'end': b, 'tracked': False})
    return {'vehicle_id': 'V', 'intervals': intervals, 'crossed_stop_line': crossing_at is not None,
            'crossing_at': crossing_at}


@pytest.mark.parametrize('spans,crossing_at,collected,reason', [
    (((-5., 13.),), None, True, None),                 # already queued, tracked through B release
    (((.8, 13.),), None, True, None),                  # first seen inside the 1 s tolerance
    (((1.5, 13.),), None, False, 'not_detected_at_cycle_start'),
    (((-5., -1.),), None, False, 'not_detected_at_cycle_start'),
    (((-5., 6.), (6.5, 13.)), None, False, 'tracking_broken_before_cycle_end'),
    (((-5., 11.),), None, False, 'tracking_broken_before_cycle_end'),
    (((0., 9.),), 9., True, None),                     # journey completed by crossing
])
def test_collection_requires_start_detection_and_unbroken_tracking(spans, crossing_at, collected, reason):
    result = collection_eligibility(tracked_row(*spans, crossing_at=crossing_at), episode(), 1.)
    assert result['collected'] is collected
    assert result['collection_reasons'] == ([] if reason is None else [reason])


def test_collection_tolerance_is_configurable_and_bounded():
    late = tracked_row((1.5, 13.))
    assert collection_eligibility(late, episode(), 2.)['collected']
    with pytest.raises(ValueError):
        SchedulePolicy(collection_start_tolerance_sec=-1)
