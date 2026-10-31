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
