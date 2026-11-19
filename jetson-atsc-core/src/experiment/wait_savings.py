"""Per-vehicle observed-interval schedule estimates, never physical queue simulation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class SchedulePolicy:
    enabled: bool = False
    a_movement_id: str = 'approach-A'
    b_movement_id: str = 'approach-1'
    relationship: str = 'a_then_b'
    unchanged_clearance: bool = False
    phase_source: str = 'camera_roi'
    max_phase_skew_sec: float = .5
    max_episode_sec: float = 300.
    # Acquisition sampling: a vehicle is collected for an A-green episode only when
    # Camera 2 tracks it within this many seconds of A-green onset and keeps tracking
    # it, without a break, until B is released or the vehicle crosses the stop line.
    collection_start_tolerance_sec: float = 1.

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.unchanged_clearance) is not bool:
            raise ValueError('Schedule enable/clearance settings must be booleans')
        if self.relationship != 'a_then_b' or self.phase_source != 'camera_roi':
            raise ValueError('Unsupported phase relationship or phase source')
        for value in (self.a_movement_id, self.b_movement_id):
            if not isinstance(value, str) or not 1 <= len(value.strip()) <= 100:
                raise ValueError('Explicit A and B movement names are required')
        if self.a_movement_id == self.b_movement_id:
            raise ValueError('A and B must be different movements')
        for key, bounds in {'max_phase_skew_sec': (.01, 2.), 'max_episode_sec': (10., 600.),
                            'collection_start_tolerance_sec': (0., 10.)}.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'Invalid schedule setting: {key}')
        if self.enabled and not self.unchanged_clearance:
            raise ValueError('A/B comparison requires an explicit unchanged-clearance assumption')


COLLECTION_POLICY = 'tracked_from_a_green_start_to_b_release_v1'
EPSILON = 1e-6


def tracked_spans(row):
    """Merge consecutive break-free intervals into [start, end] tracking spans."""
    spans = []
    for interval in sorted(row.get('intervals', []), key=lambda i: i['start']):
        if not interval.get('tracked'):
            continue
        if spans and interval['start'] <= spans[-1][1]+EPSILON:
            spans[-1][1] = max(spans[-1][1], interval['end'])
        else:
            spans.append([interval['start'], interval['end']])
    return spans


def collection_eligibility(row, episode, tolerance_sec=1.):
    """Did Camera 2 see this vehicle at the start of the A-green cycle and follow it to the end?

    Cycle start is A-green onset (B red); cycle end is B release. A vehicle that
    crosses the stop line earlier has completed its journey at the crossing.
    """
    start, end = episode.get('a_start'), episode.get('b_release')
    result = {'collected': False, 'collection_reasons': [], 'collection_policy': COLLECTION_POLICY,
              'collection_window': {'start': start, 'end': end, 'start_tolerance_sec': tolerance_sec}}
    if start is None or end is None or not math.isfinite(start) or not math.isfinite(end):
        result['collection_reasons'].append('cycle_boundaries_unknown')
        return result
    target = end
    if row.get('crossed_stop_line') and row.get('crossing_at') is not None and start < row['crossing_at'] < end:
        target = row['crossing_at']
    spans = tracked_spans(row)
    # Condition 1: tracked at some moment inside [start, start+tolerance].
    present = [s for s in spans if s[0] <= start+tolerance_sec+EPSILON and s[1] >= start-EPSILON]
    if not present:
        result['collection_reasons'].append('not_detected_at_cycle_start')
        return result
    # Condition 2: that same unbroken span lasts to the end of the cycle.
    span = max(present, key=lambda s: s[1])
    result['tracked_from'] = round(max(span[0], start), 6)
    result['tracked_until'] = round(span[1], 6)
    if span[1] < target-EPSILON:
        result['collection_reasons'].append('tracking_broken_before_cycle_end')
        return result
    result['collected'] = True
    return result


def vehicle_savings(row, episode):
    result = {'savings_id': f"{episode['episode_id']}:{row['vehicle_id']}",
        'episode_id': episode['episode_id'], 'b_episode_id': episode.get('b_episode_id'),
        'vehicle_id': row['vehicle_id'], 'b_cycle_id': episode.get('b_cycle_id'),
        'assumption_policy': 'observed_intersection_unchanged_clearance_v1',
        'available': False, 'online_eligible': False, 'qualified': False,
        'potential_saving_sec': None, 'estimated_schedule_red_stop_sec': None,
        'reasons': list(episode.get('reasons', []))}
    required = ('a_start', 'a_end', 'b_red_start', 'b_release', 'recommended_green_sec', 'completed_at')
    if any(episode.get(k) is None or not math.isfinite(episode[k]) for k in required):
        result['reasons'].append('missing_episode_boundaries_or_recommendation')
    if not episode.get('a_start_complete') or not episode.get('b_start_complete'):
        result['reasons'].append('incomplete_phase_episode')
    if not episode.get('unchanged_clearance'):
        result['reasons'].append('controller_assumption_missing')
    if result['reasons']:
        return result
    delta = episode['a_end']-episode['a_start']-episode['recommended_green_sec']
    release = episode['b_release']-delta
    result['proposed_b_release'] = release
    if delta < -1e-8:
        result['reasons'].append('later_release_requires_behavior_model')
        return result
    if release < max(episode['b_red_start'], episode['a_start'])-1e-8 or episode['b_release'] < episode['a_end']:
        result['reasons'].append('unsupported_controller_timing')
        return result
    cycle = episode['b_cycle_id']
    observed = row.get('wait_by_cycle', {}).get(str(cycle), {})
    confirmed = observed.get('red_wait_sec', 0.)
    estimated = observed.get('estimated_red_wait_sec', 0.)
    saved_confirmed = saved_estimated = 0.
    for interval in row.get('intervals', []):
        if interval.get('valid') and interval.get('stopped') and interval.get('phase') == 'RED' and interval.get('cycle_id') == cycle:
            saved_confirmed += max(0., min(episode['b_release'], interval['end'])-max(release, interval['start']))
        gap = interval.get('gap_evidence', {})
        if gap.get('status') == 'accepted':
            for part in gap.get('red_slices', []):
                if part.get('cycle_id') == cycle:
                    saved_estimated += max(0., min(episode['b_release'], part['end'])-max(release, part['start']))
    if saved_confirmed > confirmed+1e-6 or saved_estimated > estimated+1e-6:
        result['reasons'].append('interval_totals_do_not_reconcile')
        return result
    deadline=min(release,episode['a_start']+episode['recommended_green_sec'])
    result['action_deadline']=deadline
    online = episode['completed_at'] <= deadline and episode.get('captured_at', deadline) <= deadline
    result.update(available=True, online_eligible=online,
        comparison_mode='schedule_estimate' if online else 'retrospective_schedule_estimate',
        observed_red_stop_sec=round(confirmed, 6), estimated_actual_red_stop_sec=round(estimated, 6),
        confirmed_potential_saving_sec=round(saved_confirmed, 6),
        estimated_potential_saving_sec=round(saved_estimated, 6),
        potential_saving_sec=round(saved_confirmed+saved_estimated, 6),
        estimated_schedule_confirmed_component_sec=round(max(0., confirmed-saved_confirmed), 6),
        estimated_schedule_hidden_component_sec=round(max(0., estimated-saved_estimated), 6),
        estimated_schedule_red_stop_sec=round(max(0., confirmed+estimated-saved_confirmed-saved_estimated), 6),
        measurement_coverage='partial' if row.get('quality_flags') or row.get('left_censored') or row.get('right_censored') else 'observed',
        unresolved_coverage_sec=row.get('unresolved_coverage_sec', 0.))
    if not online:
        result['reasons'].append('inference_completed_after_proposed_deadline')
    return result
