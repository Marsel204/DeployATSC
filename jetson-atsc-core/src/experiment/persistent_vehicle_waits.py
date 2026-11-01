"""Observed accounting plus separately evidenced, bounded stationary-gap estimates."""
from collections import deque
from dataclasses import replace

from src.experiment.contracts import VehicleObservation
from src.experiment.vehicle_waits import VehicleWaits
from src.experiment.identity_contracts import ASSOCIATION_VERSION


class PersistentVehicleWaits(VehicleWaits):
    def __init__(self, run_id, policy, side, identity_policy):
        super().__init__(run_id, replace(policy, association_gap_sec=identity_policy.occlusion_retention_sec), side)
        self.identity_policy = identity_policy
        self.timeline = deque()
        self.gaps = {}
        self.previous_phase = 'UNKNOWN'
        self.previous_cycle = None
        self.first_source_time=None

    def _new(self, observation, timestamp, sequence, segment=0):
        state = super()._new(observation, timestamp, sequence, segment)
        state.update(gap_candidate=None, exit_candidate=None)
        state['row'].update(identity_scheme='persistent_v1', identity_generation=observation.epoch,
            estimated_red_wait_sec=0., pending_estimated_red_wait_sec=0., display_red_wait_sec=0.,
            unresolved_coverage_sec=0., native_reference=None, recent_native_references=[],
            gap_decision=None, association_policy=self.identity_policy.association)
        return state

    def _red_slices(self, start, end):
        slices = []
        for a, b, phase, cycle, healthy in self.timeline:
            left, right = max(start, a), min(end, b)
            if right > left and phase == 'RED' and healthy:
                if slices and slices[-1]['end'] == left and slices[-1]['cycle_id'] == cycle:
                    slices[-1]['end'] = right
                else:
                    slices.append({'start': left, 'end': right, 'cycle_id': cycle})
        return slices

    def _amount(self, gap, end):
        if not gap['supported'] or end-gap['start'] > self.identity_policy.max_estimated_gap_sec+1e-8:
            return 0.
        return sum(s['end']-s['start'] for s in self._red_slices(gap['start'], end))

    def _decide(self, key, status, reason):
        gap = self.gaps.pop(key)
        state = self.active[key]
        row = state['row']
        end = gap.get('end', self.last_time)
        slices = self._red_slices(gap['start'], end) if status == 'accepted' else []
        amount = sum(s['end']-s['start'] for s in slices)
        row['estimated_red_wait_sec'] += amount
        for part in slices:
            cycle = part['cycle_id']
            if cycle is not None:
                if cycle not in row['cycles']:
                    row['cycles'].append(cycle)
                values = row['wait_by_cycle'].setdefault(str(cycle), {'red_wait_sec': 0., 'total_stopped_sec': 0.})
                values['estimated_red_wait_sec'] = values.get('estimated_red_wait_sec', 0.)+part['end']-part['start']
        decision = {'gap_id': f"{row['vehicle_id']}:{gap['start']:.9f}", 'start': gap['start'],
            'end': end, 'status': status, 'reason': reason, 'estimated_red_wait_sec': amount,
            'red_slices': slices, 'policy': self.identity_policy.gap_accounting}
        row['gap_decision'] = decision
        # This interval is protected from prefix sealing while the gap is pending.
        interval = next((i for i in row['intervals'] if abs(i['start']-gap['start']) < 1e-8 and abs(i['end']-end) < 1e-8), None)
        if interval is not None:
            interval['gap_evidence'] = decision
        row['pending_estimated_red_wait_sec'] = 0.
        row['pending_gap']=None
        state['gap_candidate'] = None
        if status != 'accepted':
            self._flag(row, 'hidden_wait_unresolved')

    def _finish_missing(self, key, timestamp):
        gap = self.gaps[key]
        if 'end' not in gap:
            row = self.active[key]['row']
            interval = {'start': gap['start'], 'end': timestamp, 'phase': 'UNKNOWN',
                'cycle_id': None, 'valid': False, 'stopped': False}
            if timestamp > gap['start']:
                row['intervals'].append(interval)
                self.work += 1
                self._account(row, interval)
            gap['end'] = timestamp
        self._decide(key, 'unresolved', 'visit_ended_without_supported_recovery')

    def observe_resolved(self, resolved, phase, timestamp, sequence, cycle_id, *, evidence_healthy=True,ambiguous_ids=()):
        if self.last_sequence is not None and (sequence <= self.last_sequence or timestamp <= self.last_time):
            return
        if self.first_source_time is None:self.first_source_time=timestamp
        continuous = self.last_sequence is not None and sequence == self.last_sequence+1 and 0 < timestamp-self.last_time <= self.policy.max_gap_sec
        if self.last_time is not None:
            self.timeline.append((self.last_time, timestamp, self.previous_phase, self.previous_cycle, continuous))
        floor = timestamp-self.identity_policy.occlusion_retention_sec-self.policy.max_gap_sec-1
        while self.timeline and self.timeline[0][1] < floor:
            self.timeline.popleft()
        present = {(x.generation, x.logical_id): x for x in resolved}
        for key, state in list(self.active.items()):
            row = state['row']
            if row['completed']:
                continue
            if key not in present and key not in self.gaps and state['previous']:
                self.gaps[key] = {'start': row['last_seen'], 'supported': bool(
                    continuous and evidence_healthy and state['previous']['valid_motion'] and state['stopped']),
                    'return_stop_since': None}
                state['gap_candidate'] = row['last_seen']
            gap = self.gaps.get(key)
            if gap:
                if not continuous or not evidence_healthy:
                    gap['supported'] = False
                    gap['unsupported_reason']='camera_packet_gap' if not continuous else 'background_unobservable'
                if key[1] in ambiguous_ids:
                    gap['supported']=False;gap['unsupported_reason']='identity_ambiguous'
                    self._flag(row,'identity_ambiguous')
                if key in present and 'end' not in gap:
                    gap['end'] = timestamp
        observations = [VehicleObservation(x.logical_id, x.detection.class_name,
            x.speed_mps, x.detection.point, stable_id=True,
            velocity_valid=x.velocity_valid, in_region=x.detection.in_region or x.exit_pending,
            epoch=x.generation) for x in resolved]
        super().observe(observations, phase, timestamp, sequence, cycle_id)
        self.previous_phase, self.previous_cycle = phase, cycle_id if phase != 'UNKNOWN' else None
        for key, item in present.items():
            state = self.active.get(key)
            if state is None or state['row']['vehicle_id'] not in self.records:
                continue
            row = state['row']
            if row['completed'] and row['completed_at']!=timestamp:continue
            row.setdefault('first_detected_at',item.birth_timestamp if item.birth_timestamp is not None else row['first_seen'])
            row.setdefault('admitted_at',row['first_seen'])
            row.setdefault('first_native_reference',None if item.first_native_reference is None else {
                'epoch':item.first_native_reference[0],'local_id':item.first_native_reference[1]})
            if row['first_native_reference'] is None and item.first_native_reference is not None:
                row['first_native_reference']={'epoch':item.first_native_reference[0],'local_id':item.first_native_reference[1]}
            if item.birth_timestamp is not None and item.birth_timestamp<=self.first_source_time+1e-8:
                row['left_censored']=True
            reference = {'epoch': item.detection.native_epoch,
                'local_id': item.detection.native_id if item.detection.native_stable else None}
            row['native_reference'] = reference
            if item.recovered:self._flag(row,'identity_recovery_assumption')
            if row['intervals'] and abs(row['intervals'][-1]['end']-timestamp) < 1e-8:
                row['intervals'][-1].update(native_reference=reference, identity_association=item.association)
            if not row['recent_native_references'] or row['recent_native_references'][-1] != reference:
                row['recent_native_references'] = (row['recent_native_references']+[reference])[-16:]
            state['exit_candidate'] = timestamp if item.exit_pending and state['exit_candidate'] is None else (state['exit_candidate'] if item.exit_pending else None)
            row['identity_state'] = ('CROSSED' if row['crossed_stop_line'] else 'EXITED') if row['completed'] else ('EXIT_PENDING' if item.exit_pending else 'VISIBLE')
        for key, gap in list(self.gaps.items()):
            state = self.active[key]
            row = state['row']
            end = gap.get('end', timestamp)
            row['pending_estimated_red_wait_sec'] = self._amount(gap, end)
            row['identity_state'] = 'AMBIGUOUS' if key[1] in ambiguous_ids else ('RECOVERING' if key in present else 'OCCLUDED')
            if row['completed']:
                self._finish_missing(key, timestamp)
            elif 'end' in gap:
                item = present.get(key)
                if not gap['supported'] or end-gap['start'] > self.identity_policy.max_estimated_gap_sec+1e-8:
                    self._decide(key, 'unresolved', 'gap_exceeded_estimate_cap' if end-gap['start']>self.identity_policy.max_estimated_gap_sec+1e-8 else gap.get('unsupported_reason','unsupported_stationary_gap'))
                elif item is not None and item.velocity_valid:
                    if item.speed_mps > self.policy.stopped_speed_mps:
                        self._decide(key, 'rejected', 'returning_vehicle_is_moving')
                    else:
                        if gap['return_stop_since'] is None:
                            gap['return_stop_since'] = timestamp
                        if timestamp-gap['return_stop_since'] >= self.policy.stop_persistence_sec-1e-8:
                            self._decide(key, 'accepted', 'unique_stationary_recovery')
                elif item is None:
                    gap['supported'] = False
            if key in self.gaps and timestamp-end > self.policy.max_gap_sec+max(1., self.policy.stop_persistence_sec):
                self._decide(key, 'unresolved', 'motion_confirmation_timeout')
        self._refresh(timestamp)
        self._coalesce()

    def _coalesce(self):
        # Preserve revisable hysteresis/gap suffixes. Older identical evidence is
        # represented by a duration, not one dictionary for every decoded frame.
        fields=('phase','cycle_id','valid','stopped','native_reference','identity_association')
        for state in self.active.values():
            intervals=state['row'].get('intervals',[])
            candidates=[state[k] for k in ('stop_candidate','move_candidate','gap_candidate','exit_candidate') if state.get(k) is not None]
            cutoff=min(candidates) if candidates else float('inf')
            if len(intervals)<2:continue
            if len(intervals)-2<state.get('_prepared_prefix_count',0):continue
            a,b=intervals[-2:]
            if b['end']<=cutoff+1e-8 and abs(a['end']-b['start'])<1e-8 and 'gap_evidence' not in a and 'gap_evidence' not in b and all(a.get(k)==b.get(k) for k in fields):
                a['end']=b['end'];intervals.pop();self.work-=1

    def _refresh(self, timestamp):
        for key, state in self.active.items():
            row = state['row']
            if 'red_wait_sec' not in row:
                continue
            pending = row['pending_estimated_red_wait_sec']
            estimated = row['estimated_red_wait_sec']
            row['display_red_wait_sec'] = row['red_wait_sec']+estimated+pending
            gap = self.gaps.get(key)
            open_unknown = max(0., timestamp-gap['start']) if gap and 'end' not in gap else 0.
            row['pending_gap']=None if gap is None else {'start':gap['start'],'end':gap.get('end'),
                'through_time':gap.get('end',timestamp),'supported':bool(gap['supported'])}
            row['unresolved_coverage_sec'] = max(0., row['unknown_sec']+open_unknown-estimated-pending)

    def invalidate(self, timestamp, reason):
        for key in list(self.gaps):
            self._finish_missing(key, timestamp)
        super().invalidate(timestamp, reason)
        for state in self.active.values():
            row = state['row']
            if not row['completed']:
                row.update(completed=True, completed_at=timestamp, right_censored=True, identity_state='LOST_UNRESOLVED')
                self.work += 1
        self._refresh(timestamp)
        self.active.clear()
        self.timeline.clear()
        self.first_source_time=None

    def close(self, timestamp, reason='recording_end'):
        self.invalidate(timestamp, reason)
