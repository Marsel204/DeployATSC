"""One bounded still-image worker and explicit A-green/B-red episode binding."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import math
import threading
import time

from src.common.source_identity import public_data, safe_text
from src.experiment.wait_savings import vehicle_savings


class ScheduleEpisodes:
    def __init__(self, run_id, policy, evaluate, clock):
        self.run_id, self.policy, self.evaluate, self.clock = run_id, policy, evaluate, clock
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='ANFIS-A-episode')
        self.future = None
        self.records, self.savings = {}, {}
        self.current_a = self.b_red = None
        self.last_b_time = None
        self.serial = self.generation = self.revision = 0
        self.closed = False
        self.rejected_episode_count = 0

    @property
    def resources_released(self):
        with self.lock:
            return self.future is None or self.future.done()

    def await_idle(self, timeout=5):
        future = self.future
        if future is not None:
            future.result(timeout=timeout)

    def _bind(self, record):
        if self.b_red and self.b_red['start'] <= record['captured_at']+1e-8 and abs(record['captured_at']-self.last_b_time) <= self.policy.max_phase_skew_sec:
            record.update(b_episode_id=self.b_red['episode_id'], b_cycle_id=self.b_red['cycle_id'],
                b_red_start=self.b_red['start'], b_start_complete=self.b_red['start_complete'])
            for other in self.records.values():
                if other is not record and other['b_episode_id']==record['b_episode_id']:
                    for item in (other,record):
                        if not item['sealed']:
                            item['reasons'].append('multiple_a_greens_in_one_b_red')
                            item['status']='unavailable'

    def observe_b(self, phase, timestamp, cycle_id, transition):
        with self.lock:
            self.last_b_time = timestamp
            if transition and transition['phase'] == 'RED':
                self.b_red = {'episode_id': f"{self.run_id}:B:{cycle_id}", 'cycle_id': cycle_id,
                    'start': transition['timestamp'], 'start_complete': transition['from'] in ('GREEN', 'YELLOW')}
            if phase == 'RED' and self.b_red:
                for record in self.records.values():
                    if not record['sealed'] and record['b_episode_id'] is None and abs(record['captured_at']-timestamp) <= self.policy.max_phase_skew_sec:
                        self._bind(record)
            if transition and transition['phase'] in ('GREEN', 'UNKNOWN') and self.b_red:
                for record in self.records.values():
                    if not record['sealed'] and record['b_episode_id'] == self.b_red['episode_id']:
                        record['b_release'] = transition['timestamp'] if transition['phase'] == 'GREEN' else None
                        if transition['phase'] == 'UNKNOWN':
                            record['reasons'].append('b_phase_discontinuity')
                            record['status'] = 'unavailable'
                self.b_red = None
            self._expire(timestamp)
            self.revision += 1

    def observe_a(self, transition, frame, analyze):
        with self.lock:
            if self.closed:
                return
            timestamp = frame['timestamp']
            self._expire(timestamp)
            if not transition:
                return
            if transition['phase'] != 'GREEN':
                record = self.records.get(self.current_a)
                if record and not record['sealed']:
                    record['a_end'] = transition['timestamp']
                    if transition['phase'] != 'YELLOW':
                        record['reasons'].append('a_green_end_not_observed')
                        record['status'] = 'unavailable'
                self.current_a = None
                self.revision += 1
                return
            self.serial += 1
            if len(self.records) >= 256:
                self.rejected_episode_count += 1
                self.revision += 1
                return
            identity = f'{self.run_id}:A:{self.serial}'
            record = {'episode_id': identity, 'a_movement_id': self.policy.a_movement_id,
                'b_movement_id': self.policy.b_movement_id, 'a_start': transition['timestamp'],
                'a_confirmed_at': transition['confirmed_at'], 'a_start_complete': transition['from'] in ('RED', 'YELLOW'),
                'a_end': None, 'captured_at': timestamp, 'camera1_frame': public_data(deepcopy(frame)),
                'completed_at': None, 'b_episode_id': None, 'b_cycle_id': None, 'b_red_start': None,
                'b_start_complete': False, 'b_release': None, 'recommended_green_sec': None,
                'unchanged_clearance': self.policy.unchanged_clearance, 'generation': self.generation,
                'status': 'predicting', 'sealed': False, 'reasons': [], 'camera_analysis': {'inference_calls': 0,'state':'pending'},
                'qualification': {'state': 'diagnostic', 'reasons': ['single_frame_spatial_proxy', 'schedule_assumptions']}}
            self._bind(record)
            self.records[identity] = record
            self.current_a = identity
            if not self.resources_released:
                record.update(status='unavailable', sealed=True)
                record['reasons'].append('previous_a_inference_still_running')
            else:
                self.future = self.executor.submit(self._run, identity, self.generation, analyze)
            self.revision += 1

    def _run(self, identity, generation, analyze):
        with self.lock:
            record=self.records.get(identity)
            if self.closed or generation!=self.generation or record is None or record['sealed']:return
            record['camera_analysis'].update(inference_calls=1,state='analysing')
        started = time.perf_counter()
        snapshot = analysis = diagnostics = None
        try:
            snapshot, analysis = analyze()
            if snapshot.queue_meters is None or not all(math.isfinite(x) and x >= 0 for x in
                    (snapshot.v_w_pcu, snapshot.occupancy_pct, snapshot.queue_meters)):
                raise ValueError('A still-image metrics are unavailable or invalid')
            decision = self.evaluate(snapshot, 0)
            value = float(decision.green_seconds)
            diagnostics = public_data(deepcopy(getattr(decision, 'inference_diagnostics', None)))
            if not math.isfinite(value) or not 10 <= value <= 120 or decision.failsafe_active or (diagnostics or {}).get('backend_fallback'):
                raise ValueError('A recommendation failed ANFIS bounds/backend checks')
            error = None
        except Exception as failure:
            value, error = None, safe_text(failure)
        completed = self.clock()
        with self.lock:
            record = self.records.get(identity)
            if self.closed or generation != self.generation or record is None or record['sealed']:
                return
            record.update(recommended_green_sec=value, completed_at=completed,
                inference_latency_ms=round((time.perf_counter()-started)*1000, 3),
                inference_diagnostics=diagnostics, camera_analysis={**(analysis or {}), 'inference_calls': 1,
                    'state':'complete' if error is None else 'failed','error':error},
                snapshot=public_data(asdict(snapshot)) if snapshot is not None else None,
                status='ready' if error is None and not record['reasons'] else 'unavailable')
            if error:
                record['reasons'].append(error)
            if (analysis or {}).get('input_sha256'):record['camera1_frame']['sha256']=analysis['input_sha256']
            self.revision += 1

    def _expire(self, timestamp):
        for record in self.records.values():
            if not record['sealed'] and timestamp-record['a_start'] > self.policy.max_episode_sec:
                record.update(status='unavailable', sealed=True)
                record['reasons'].append('episode_timeout')
                record['camera_analysis']['state']='interrupted' if record['camera_analysis']['inference_calls'] else 'cancelled'

    def protected_since(self):
        with self.lock:
            values = [r['a_start'] for r in self.records.values() if not r['sealed']]
            return min(values) if values else None

    def protected_cycles(self):
        with self.lock:
            return {r['b_cycle_id'] for r in self.records.values() if not r['sealed'] and r['b_cycle_id'] is not None}

    def finalize(self, rows, active):
        with self.lock:
            rows, active = list(rows), list(active)
            for record in self.records.values():
                if record['sealed'] or record['status'] == 'predicting':
                    continue
                if record['status'] != 'unavailable' and (record['a_end'] is None or record['b_release'] is None):
                    continue
                cycle = record['b_cycle_id']
                if any(state.get(k) is not None and state[k] <= (record['b_release'] or float('inf'))
                        for state in active for k in ('gap_candidate', 'stop_candidate', 'move_candidate')
                        if cycle in state['row'].get('cycles', [])):
                    continue
                probe=vehicle_savings({'vehicle_id':'episode-validation'},record)
                if not probe['available']:
                    record['reasons']=list(dict.fromkeys(record['reasons']+probe['reasons']))
                for row in rows:
                    if cycle in row.get('cycles', []):
                        value = vehicle_savings(row, record)
                        self.savings[value['savings_id']] = value
                record.update(sealed=True, status='unavailable' if record['reasons'] else 'complete')
                if record['camera_analysis']['state'] in ('pending','analysing'):
                    record['camera_analysis']['state']='discarded_after_unavailable'
                self.revision += 1

    def view(self):
        with self.lock:
            return {'policy': asdict(self.policy), 'episodes': deepcopy(list(self.records.values())),
                'vehicle_savings': deepcopy(list(self.savings.values())), 'revision': self.revision,
                'busy': not self.resources_released}

    def invalidate(self, reason):
        with self.lock:
            self.generation += 1
            self.current_a = self.b_red = None
            for record in self.records.values():
                if not record['sealed']:
                    record.update(status='unavailable', sealed=True)
                    record['reasons'].append(reason)
                    record['camera_analysis']['state']='interrupted' if record['camera_analysis']['inference_calls'] else 'cancelled'
            self.revision += 1

    def close(self, reason):
        self.invalidate(reason)
        with self.lock:
            self.closed = True
            self.executor.shutdown(wait=False, cancel_futures=True)
