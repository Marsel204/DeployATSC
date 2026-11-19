"""Native fragment continuation through a real ObservationSession: a handoff
and a late return of the handed-off ID, through retention eviction, telemetry
and the saved export."""
import csv
import io
import json
import zipfile

import pytest

from src.perception.detector_tracker import TrackedVehicle
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeANFIS, run_payload

FPS = 25


def script(index):
    """ID 1 approaches, vanishes for 2 frames, ID 2 continues it across the
    stop line (y=210); much later ID 1 is seen again upstream."""
    if index <= 11:return [(1, 60+5*index, 100)]
    if 14 <= index <= 36:return [(2, 60+5*index, 100)]
    if 44 <= index <= 47:return [(1, 80, 200)]
    return []


class ScriptedDetector:
    def __init__(self, *args, **kwargs):self.epoch = 0;self.track_trajectories = {}
    def track(self, frames, timestamp=None):
        index = round(timestamp*FPS)
        return [[TrackedVehicle(track_id, 2, 'car', [x-14, y-30, x+14, y], .9, timestamp, obs_count=6, epoch=self.epoch)
                 for track_id, y, x in script(index)] for _ in frames]
    def update_velocities(self, vehicles, project, timestamp=None):
        for v in vehicles:v.velocity_valid = True;v.velocity_mps = 2.;v.ground_pos = project(*v.bottom_center)
    def reset_tracking(self):self.epoch += 1


@pytest.mark.parametrize('enabled', [False, True])
def test_session_handoff_and_late_return_reach_telemetry_and_export(video_factory, tmp_path, enabled):
    settings = run_payload(video_factory)
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1, 'fragment_continuation': enabled}
    settings['retention_budget'] = 2
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session = Manual(settings, tmp_path/'runs', detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
    try:
        assert session.telemetry()['policy']['fragment_continuation'] is enabled
        labels = []
        for step in range(50):
            session._process_batch()
            if step == 20:
                probe = TrackedVehicle(2, 2, 'car', [86, 130, 114, 160], .9, 20/FPS, obs_count=6)
                labels = [text for _, text in session._preview_labels('acquisition', [probe])]
            session.persist()
        session.state = session.applied_state = 'ended';session._cleanup()
        with zipfile.ZipFile(io.BytesIO(session.export())) as bundle:
            vehicles = json.loads(bundle.read('evidence.json'))['vehicles']
            table = list(csv.DictReader(io.StringIO(bundle.read('vehicles.csv').decode())))
    finally:
        if not session.resources_released:session.state = session.applied_state = 'stopped';session._cleanup()
    rows = {r['vehicle_id'].split(':', 1)[1]: r for r in vehicles}
    assert len(table) == len(rows)
    crossed = [k for k, r in rows.items() if r['crossed_stop_line']]
    if not enabled:
        assert set(rows) == {'acquisition:0:1', 'acquisition:0:2', 'acquisition:0:1:segment-1'}
        assert crossed == ['acquisition:0:2']
        assert not any('native_ids' in r for r in rows.values())
        return
    assert set(rows) == {'acquisition:0:1', 'acquisition:0:1:segment-1'}
    continued, returned = rows['acquisition:0:1'], rows['acquisition:0:1:segment-1']
    assert [n['local_id'] for n in continued['native_ids']] == [1, 2]
    assert labels and labels[0].startswith('#1 car (native #2)')
    assert crossed == ['acquisition:0:1']
    # The crossing was observed on ID 2's own consecutive frames, after the handoff.
    assert continued['crossing_evidence']['from_time'] > continued['identity_links'][0]['linked_at']
    assert continued['unknown_sec'] >= 3/FPS-1e-6
    # The row finished before ID 1 returned, so it is not rewritten; the return is its own row.
    assert 'identity_link_contested' not in continued['quality_flags']
    assert returned['left_censored'] and not returned['crossed_stop_line']
