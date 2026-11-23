"""Native fragment continuation through a real ObservationSession: a handoff
and a late return of the handed-off ID, through retention eviction, telemetry
and the saved export."""
import csv
import io
import json
import zipfile

import pytest

from frozen_tracking import FrozenTrackModel, frozen_detector
from src.perception.detector_tracker import TrackedFrame, TrackedVehicle
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


def scripted_vehicle(track_id, x, y, timestamp, epoch):
    vehicle = TrackedVehicle(track_id, 2, 'car', [x-14, y-30, x+14, y], .9, timestamp, obs_count=6, epoch=epoch)
    vehicle.source_bbox = tuple(vehicle.bbox)      # its own raw detection: complete attribution
    return vehicle


class ScriptedDetector:
    """Scripted tracks; declares that the tracker withheld nothing in any frame."""
    def __init__(self, *args, **kwargs):self.epoch = 0;self.track_trajectories = {}
    def track(self, frames, timestamp=None):
        index = round(timestamp*FPS)
        return [TrackedFrame([scripted_vehicle(track_id, x, y, timestamp, self.epoch)
                 for track_id, y, x in script(index)], withheld=(), epoch=self.epoch) for _ in frames]
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


class PlainListDetector(ScriptedDetector):
    """A detector without frame evidence (plain lists, as any non-DetectorTracker returns)."""
    def track(self, frames, timestamp=None):
        return [list(frame) for frame in super().track(frames, timestamp)]


def run_session(video_factory, tmp_path, factory, steps):
    settings = run_payload(video_factory)
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1, 'fragment_continuation': True}
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session = Manual(settings, tmp_path/'runs', detector_factory=factory, anfis_factory=FakeANFIS)
    try:
        for _ in range(steps):session._process_batch()
        waits = session.engine.waits
        return ({k.split(':', 1)[1]: r for k, r in waits.records.items()}, waits.continuation_unavailable_frames)
    finally:
        session.state = session.applied_state = 'stopped';session._cleanup()


def test_session_without_frame_evidence_never_guesses_a_handoff(video_factory, tmp_path):
    rows, unavailable = run_session(video_factory, tmp_path, PlainListDetector, 40)
    assert not any(r.get('identity_links') for r in rows.values())
    assert {'acquisition:0:1', 'acquisition:0:2'} <= set(rows) and unavailable >= 1


def frozen_factory(competitor):
    """Sol's repro geometry (40x6 px) through the installed tracker callbacks, inside the ROI."""
    def boxes(index, image):
        def box(x):return [x-3, 100, x+3, 140, .95, 1]
        if index <= 4:return [box(100)]
        if index == 5:return [box(112)]
        return [box(112)]+([box(94)] if competitor and index == 6 else [])
    # Capture starts disabled: the session's opt-in must enable it.
    return lambda *args, **kwargs:frozen_detector(FrozenTrackModel(boxes), evidence=False)


@pytest.mark.parametrize('competitor', [True, False])
def test_session_vetoes_with_bytetrack_withheld_competitor(video_factory, tmp_path, competitor):
    rows, unavailable = run_session(video_factory, tmp_path, frozen_factory(competitor), 12)
    linked = [r for r in rows.values() if r.get('identity_links')]
    assert unavailable == 0
    if competitor:
        assert linked == [] and {'acquisition:0:1', 'acquisition:0:2'} <= set(rows)
    else:
        assert [[n['local_id'] for n in r['native_ids']] for r in linked] == [[1, 2]]


class EvidenceSpyDetector(ScriptedDetector):
    enabled = []
    def enable_withheld_evidence(self):
        type(self).enabled.append(self)
        return True


@pytest.mark.parametrize('enabled,identity,expect', [(False, None, False), (True, None, True),
                                                     (True, 'persistent_v1', False), (False, 'persistent_v1', False)])
def test_withheld_capture_is_enabled_only_for_native_fragment_continuation(video_factory, tmp_path,
                                                                           enabled, identity, expect):
    settings = run_payload(video_factory)
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1, 'fragment_continuation': enabled}
    if identity:settings['vehicle_identity'] = {'mode': identity}
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    EvidenceSpyDetector.enabled = []
    session = Manual(settings, tmp_path/'runs', detector_factory=EvidenceSpyDetector, anfis_factory=FakeANFIS)
    try:
        assert session.fragment_continuation is expect
        # Only the acquisition detector, and only when continuation can use the evidence.
        assert EvidenceSpyDetector.enabled == ([session.detectors['acquisition']] if expect else [])
    finally:
        session.state = session.applied_state = 'stopped';session._cleanup()


def test_default_off_session_keeps_the_historical_observe_acquisition_contract(video_factory, tmp_path, monkeypatch):
    """A previously compatible engine override (no unassigned_boxes) still works when continuation is off."""
    from src.ui import observation_session as module
    calls = []
    class Legacy(module.ObservationEngine):
        def observe_acquisition(self, frame, timestamp, sequence, observations, *, generation=None):
            calls.append(sequence)
            return super().observe_acquisition(frame, timestamp, sequence, observations, generation=generation)
    monkeypatch.setattr(module, 'ObservationEngine', Legacy)
    settings = run_payload(video_factory)
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1}
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session = Manual(settings, tmp_path/'runs', detector_factory=frozen_factory(True), anfis_factory=FakeANFIS)
    try:
        for _ in range(8):session._process_batch()
        assert calls and session.fragment_continuation is False
        assert getattr(session.detectors['acquisition'], '_provenance', None) is None
    finally:
        session.state = session.applied_state = 'stopped';session._cleanup()


@pytest.mark.parametrize('bad_withheld', [[()], [((0, 0, 10**400, 4), 'car', .9)], {}, [((-1e308, -1e308, 1e308, 1e308), 'car', .9)]])
def test_malformed_withheld_entries_degrade_to_unavailable_evidence_without_crashing(video_factory, tmp_path, bad_withheld):
    """Sol S3/S6/S7: malformed entries (mappings, integer overflow, infinite-span coordinates) in
    TrackedFrame.withheld yield unavailable evidence (None) and never crash ObservationSession."""
    from src.ui.observation_session import _withheld_boxes
    clean_frame = TrackedFrame([scripted_vehicle(1, 100., 60., 0., 0)], withheld=(), epoch=0)
    assert _withheld_boxes(clean_frame, 0) == ()
    for bad in ([()], [None], [123], [('', 'car', .9)], [({}, 'car', .9)],
                [((0, 0, 10**400, 4), 'car', .9)], [((-1e308, -1e308, 1e308, 1e308), 'car', .9)], {}, 42, ''):
        bad_frame = TrackedFrame([scripted_vehicle(1, 100., 60., 0., 0)], withheld=bad, epoch=0)
        assert _withheld_boxes(bad_frame, 0) is None

    class MalformedWithheldDetector(ScriptedDetector):
        def track(self, frames, timestamp=None):
            index = round(timestamp*FPS)
            withheld = bad_withheld if index == 14 else ()
            return [TrackedFrame([scripted_vehicle(track_id, x, y, timestamp, self.epoch)
                     for track_id, y, x in script(index)], withheld=withheld, epoch=self.epoch) for _ in frames]

    rows, unavailable = run_session(video_factory, tmp_path, MalformedWithheldDetector, 20)
    assert not any(r.get('identity_links') for r in rows.values())
    assert {'acquisition:0:1', 'acquisition:0:2'} <= set(rows)
    assert unavailable >= 1


@pytest.mark.parametrize('bad_bbox', [[200., 100., 200., 130.], [-1e308, -1e308, 1e308, 1e308]])
def test_zero_width_emitted_box_makes_frame_evidence_unavailable(video_factory, tmp_path, bad_bbox):
    """Sol S6/S7: an emitted track with a degenerate, mapping, or overflow-span bbox must mark frame
    evidence unavailable even when its source_bbox is valid."""
    from src.ui.observation_session import _finite_box, _withheld_boxes
    assert _finite_box({}) is None
    mapped = TrackedVehicle(3, 2, 'car', [200., 100., 210., 140.], .9, 14/FPS, obs_count=6, epoch=0)
    mapped.bbox = {}
    mapped.source_bbox = (186., 100., 214., 130.)
    assert _withheld_boxes(TrackedFrame([mapped], withheld=(), epoch=0), 0) is None

    bad = TrackedVehicle(3, 2, 'car', bad_bbox, .9, 14/FPS, obs_count=6, epoch=0)
    bad.source_bbox = (186., 100., 214., 130.)
    assert _withheld_boxes(TrackedFrame([bad], withheld=(), epoch=0), 0) is None

    class DegenerateEmittedDetector(ScriptedDetector):
        def track(self, frames, timestamp=None):
            index = round(timestamp*FPS)
            vehicles = [scripted_vehicle(track_id, x, y, timestamp, self.epoch) for track_id, y, x in script(index)]
            if index == 14:
                other = TrackedVehicle(3, 2, 'car', bad_bbox, .9, timestamp, obs_count=6, epoch=self.epoch)
                other.source_bbox = (186., 100., 214., 130.)
                vehicles.append(other)
            return [TrackedFrame(vehicles, withheld=(), epoch=self.epoch) for _ in frames]

    rows, unavailable = run_session(video_factory, tmp_path, DegenerateEmittedDetector, 20)
    assert not any(r.get('identity_links') for r in rows.values())
    assert {'acquisition:0:1', 'acquisition:0:2'} <= set(rows)
    assert unavailable >= 1


@pytest.mark.parametrize('mode,continuing,expect_box_parses,expect_obs', [
    ('native', False, False, True),
    ('native', True, True, True),
    ('persistent_v1', False, False, False),
    ('persistent_v1', True, False, False),
])
def test_default_off_and_persistent_sessions_skip_continuation_box_parsing(
    video_factory, tmp_path, monkeypatch, mode, continuing, expect_box_parses, expect_obs
):
    """Sol S4: default-off native and persistent_v1 sessions do not parse continuation boxes,
    and persistent_v1 does not construct native VehicleObservation objects."""
    from src.ui import observation_session as module
    box_calls, source_calls, obs_calls = [], [], []
    orig_box, orig_source, orig_obs = module._box, module._source_box, module.VehicleObservation

    def spy_box(v):
        box_calls.append(v); return orig_box(v)

    def spy_source(v):
        source_calls.append(v); return orig_source(v)

    def spy_obs(*args, **kwargs):
        obs_calls.append((args, kwargs)); return orig_obs(*args, **kwargs)

    monkeypatch.setattr(module, '_box', spy_box)
    monkeypatch.setattr(module, '_source_box', spy_source)
    monkeypatch.setattr(module, 'VehicleObservation', spy_obs)

    settings = run_payload(video_factory)
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1, 'fragment_continuation': continuing}
    if mode == 'persistent_v1':
        settings['vehicle_identity'] = {'mode': 'persistent_v1'}

    class Manual(ObservationSession):
        def _worker(self):self.ready.set()

    session = Manual(settings, tmp_path/'runs', detector_factory=ScriptedDetector, anfis_factory=FakeANFIS)
    try:
        for _ in range(5):session._process_batch()
    finally:
        session.state = session.applied_state = 'stopped';session._cleanup()
    assert bool(obs_calls) is expect_obs
    assert bool(box_calls) is expect_box_parses
    assert bool(source_calls) is expect_box_parses


def test_session_exit_persistence_bridges_border_dip_and_leaves_persistent_mode_unchanged(video_factory, tmp_path):
    """RunPolicy.exit_persistence_sec reaches ObservationSession telemetry, keeps a
    briefly dipping native vehicle on one row, and leaves persistent_v1 identical."""
    class DippingDetector(ScriptedDetector):
        def track(self, frames, timestamp=None):
            index = round(timestamp*FPS)
            # In run_payload, left polygon edge runs (20,30)->(10,220), so at y=60 edge is x~18.4:
            # x=15 dips ~3.4 px outside while keeping an in-frame positive bbox [1,30,29,60].
            x = 15 if index in (5, 6) else 25
            vehicles = [scripted_vehicle(1, x, 60, timestamp, self.epoch)] if index < 12 else []
            return [TrackedFrame(vehicles, withheld=(), epoch=self.epoch) for _ in frames]

    def run_with(mode, exit_sec):
        settings = run_payload(video_factory)
        settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1, 'exit_persistence_sec': exit_sec}
        if mode == 'persistent_v1':
            settings['vehicle_identity'] = {'mode': 'persistent_v1'}
        class Manual(ObservationSession):
            def _worker(self):self.ready.set()
        session = Manual(settings, tmp_path/f'{mode}-{exit_sec}', detector_factory=DippingDetector, anfis_factory=FakeANFIS)
        try:
            assert session.telemetry()['policy']['exit_persistence_sec'] == pytest.approx(exit_sec)
            for _ in range(14):session._process_batch()
            snap = session.engine.snapshot(limit=100, evidence=True)
            return {r['vehicle_id'].split(':', 1)[1]: {**r, 'vehicle_id': r['vehicle_id'].split(':', 1)[1]} for r in snap['vehicles']}
        finally:
            session.state = session.applied_state = 'stopped';session._cleanup()

    native_off = run_with('native', 0.0)
    native_on = run_with('native', 0.3)
    assert set(native_off) == {'acquisition:0:1', 'acquisition:0:1:segment-1'}
    assert set(native_on) == {'acquisition:0:1'}

    # persistent_v1 rows must be identical regardless of RunPolicy.exit_persistence_sec.
    assert run_with('persistent_v1', 0.0) == run_with('persistent_v1', 0.3)


def test_session_malformed_box_behind_stop_line_never_starts_false_crossing(video_factory, tmp_path):
    """Sol S14 (Round 14/15): ObservationSession with fragment_continuation=True preserves
    malformed-box information instead of degrading a zero-width or NaN box to bbox=None,
    and skips malformed boxes in preview rendering so acquisition continues cleanly without
    falsely completing SERVED (while a valid transition across stop_y=210 still serves)."""
    def make_detector(first_kind, y_first, y_second):
        class CrossingDetector(ScriptedDetector):
            def track(self, frames, timestamp=None):
                index = round(timestamp*FPS)
                if index == 0:
                    if first_kind == 'zero_width':
                        bbox = [100., y_first-30., 100., y_first]
                    elif first_kind == 'nan':
                        bbox = [86., float('nan'), 114., y_first]
                    else:
                        bbox = [86., y_first-30., 114., y_first]
                    v = TrackedVehicle(1, 2, 'car', bbox, .9, timestamp, obs_count=6, epoch=self.epoch)
                    v.source_bbox = tuple(bbox)
                    vehicles = [v]
                elif index == 1:
                    vehicles = [scripted_vehicle(1, 100., y_second, timestamp, self.epoch)]
                else:
                    vehicles = []
                return [TrackedFrame(vehicles, withheld=(), epoch=self.epoch) for _ in frames]
        return CrossingDetector

    def run_crossing(tag, first_kind, y_first, y_second, flip_stop_line=False):
        settings = run_payload(video_factory)
        if flip_stop_line:
            p0, p1 = settings['acquisition']['stop_line']
            settings['acquisition']['stop_line'] = [p1, p0]
        settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1, 'fragment_continuation': True}
        class Manual(ObservationSession):
            def _worker(self):self.ready.set()
        session = Manual(settings, tmp_path/f'{tag}-{first_kind}',
                         detector_factory=make_detector(first_kind, y_first, y_second), anfis_factory=FakeANFIS)
        try:
            assert session.calibrations['acquisition'].side((100., y_first)) < 0 <= session.calibrations['acquisition'].side((100., y_second))
            for _ in range(3):session._process_batch()
            assert session.runtime['acquisition']['frames'] == 3
            rows = {k.split(':', 1)[1]: r for k, r in session.engine.waits.records.items()}
            return rows['acquisition:0:1']
        finally:
            session.state = session.applied_state = 'stopped';session._cleanup()

    # Exact Round 14/15 repros (upward y=212 -> 206 AND downward y=206 -> 212 across stop_y=210).
    for tag, y0, y1, flip in (('up', 212., 206., True), ('down', 206., 212., False)):
        for bad_kind in ('zero_width', 'nan'):
            bad_row = run_crossing(tag, bad_kind, y0, y1, flip_stop_line=flip)
            assert not bad_row['crossed_stop_line'] and bad_row['tracking_state'] != 'SERVED'
        good_row = run_crossing(tag, 'valid', y0, y1, flip_stop_line=flip)
        assert good_row['crossed_stop_line'] and good_row['tracking_state'] == 'SERVED'


def test_session_nan_footpoint_does_not_confirm_false_roi_exit_or_split_segment(video_factory, tmp_path):
    """Sol Round 16: non-finite footpoints (e.g. y2=NaN) must be treated as missing
    observations rather than outside-ROI sightings, so 9 NaN frames under exit_persistence_sec=0.3
    do not falsely close the row as EXITED or split the return into :segment-1."""
    class NanFootpointDetector(ScriptedDetector):
        def track(self, frames, timestamp=None):
            index = round(timestamp*FPS)
            if index == 0 or index == 10:
                bbox = [86., 120., 114., 150.]
            elif 1 <= index <= 9:
                bbox = [86., 120., 114., float('nan')]
            else:
                return [TrackedFrame([], withheld=(), epoch=self.epoch) for _ in frames]
            v = TrackedVehicle(1, 2, 'car', bbox, .9, timestamp, obs_count=6, epoch=self.epoch)
            v.source_bbox = tuple(bbox)
            return [TrackedFrame([v], withheld=(), epoch=self.epoch) for _ in frames]

    settings = run_payload(video_factory)
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1,
                          'fragment_continuation': True, 'exit_persistence_sec': 0.3}
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session = Manual(settings, tmp_path/'nan-footpoint', detector_factory=NanFootpointDetector, anfis_factory=FakeANFIS)
    try:
        for _ in range(12):session._process_batch()
        rows = {k.split(':', 1)[1]: r for k, r in session.engine.waits.records.items()}
        assert set(rows) == {'acquisition:0:1'}
        row = rows['acquisition:0:1']
        assert not row['completed'] and row['tracking_state'] != 'EXITED'
        assert 'observation_gap' in row['quality_flags']
    finally:
        session.state = session.applied_state = 'stopped';session._cleanup()


def test_session_nan_footpoint_does_not_corrupt_detector_velocity_or_red_wait(video_factory, tmp_path):
    """Sol Round 17/18: DetectorTracker.update_velocities and ObservationSession must reject
    non-finite and float32-overflowing footpoints (NaN, 1e40) before OpenCV perspectiveTransform
    and ROI/crossing accounting, preserving the exact red wait and tracking state of missing frames."""
    from src.perception.detector_tracker import DetectorTracker

    def make_velocity_detector(bad_y2):
        class RealVelocityDetector(ScriptedDetector):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.track_obs_counts = {}
                self.max_history_len = 5
                self.last_seen_timestamp = 0.0
                self.max_gap_sec = 5.0
            def track(self, frames, timestamp=None):
                index = round(timestamp*FPS)
                if index < 12 or 22 <= index < 36:
                    bbox = [86., 120., 114., 150.]
                elif bad_y2 is not None and 12 <= index < 22:
                    bbox = [86., 120., 114., bad_y2]
                else:
                    return [TrackedFrame([], withheld=(), epoch=self.epoch) for _ in frames]
                v = TrackedVehicle(1, 2, 'car', bbox, .9, timestamp, obs_count=6, epoch=self.epoch)
                v.source_bbox = tuple(bbox)
                return [TrackedFrame([v], withheld=(), epoch=self.epoch) for _ in frames]
            def update_velocities(self, vehicles, project, timestamp=None):
                DetectorTracker.update_velocities(self, vehicles, project, timestamp=timestamp)
        return RealVelocityDetector

    def run_red_wait(tag, bad_y2):
        settings = run_payload(video_factory)
        settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': .1,
                              'fragment_continuation': True, 'exit_persistence_sec': 0.3}
        class Manual(ObservationSession):
            def _worker(self):self.ready.set()
        session = Manual(settings, tmp_path/f'nan-vel-{tag}',
                         detector_factory=make_velocity_detector(bad_y2), anfis_factory=FakeANFIS)
        try:
            for _ in range(36):session._process_batch()
            row = next(iter(session.engine.waits.records.values()))
            return row['red_wait_sec'], row['tracking_state'], row['crossed_stop_line'], len(session.engine.waits.records)
        finally:
            session.state = session.applied_state = 'stopped';session._cleanup()

    wait_missing, state_missing, crossed_missing, count_missing = run_red_wait('missing', None)
    assert count_missing == 1 and not crossed_missing and wait_missing > 0.75
    for tag, bad_y2 in (('nan', float('nan')), ('overflow_1e40', 1e40), ('overflow_3e38', 3e38)):
        wait_bad, state_bad, crossed_bad, count_bad = run_red_wait(tag, bad_y2)
        assert count_bad == 1 and crossed_bad is False
        assert state_bad == state_missing
        assert wait_bad == pytest.approx(wait_missing, abs=1e-6)


def test_observation_start_rejects_overflowing_exit_persistence_int_with_422(video_factory, tmp_path, monkeypatch):
    """Sol Round 28: out-of-range JSON integers (e.g. 10**309) in policy.exit_persistence_sec
    must raise ValueError in RunPolicy and return HTTP 422 (never HTTP 500 OverflowError)."""
    from functools import partial
    from fastapi.testclient import TestClient
    from src.experiment.contracts import RunPolicy
    from src.ui.app import app
    import src.ui.observation_api as api
    from src.ui.observation_session import ObservationManager

    for bad in (10**309, -(10**309)):
        with pytest.raises(ValueError, match='finite'):
            RunPolicy(exit_persistence_sec=bad)

    manager = ObservationManager(
        tmp_path/'runs', tmp_path/'profiles',
        partial(ObservationSession, detector_factory=ScriptedDetector, anfis_factory=FakeANFIS),
    )
    monkeypatch.setattr(api, 'observations', manager)
    with TestClient(app) as client:
        for bad in (10**309, -(10**309)):
            payload = run_payload(video_factory)
            payload['policy']['exit_persistence_sec'] = bad
            response = client.post('/api/observation/start',
                                   content=json.dumps(payload),
                                   headers={'content-type': 'application/json'})
            assert response.status_code == 422, response.text
    if manager.current:
        manager.current.stop()


@pytest.mark.parametrize('policy_extra', [
    {'fragment_continuation': True},
    {'exit_persistence_sec': 0.3},
])
def test_session_exact_max_gap_replay_preserves_detector_velocity_and_red_wait(video_factory, tmp_path, policy_extra):
    """Sol Round 31: five 10 fps RED frames with max_gap_sec=0.1 (where 0.30000000000000004 - 0.2 > 0.1)
    must not reset DetectorTracker velocity history when either native option is enabled, preserving
    STOPPED state and red_wait_sec=0.2 (matching max_gap_sec=0.100000001)."""
    import cv2
    import numpy as np
    from src.perception.detector_tracker import DetectorTracker

    red_path = tmp_path/'red_10fps.avi'
    writer = cv2.VideoWriter(str(red_path), cv2.VideoWriter_fourcc(*'MJPG'), 10.0, (320, 240))
    frame = np.full((240, 320, 3), (0, 0, 255), np.uint8)
    for _ in range(8):
        writer.write(frame)
    writer.release()

    class StationaryRealVelocityDetector(ScriptedDetector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.track_obs_counts = {}
            self.max_history_len = 10
            self.last_seen_timestamp = 0.0
            self.max_gap_sec = 5.0
            self.gap_tolerance = 0.0
        def track(self, frames, timestamp=None):
            bbox = [86., 120., 114., 150.]
            v = TrackedVehicle(1, 2, 'car', bbox, .9, timestamp, obs_count=1, epoch=self.epoch)
            v.source_bbox = tuple(bbox)
            return [TrackedFrame([v], withheld=(), epoch=self.epoch) for _ in frames]
        def update_velocities(self, vehicles, project, timestamp=None):
            DetectorTracker.update_velocities(self, vehicles, project, timestamp=timestamp)

    settings = run_payload(video_factory)
    settings['system']['source'] = str(red_path)
    settings['acquisition']['source'] = str(red_path)
    settings['shared_test'] = True
    settings['policy'] = {'debounce_frames': 1, 'stop_persistence_sec': 0.0, 'max_gap_sec': 0.1, **policy_extra}
    class Manual(ObservationSession):
        def _worker(self):self.ready.set()
    session = Manual(settings, tmp_path/f"gap10-{list(policy_extra)[0]}",
                     detector_factory=StationaryRealVelocityDetector, anfis_factory=FakeANFIS)
    try:
        assert session.detectors['acquisition'].max_gap_sec == 0.1
        for _ in range(5):
            session._process_batch()
        row = next(iter(session.engine.waits.records.values()))
        assert row['tracking_state'] == 'STOPPED'
        assert row['red_wait_sec'] == pytest.approx(0.2, abs=1e-6)
        assert row['unknown_sec'] == pytest.approx(0.2, abs=1e-6)
    finally:
        session.state = session.applied_state = 'stopped'; session._cleanup()
