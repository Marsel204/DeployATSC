"""Opt-in fragment continuation sees the frame's actual post-NMS detections
that the installed ByteTrack withheld (discarded or unconfirmed), identified by
the tracker's own detection index. Those boxes only veto links."""
import os
import time

import numpy as np
import pytest
import torch

from frozen_tracking import HIDES_UNCONFIRMED, FrozenTrackModel, failing_trackers, frozen_detector
from src.experiment.contracts import RunPolicy, VehicleObservation
from src.experiment.persistent_vehicle_waits import PersistentVehicleWaits
from src.experiment.vehicle_waits import VehicleWaits
from src.perception.detector_tracker import DetectorTracker, TrackedFrame
from src.ui.observation_session import _withheld_boxes

IMAGE = np.zeros((240, 320, 3), np.uint8)
FPS = 30


def box(x, conf=.95, cls=1, top=100, bottom=140, width=6):
    return [x-width/2, top, x+width/2, bottom, conf, cls]


def review_repro(competitor=True):
    """Sol's repro: 40x6 px boxes, x=50 on f0-4, new x=62 on f5, x=62 and x=44 on f6, all 0.95."""
    return [[box(50)] for _ in range(5)]+[[box(62)]]+[[box(62)]+([box(44)] if competitor else [])]


def observations(frame):
    return [VehicleObservation(v.track_id, v.class_name, None, v.bottom_center, stable_id=v.stable_id,
                               epoch=v.epoch, bbox=tuple(v.bbox), source_bbox=v.source_bbox) for v in frame]


def replay(frames, *, enabled=True, evidence=True, waits_type=VehicleWaits):
    detector = frozen_detector(FrozenTrackModel(frames))
    waits = waits_type('run', RunPolicy(fragment_continuation=enabled), lambda point:-1)
    tracked = []
    for f in range(len(frames)):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
        tracked.append(frame)
        unassigned = _withheld_boxes(frame, detector.epoch) if evidence else ()
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=unassigned)
    return waits, tracked, detector


def links(waits):
    return [(link['from']['local_id'], link['to']['local_id'])
            for row in waits.records.values() for link in row.get('identity_links', ())]


def test_review_repro_competitor_withheld_by_bytetrack_now_vetoes_the_link():
    waits, tracked, _ = replay(review_repro())
    # The installed tracker emits only the confirmed x=62 track on f6; x=44 is a new unconfirmed track.
    assert [(v.track_id, v.bbox) for v in tracked[6]] == [(2, [59., 100., 65., 140.])]
    assert [w[0] for w in tracked[6].withheld] == [(41., 100., 47., 140.)]
    assert tracked[6].withheld[0][1] == 'car' and tracked[6].withheld[0][2] == pytest.approx(.95)
    assert links(waits) == []
    assert set(waits.records) == {'run:acquisition:0:1', 'run:acquisition:0:2'}
    # Without the withheld box (the 6e7aa7a behaviour) the same frames link 1->2.
    blind, _, _ = replay(review_repro(), evidence=False)
    assert links(blind) == [(1, 2)]


def test_positive_control_without_competitor_still_links():
    waits, tracked, _ = replay(review_repro(competitor=False))
    assert tracked[6].withheld == ()
    assert links(waits) == [(1, 2)]
    row = waits.records['run:acquisition:0:1']
    assert [n['local_id'] for n in row['native_ids']] == [1, 2]
    assert waits.continuation_unavailable_frames == 0


def test_hidden_unconfirmed_frame_reports_every_detection_as_withheld():
    _, tracked, _ = replay(review_repro())
    if HIDES_UNCONFIRMED:
        assert list(tracked[5]) == []      # Ultralytics 8.4 hides the frame while a new track awaits confirmation
        assert [w[0] for w in tracked[5].withheld] == [(59., 100., 65., 140.)]
    else:
        # Ultralytics 8.3 passes the raw detection through untracked: an observation, not withheld.
        assert [(v.stable_id, v.bbox) for v in tracked[5]] == [(False, [59., 100., 65., 140.])]
        assert tracked[5].withheld == ()
    assert all(frame.withheld == () for frame in tracked[:5])


def test_own_source_box_is_excluded_by_tracker_index_despite_kalman_jitter():
    # A moving, size-jittering car: the emitted box is the Kalman state, not the detection.
    rng = np.random.default_rng(3)
    frames = []
    for f in range(12):
        x = 100+4*f+rng.normal(0, 1.5)
        # Plus a low-score box (below new_track_thresh): second association only, then discarded.
        frames.append([[x-10, 80+rng.normal(0, 1), x+10, 120+rng.normal(0, 1), .9, 1],
                       [250, 80, 262, 104, .2, 2]])
    detector = frozen_detector(FrozenTrackModel(frames))
    jittered = 0
    for f in range(12):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
        raw = tuple(float(v) for v in frames[f][0][:4])
        if f == 0:continue
        assert len(frame) == 1 and frame[0].track_id == 1
        jittered += not np.allclose(frame[0].bbox, raw, atol=.01)
        # Its own detection is never a competitor, however far the emitted box drifted from it.
        assert [w[0] for w in frame.withheld] == [(250., 80., 262., 104.)]
    assert jittered >= 8


def test_non_vehicle_and_degenerate_withheld_boxes_are_not_evidence():
    # Frame 1: the tracked car, a new valid car, a person and a zero-width car (all withheld by ByteTrack).
    degenerate = [200., 100., 200., 140., .95, 1]
    detector = frozen_detector(FrozenTrackModel([[box(50)], [box(50), box(120), box(160, cls=4), degenerate]]))
    detector.model.names = {0: 'motorcycle', 1: 'car', 2: 'bus', 3: 'truck', 4: 'person'}
    detector.track([IMAGE], timestamp=0.)
    frame, = detector.track([IMAGE], timestamp=1/FPS)
    assert [v.track_id for v in frame] == [1]
    # Only the valid vehicle-class box is evidence: the person and the zero-area box are not.
    assert [(w[0], w[1]) for w in frame.withheld] == [((117., 100., 123., 140.), 'car')]
    none = type('R', (), {'boxes': None})()
    calls = [(np.array([[0, 0, 1, 1.]]), np.array([4.])),              # valid person
             (np.array([[5, 5, 5, 9.]]), np.array([1.])),              # zero-width car
             (np.array([[5, 9, 8, 5.]]), np.array([1.])),              # inverted car
             (np.array([[0, 0, np.nan, 4.]]), np.array([1.]))]          # non-finite car
    for xyxy, cls in calls:
        assert detector._withheld(none, (xyxy, cls, np.array([.9]), np.zeros(0, int))) == ()


def test_callback_order_with_the_real_predictor_and_model(monkeypatch):
    """Real YOLO predictor, real stream_inference event order, real tracker callbacks;
    only the network's post-NMS output is scripted."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Final.pt')
    if not os.path.exists(path):
        pytest.skip('models/Final.pt not available')
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    scripted = iter(review_repro())
    events = []

    def postprocess(self, preds, img, orig_imgs):
        boxes = torch.tensor(next(scripted), dtype=torch.float32).reshape(-1, 6)
        events.append('postprocess')
        return [Results(orig_imgs[0], path='scripted', names=self.model.names, boxes=boxes)]
    monkeypatch.setattr(DetectionPredictor, 'postprocess', postprocess)
    detector = DetectorTracker(path, device='cpu', imgsz=64)
    assert getattr(detector, '_provenance', None) is None      # capture is opt-in
    assert detector.enable_withheld_evidence() is True
    hook = detector._provenance
    assert hook in detector.model.callbacks['on_predict_batch_start']

    def spy(predictor):
        events.append(('batch_start', len(getattr(predictor, 'trackers', ()))))
    detector.model.add_callback('on_predict_batch_start', spy)
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    image = np.zeros((64, 128, 3), np.uint8)
    for f in range(7):
        frame, = detector.track([image], timestamp=f/FPS)
        assert isinstance(frame, TrackedFrame) and frame.withheld is not None
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=_withheld_boxes(frame, detector.epoch))
    # Trackers existed at every batch start (on_predict_start ran first), before each postprocess.
    assert events[:2] == [('batch_start', 1), 'postprocess']
    assert all(e == ('batch_start', 1) for e in events[0::2]) and events[1::2] == ['postprocess']*7
    # Exactly one wrapper per tracker, even across repeated model.track registrations.
    trackers = detector.model.predictor.trackers
    assert len(trackers) == 1 and trackers[0].update._atsc_provenance is hook
    assert [w[0] for w in frame.withheld] == [(41., 100., 47., 140.)]
    assert links(waits) == []


def test_tracker_error_falls_back_without_evidence_and_leaves_no_stale_state():
    frames = review_repro(competitor=False)+[[box(62)]]
    model = FrozenTrackModel(frames, fail_on={6})
    failing_trackers(model)
    detector = frozen_detector(model)
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    seen = []
    for f in range(8):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
        seen.append(frame)
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=_withheld_boxes(frame, detector.epoch))
    assert model.predict_calls == 1
    # The failed frame fell back to untracked detections: no evidence, no IDs, no link guessed.
    assert seen[6].withheld is None and not any(v.stable_id for v in seen[6])
    assert detector._provenance.armed is False and detector._provenance.calls == []
    # The next frame carries only its own evidence.
    assert seen[7].withheld == () and [v.track_id for v in seen[7]] == [2]
    # ID 2 is first emitted on f7, with complete evidence and no competitor: only then may it link.
    assert links(waits) == [(1, 2)]
    assert waits.records['run:acquisition:0:1']['identity_links'][0]['linked_at'] == pytest.approx(7/FPS)


def test_reset_rewraps_new_trackers_and_epoch_mismatch_is_unavailable():
    detector = frozen_detector(FrozenTrackModel([[box(50)]]*4+[[box(50), box(150)]]*2))
    first, = detector.track([IMAGE], timestamp=0.)
    old = detector.model.predictor.trackers[0]
    detector.reset_tracking()
    assert detector.model.predictor is None
    second, = detector.track([IMAGE], timestamp=1/FPS)
    new = detector.model.predictor.trackers[0]
    assert new is not old and new.update._atsc_provenance is detector._provenance
    assert (first.epoch, second.epoch) == (0, 1)
    assert _withheld_boxes(first, detector.epoch) is None      # a frame of the previous tracker epoch
    assert _withheld_boxes(second, detector.epoch) == ()
    assert _withheld_boxes([], detector.epoch) is None           # any other detector's plain list
    for f in range(2, 6):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
    assert [w[0] for w in frame.withheld] == [(147., 100., 153., 140.)] or frame.withheld == ()


def test_evidence_is_frame_local():
    detector = frozen_detector(FrozenTrackModel([[box(50)], [box(50), box(150)], [box(50)], []]))
    frames = [detector.track([IMAGE], timestamp=f/FPS)[0] for f in range(4)]
    assert [len(f.withheld) for f in frames] == [0, 1, 0, 0]
    assert frames[1].withheld is not frames[2].withheld


def test_unavailable_evidence_declines_every_link_and_is_counted():
    detector = frozen_detector(FrozenTrackModel(review_repro(competitor=False)))
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    for f in range(7):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=None)
    assert links(waits) == [] and waits.continuation_unavailable_frames == 1


@pytest.mark.parametrize('evidence', [(), None, ((0., 0., 5., 5.),)])
def test_default_off_and_persistent_rows_ignore_evidence(evidence):
    def rows(enabled, unassigned, waits_type=VehicleWaits):
        detector = frozen_detector(FrozenTrackModel(review_repro()))
        if waits_type is VehicleWaits:
            waits = VehicleWaits('run', RunPolicy(fragment_continuation=enabled), lambda point:-1)
        else:
            from src.experiment.identity_contracts import IdentityPolicy
            waits = PersistentVehicleWaits('run', RunPolicy(fragment_continuation=enabled), lambda point:-1,
                                           IdentityPolicy(mode='persistent_v1'))
        for f in range(7):
            frame, = detector.track([IMAGE], timestamp=f/FPS)
            waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=unassigned)
        assert waits.continuation_unavailable_frames == 0
        return waits.rows(include_intervals=True)
    assert rows(False, evidence) == rows(False, ())
    assert rows(True, evidence, PersistentVehicleWaits) == rows(False, (), PersistentVehicleWaits)


@pytest.mark.parametrize('competitor', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_one_shot_iterators_and_lists_give_identical_rows(enabled, competitor):
    _, tracked, _ = replay(review_repro(competitor=competitor))
    # With the competitor, frame 6 carries a nonempty withheld box that must veto through a one-shot iterator.
    assert bool(tracked[6].withheld) is competitor
    def run(wrap):
        waits = VehicleWaits('run', RunPolicy(fragment_continuation=enabled), lambda point:-1)
        for f, frame in enumerate(tracked):
            waits.observe(wrap(observations(frame)), 'RED', f/FPS, f+1, 1, unassigned=wrap(_withheld_boxes(frame, 0)))
        return waits.rows(include_intervals=True)
    as_list = run(list)
    assert as_list == run(iter) == run(lambda items:(x for x in items)) == run(tuple)
    assert len(as_list) == (1 if enabled and not competitor else 2)


def test_crowded_frame_with_300_boxes_and_300_withheld_stays_bounded():
    # 300 tracked rows lost at once, 300 newborns each uniquely 14 px further, 300 withheld
    # boxes far away: every pair survives the gates, so every veto scan runs over 600 boxes.
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    def car(i, y, x):
        return VehicleObservation(i, 'car', None, (x, y), stable_id=True, bbox=(x-4, y-40, x+4, y))
    grid = [(20+60*(k % 20), 100+90*(k//20)) for k in range(300)]
    for f in range(6):
        waits.observe([car(1+k, y+2*f, x) for k, (x, y) in enumerate(grid)], 'RED', f/FPS, f+1, 1)
    newborns = [car(10_000+k, y+14, x) for k, (x, y) in enumerate(grid)]
    withheld = [(x+25, y-40, x+30, y) for x, y in grid]
    # 300 x 300 pairs exceed the per-frame pair cap: the frame is skipped in bounded time.
    started = time.perf_counter()
    waits._continue_fragments(newborns, 7/FPS, withheld)
    assert time.perf_counter()-started < .05 and waits.continuation_skipped_frames == 1
    # At the cap (64 x 64 pairs), with 300 observations and 300 withheld boxes per veto scan.
    timings = []
    for trial in range(5):
        waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
        for f in range(6):
            waits.observe([car(1+k, y+2*f, x) for k, (x, y) in enumerate(grid[:64])], 'RED', f/FPS, f+1, 1)
        crowd = newborns[:64]+[VehicleObservation(None, "car", None, (x+30, y), bbox=(x+26, y-40, x+34, y))
                               for x, y in grid[64:]]
        started = time.perf_counter()
        waits._continue_fragments(crowd, 7/FPS, withheld)
        timings.append(time.perf_counter()-started)
        assert len(links(waits)) == 64
    assert sorted(timings)[2] < .1, timings


# --- declared minimum Ultralytics (8.3.0) and the installed 8.4 line --------------------

def tracker_postprocess_callbacks(model):
    from ultralytics.trackers.track import on_predict_postprocess_end
    return [cb for cb in model.callbacks['on_predict_postprocess_end'] if getattr(cb, 'func', None) is on_predict_postprocess_end]


def counting_updates(model):
    """Count tracker.update calls per frame (installed before the provenance wrapper)."""
    counts = []
    def wrap(predictor):
        for tracker in getattr(predictor, 'trackers', ()):
            if getattr(tracker, '_counted', False):continue
            update = tracker.update
            def counted(*args, _update=update, **kwargs):
                counts[-1] += 1
                return _update(*args, **kwargs)
            tracker.update = counted; tracker._counted = True
    model.callbacks['on_predict_batch_start'].insert(0, wrap)
    return counts


def test_reset_registers_one_tracker_callback_and_keeps_evidence_complete():
    # 8.3.0's Model.track re-registers whenever the predictor has no trackers and appends:
    # without dropping the old pair, each reset stacked another tracker.update per frame.
    model = FrozenTrackModel([[box(50)]]*7+[[box(50), box(150)]]*2)
    counts = counting_updates(model)
    detector = frozen_detector(model)
    seen = []
    for f in range(9):
        if f in (3, 5):detector.reset_tracking()
        counts.append(0)
        seen.append(detector.track([IMAGE], timestamp=f/FPS)[0])
        assert len(tracker_postprocess_callbacks(model)) == 1
    assert counts == [1]*9
    assert all(frame.withheld is not None for frame in seen)
    assert [frame.epoch for frame in seen] == [0]*3+[1]*2+[2]*4
    # Frame 7 (epoch 2): box(150) is a new unconfirmed track while box(50) is emitted.
    assert [w[0] for w in seen[7].withheld] == [(147., 100., 153., 140.)]


def batched_detector(per_call):
    model = FrozenTrackModel(per_call, batched=True)
    detector = frozen_detector(model)
    detector.is_tensorrt = False      # a PyTorch model accepts a two-image batch
    return model, detector


@pytest.mark.parametrize('empty_first', [True, False])
def test_mixed_batch_with_an_empty_image_keeps_per_image_provenance(empty_first):
    # One image never has detections (8.3.0 skips its tracker.update); the other is Sol's repro.
    order = (lambda empty, scene:[empty, scene]) if empty_first else (lambda empty, scene:[scene, empty])
    model, detector = batched_detector([order([], frame) for frame in review_repro()])
    counts = counting_updates(model)
    withheld = []
    for f, detections in enumerate(review_repro()):
        counts.append(0)
        empty, scene = order(*detector.track([IMAGE, IMAGE.copy()], timestamp=f/FPS))
        assert empty.withheld == () and list(empty) == []
        # Exact provenance: the raw sources of the emitted boxes plus the withheld boxes are
        # exactly this image's detections, each once (a wrong box of the same count fails).
        raw = sorted(tuple(float(v) for v in d[:4]) for d in detections)
        assert scene.withheld is not None
        assert sorted([v.source_bbox for v in scene]+[w[0] for w in scene.withheld]) == raw
        withheld.append([w[0] for w in scene.withheld])
    # The scene's own withheld boxes, frame by frame, exactly. 8.3 skips the empty image's update, so
    # the scene tracks as if alone. 8.4 runs the one shared (non-stream) tracker on both images: with
    # the empty image first, the scene's first update is not tracker frame 1, so x=50 is never confirmed.
    car50, car62, car44 = (47., 100., 53., 140.), (59., 100., 65., 140.), (41., 100., 47., 140.)
    if not HIDES_UNCONFIRMED:
        expected = [[]]*6+[[car44]]
    elif empty_first:
        expected = [[car50]]*5+[[car62], [car62, car44]]
    else:
        expected = [[]]*5+[[car62], [car62, car44]]
    assert withheld == expected
    # 8.3.0 runs no update for the empty image; 8.4 runs one for each image.
    assert set(counts) == ({2} if HIDES_UNCONFIRMED else {1})


def test_mixed_batch_with_stacked_tracker_callbacks_fails_closed_per_image():
    # Stacked registrations (8.3.0 before the reset fix) ran two updates for the image with
    # detections and none for the empty one: positional pairing gave image 0 image 1's call.
    from functools import partial
    from ultralytics.trackers.track import on_predict_postprocess_end
    model, detector = batched_detector([[[], [box(50), box(150, conf=.2)]]]*2)
    detector.track([IMAGE, IMAGE.copy()], timestamp=0.)
    model.add_callback('on_predict_postprocess_end', partial(on_predict_postprocess_end, persist=True))
    empty, scene = detector.track([IMAGE, IMAGE.copy()], timestamp=1/FPS)
    # 8.3.0 skips the empty image (no detections: nothing withheld); 8.4 updates it twice too.
    assert empty.withheld == (None if HIDES_UNCONFIRMED else ())
    assert scene.withheld is None


class FakeResult:
    def __init__(self, image, n=0):
        self.orig_img = image
        self.boxes = type('B', (), {'id': None, '__len__': lambda self:n})()


def test_frame_evidence_attribution_fails_closed():
    detector = frozen_detector(FrozenTrackModel([]), evidence=False)
    real = detector._frame_evidence
    detector._frame_evidence = lambda *args:[withheld for withheld, _ in real(*args)]
    a, b = np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2, 3), np.uint8)
    call = (np.array([[0, 0, 4, 4.]]), np.array([1.]), np.array([.9]), np.zeros(0, int))
    one = FakeResult(a, 1)
    # One call per image, attributed by image identity regardless of order.
    assert detector._frame_evidence([FakeResult(b), one], [(a, call)]) == [(), ()]
    # A call for an image not in the batch, or ambiguous between two results: nothing is trusted.
    assert detector._frame_evidence([one, FakeResult(b)], [(np.zeros(1), call)]) == [None, None]
    assert detector._frame_evidence([one, FakeResult(a, 1)], [(a, call)]) == [None, None]
    # An image with detections and no update call, or with two calls, has no evidence.
    assert detector._frame_evidence([one, FakeResult(b, 1)], [(a, call)]) == [(), None]
    assert detector._frame_evidence([one], [(a, call), (a, call)]) == [None]
    # An unreadable call poisons only its own image.
    assert detector._frame_evidence([one, FakeResult(b, 1)], [(a, None), (b, call)]) == [None, ()]


def test_capture_is_opt_in_and_costs_nothing_when_off():
    model = FrozenTrackModel(review_repro())
    detector = frozen_detector(model, evidence=False)
    for f in range(7):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
        assert isinstance(frame, TrackedFrame) and frame.withheld is None
    assert getattr(detector, '_provenance', None) is None
    assert not any(isinstance(cb, type(None)) or getattr(cb, '__class__', None).__name__ == '_TrackerProvenance'
                   for cbs in model.callbacks.values() for cb in cbs)
    assert not hasattr(detector.model.predictor.trackers[0].update, '_atsc_provenance')
    # Without evidence, continuation declines the link it would otherwise make.
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    detector = frozen_detector(FrozenTrackModel(review_repro(competitor=False)), evidence=False)
    for f in range(7):
        frame, = detector.track([IMAGE], timestamp=f/FPS)
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=_withheld_boxes(frame, detector.epoch))
    assert links(waits) == [] and waits.continuation_unavailable_frames == 1


def test_omitted_evidence_is_unavailable_not_empty():
    _, tracked, _ = replay(review_repro(competitor=False))
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    for f, frame in enumerate(tracked):
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1)
    assert links(waits) == [] and waits.continuation_unavailable_frames == 1


def test_emitted_boxes_carry_their_raw_source_and_missing_attribution_fails_closed():
    _, tracked, _ = replay(review_repro())
    for frame in tracked:
        assert all(v.source_bbox is not None for v in frame)
    # The smoothed emitted box and its raw detection are both reported; the detection is exact.
    assert tracked[4][0].source_bbox == (47., 100., 53., 140.)
    # A frame claiming complete withheld evidence with an unattributed emitted box is unavailable.
    frame = TrackedFrame(list(tracked[4]), withheld=(), epoch=0)
    assert _withheld_boxes(frame, 0) == ()
    frame[0].source_bbox = None
    assert _withheld_boxes(frame, 0) is None
    # Evidence without sources (an emitted row the call cannot explain) is withdrawn by the detector.
    detector = frozen_detector(FrozenTrackModel([]), evidence=False)
    one = FakeResult(np.zeros((2, 2, 3), np.uint8), 2)
    assert detector._evidence(one, (np.array([[0, 0, 4, 4.]]), np.array([1.]), np.array([.9]), np.zeros(0, int))) == (None, None)


# --- the real predictor (models/Final.pt), only the network's post-NMS output scripted -------

def real_detector(monkeypatch, frames, fail=lambda frame:False):
    """DetectorTracker on the real YOLO predictor and installed ByteTrack; ``fail(frame)`` makes
    BYTETracker.update raise on that frame (each call, so a fallback that re-runs it fails too)."""
    path = os.path.join(os.path.dirname(__file__), '..', 'models', 'Final.pt')
    if not os.path.exists(path):
        pytest.skip('models/Final.pt not available')
    from ultralytics.engine.results import Results
    from ultralytics.models.yolo.detect.predict import DetectionPredictor
    from ultralytics.trackers.byte_tracker import BYTETracker
    current = [0]
    updates = []

    def postprocess(self, preds, img, orig_imgs):
        boxes = torch.tensor(frames[current[0]], dtype=torch.float32).reshape(-1, 6)
        return [Results(orig_imgs[0], path='scripted', names=self.model.names, boxes=boxes)]
    original = BYTETracker.update

    def update(self, *args, **kwargs):
        updates.append(current[0])
        if fail(current[0]):
            raise RuntimeError('injected tracker failure')
        return original(self, *args, **kwargs)
    monkeypatch.setattr(DetectionPredictor, 'postprocess', postprocess)
    monkeypatch.setattr(BYTETracker, 'update', update)
    detector = DetectorTracker(path, device='cpu', imgsz=64)
    assert detector.enable_withheld_evidence() is True
    image = IMAGE.copy()      # 240x320: no box is clipped

    def step(f):
        current[0] = f
        frame, = detector.track([image], timestamp=f/FPS)
        return frame
    return detector, step, updates


def test_real_predictor_tracked_competitor_vetoes_by_its_raw_source_box(monkeypatch):
    """Sol's probe: ID 1 lost at x=100 (gate .35 x 40 px = 14 px); newborn at x=112; an existing
    track's detection jumps to x=86.1 (inside the gate) while its Kalman box stays outside."""
    def frames(competitor_x):
        # 10x40 px boxes: the competitor (ID 2) sits at x=84 and its detection steps to competitor_x.
        steady = [box(100, width=10), box(84, width=10)]
        return [steady]*5+[[box(112, width=10), box(84, width=10)]]+[[box(112, width=10), box(competitor_x, width=10)]]
    def run(competitor_x, attributed=True):
        detector, step, _ = real_detector(monkeypatch, frames(competitor_x))
        waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
        seen = []
        for f in range(7):
            frame = step(f)
            seen.append(frame)
            if not attributed:
                for v in frame:v.source_bbox = tuple(v.bbox)
            waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=_withheld_boxes(frame, detector.epoch))
        return waits, seen
    waits, seen = run(86.1)
    newborn, = [v for v in seen[6] if v.track_id not in (1, 2)]
    competitor, = [v for v in seen[6] if v.track_id == 2]
    smoothed = (competitor.bbox[0]+competitor.bbox[2])/2
    raw = (competitor.source_bbox[0]+competitor.source_bbox[2])/2
    assert raw == pytest.approx(86.1, abs=1e-4) and abs(100-raw) <= 14 < abs(100-smoothed)
    assert seen[6].withheld == ()          # the competitor is emitted, not withheld
    assert links(waits) == []
    # Without the raw source the same frames link 1 -> newborn (the reported failure).
    blind, _ = run(86.1, attributed=False)
    assert links(blind) == [(1, newborn.track_id)]
    # Positive control: the competitor's raw box stays outside the gate too.
    control, _ = run(85.)
    assert links(control) == [(1, newborn.track_id)]


@pytest.mark.parametrize('failing', [{6}, {6, 7}])
def test_real_predictor_tracker_failure_falls_back_to_detection_only(monkeypatch, failing):
    """One-shot and repeated BYTETracker.update failures: the fallback never runs the tracker (8.3.x
    callbacks would run it in predict mode), returns untracked detections without evidence, and the
    next frame tracks normally with exactly one tracker registration."""
    from ultralytics.trackers.track import on_predict_postprocess_end
    frames = review_repro(competitor=False)+[[box(62)]]*4
    detector, step, updates = real_detector(monkeypatch, frames, fail=lambda f:f in failing)
    waits = VehicleWaits('run', RunPolicy(fragment_continuation=True), lambda point:-1)
    seen = []
    for f in range(len(frames)):
        frame = step(f)          # never raises
        seen.append(frame)
        waits.observe(observations(frame), 'RED', f/FPS, f+1, 1, unassigned=_withheld_boxes(frame, detector.epoch))
        registered = [cb for cb in detector.model.callbacks['on_predict_postprocess_end']
                      if getattr(cb, 'func', cb) is on_predict_postprocess_end]
        assert len(registered) == 1
    for f in failing:
        assert [v.bbox for v in seen[f]] == [[59., 100., 65., 140.]]
        assert not any(v.stable_id for v in seen[f]) and seen[f].withheld is None
        assert all(v.source_bbox is None for v in seen[f])
        # The tracker ran once, for the failed track() call; never again in the fallback predict.
        assert updates.count(f) == 1
    resumed = max(failing)+1
    assert all(updates.count(f) == 1 for f in range(len(frames)))
    assert seen[resumed].withheld is not None and [v.stable_id for v in seen[resumed]] == [True]
    assert detector._provenance.armed is False and detector._provenance.calls == []
    # Missing evidence is never guessed through: a link, if any, is made only on a tracked frame.
    for row in waits.records.values():
        for link in row.get('identity_links', ()):
            assert round(link['linked_at']*FPS) >= resumed
    # Reset after a fallback: a new epoch, one registration, normal evidence.
    detector.reset_tracking()
    frame = step(len(frames)-1)
    assert frame.epoch == 1 and frame.withheld is not None
