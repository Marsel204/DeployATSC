"""Regressions from the independent Sol High review; no native weights required."""
import ast
import importlib.metadata
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from src.perception.detector_tracker import DetectorTracker
from src.ui.observation_session import ObservationSession
from test_observation_sessions import FakeANFIS, run_payload
from test_on_demand_camera1 import factory, scripted_pair, step


def installed_method(path, name):
    tree = ast.parse(path.read_text())
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_still_predictor_dispatches_one_real_forward_including_first_call(monkeypatch, device):
    """Exercise app dispatch with installed predictor/backend call nodes, not a predict-call count."""
    import src.perception.detector_tracker as module
    library = Path(importlib.metadata.distribution('ultralytics').locate_file('ultralytics'))
    warmup = installed_method(library/'nn/autobackend.py', 'warmup')
    warmup.returns = None
    warmup.decorator_list = []
    for argument in warmup.args.args:
        argument.annotation = None

    class Boxes:
        def __getitem__(self, key):return self
        def __setitem__(self, key, value):pass
        def __imul__(self, value):return self

    class BasePredictor:
        def __init__(self, *args, **kwargs):self.done_warmup = False

    detect_module = ModuleType('ultralytics.models.yolo.detect.predict')
    detect_module.DetectionPredictor = BasePredictor
    nms_module = ModuleType('ultralytics.utils.nms')
    nms_module.non_max_suppression = lambda boxes: None
    monkeypatch.setitem(sys.modules, detect_module.__name__, detect_module)
    monkeypatch.setitem(sys.modules, nms_module.__name__, nms_module)
    monkeypatch.setitem(sys.modules, 'torchvision', ModuleType('torchvision'))
    namespace = {'torch': SimpleNamespace(rand=lambda *args, **kwargs: Boxes())}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[warmup], type_ignores=[])),
                 str(library/'nn/autobackend.py'), 'exec'), namespace)
    stream = installed_method(library/'engine/predictor.py', 'stream_inference')
    warmup_if = next(n for n in ast.walk(stream) if isinstance(n, ast.If)
                     and isinstance(n.test, ast.UnaryOp)
                     and isinstance(n.test.operand, ast.Attribute)
                     and n.test.operand.attr == 'done_warmup')
    forward = next(n for n in ast.walk(stream) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
                   and n.value.func.attr == 'inference')
    calls = compile(ast.fix_missing_locations(ast.Module(body=[warmup_if, forward], type_ignores=[])),
                    str(library/'engine/predictor.py'), 'exec')
    forwards = []
    backend = SimpleNamespace(format='pt', device=SimpleNamespace(type=device), end2end=False,
                              fp16=False, forward=lambda image: forwards.append(image))
    backend.warmup = namespace['warmup'].__get__(backend)

    class Model:
        def __init__(self):self.predictor = None; self.created = 0
        def predict(self, source, predictor=None, **kwargs):
            # Match installed YOLO.predict's predictor-type reuse condition.
            if self.predictor is None or (predictor and type(self.predictor) is not predictor):
                self.predictor = (predictor or BasePredictor)()
                self.predictor.model = backend
                self.predictor.inference = backend.forward
                self.created += 1
            exec(calls, {'self': self.predictor, 'im': source, 'args': (), 'kwargs': {}})
            return []

    helper = getattr(module, '_still_predictor_type', None)
    if helper:helper.cache_clear()
    try:
        detector = DetectorTracker.__new__(DetectorTracker)
        detector.model = Model()
        detector.conf_threshold = .28; detector.iou_threshold = .5; detector.imgsz = 640
        first = np.full((24, 32, 3), 17, np.uint8)
        second = np.full((24, 32, 3), 42, np.uint8)
        detector.detect_once(first, timestamp=1.)
        assert len(forwards) == 1 and forwards[0] is first
        detector.detect_once(second, timestamp=2.)
        assert len(forwards) == 2 and forwards[1] is second
        assert detector.model.created == 1  # do not rebuild the backend for every click
    finally:
        if helper:helper.cache_clear()


def make_session(monkeypatch, tmp_path, video_factory):
    import src.ui.observation_session as module
    monkeypatch.setattr(module, 'ReplayPair', scripted_pair())
    payload = run_payload(video_factory)
    payload['retention_budget'] = 1
    made = []; gate = threading.Event()
    session = ObservationSession(payload, tmp_path/'runs',
        detector_factory=factory(made, gate=gate), anfis_factory=FakeANFIS)
    return session, made[0], gate


@pytest.mark.parametrize('terminal', ['cancel', 'invalidate'])
def test_discarded_completions_are_archived_and_retried_without_new_inference(monkeypatch, tmp_path, video_factory, terminal):
    session, detector, gate = make_session(monkeypatch, tmp_path, video_factory)
    rows = {}
    try:
        for n in range(4):
            step(session)
            detector.entered.clear(); gate.clear()
            request = f'{terminal}-{n}'
            session.predict(request)
            assert detector.entered.wait(2)
            if terminal == 'cancel':session.cancel(request)
            else:session.engine.invalidate('test_discontinuity')
            assert session.persist()
            assert request in session.engine.predictions.records
            assert session.engine.predictions.records[request]['completed_at'] is None
            gate.set(); session.engine.predictions.await_idle()
            rows[request] = session.predict(request)
            assert rows[request]['completed_at'] is not None
            assert rows[request]['snapshot'] is None and rows[request]['recommended_green_sec'] is None
            assert rows[request]['camera_analysis']['state'] == 'discarded_after_'+rows[request]['status']
            step(session)
            assert session.persist()
        assert len(session.engine.predictions.records) <= 1
        assert session.engine.retention.archived_counts['predictions'] == 3
        for request, original in rows.items():
            assert session.predict(request) == original
        assert len(detector.detect_calls) == 4
        assert session.engine.predictions.last_analysis is None
    finally:
        gate.set(); session.stop()


def test_failed_attempt_counts_time_and_never_retries_failed_id(monkeypatch, tmp_path, video_factory):
    session, detector, gate = make_session(monkeypatch, tmp_path, video_factory)
    original = detector.detect_once
    def fail(frame, timestamp=None):
        original(frame, timestamp)
        raise RuntimeError('native failure sentinel')
    monkeypatch.setattr(detector, 'detect_once', fail)
    gate.set()
    try:
        session.predict('failed'); session.engine.predictions.await_idle()
        row = session.predict('failed'); data = session.telemetry()
        assert row['status'] == 'failed' and row['error'] == 'native failure sentinel'
        assert row['camera_analysis']['inference_calls'] == len(detector.detect_calls) == 1
        assert row['snapshot'] is None and row['recommended_green_sec'] is None
        assert data['metrics'] is None
        assert data['runtime']['system']['frames'] == 1
        assert data['runtime']['system']['last_inference_ms'] > 0
        assert data['runtime']['system']['inference_total_ms'] > 0
        assert session.predict('failed') == row and len(detector.detect_calls) == 1
    finally:
        session.stop()
