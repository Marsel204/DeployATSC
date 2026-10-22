"""Device selection and forwarding must agree on GPU and CPU hosts."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.perception.detector_tracker import DetectorTracker


@pytest.mark.parametrize("cuda_available,requested,expected", [
    (True, None, "cuda:0"),
    (False, None, "cpu"),
    (True, "cpu", "cpu"),
    (True, "cuda:1", "cuda:1"),
])
def test_detector_selects_available_device(monkeypatch, cuda_available, requested, expected):
    monkeypatch.setattr(DetectorTracker, "_load_model", lambda self: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    options = {} if requested is None else {"device": requested}
    detector = DetectorTracker("unused.pt", **options)
    assert detector.device == expected


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("tracking_fails", [False, True])
def test_device_reaches_tracking_and_prediction_fallback(monkeypatch, device, tracking_fails):
    monkeypatch.setattr(DetectorTracker, "_load_model", lambda self: None)
    detector = DetectorTracker("unused.pt", device=device)
    calls = []

    def track(**kwargs):
        calls.append(("track", kwargs))
        if tracking_fails:
            raise RuntimeError("synthetic tracker failure")
        return []

    def predict(**kwargs):
        calls.append(("predict", kwargs))
        return []

    detector.model = SimpleNamespace(track=track, predict=predict)
    detector.track([np.zeros((20, 20, 3), dtype=np.uint8)], timestamp=0)
    assert [name for name, _ in calls] == (["track", "predict"] if tracking_fails else ["track"])
    for name, options in calls:
        assert options.get("device") == device, f"{name} must use the selected device"
