import json
from dataclasses import asdict
import pytest

from src.experiment.contracts import RunPolicy, VehicleObservation, vehicle_key


@pytest.mark.parametrize("settings", [
    {"offset_sec": float("nan")}, {"moving_speed_mps": .1},
    {"max_gap_sec": 0}, {"debounce_frames": 0}, {"operation": "control"},
])
def test_invalid_run_policy_rejected(settings):
    with pytest.raises(ValueError):
        RunPolicy(**settings)


def test_run_policy_roundtrip_and_identity_namespace():
    policy = RunPolicy(offset_sec=-2.5)
    assert RunPolicy(**json.loads(json.dumps(asdict(policy)))) == policy
    assert vehicle_key("run-a", 0, 7) != vehicle_key("run-b", 0, 7)
    assert vehicle_key("run-a", 0, 7) != vehicle_key("run-a", 1, 7)


def test_missing_velocity_is_unknown_not_stopped():
    vehicle = VehicleObservation(7, "car", None, (50, 50))
    assert vehicle.speed_mps is None
    assert not vehicle.velocity_valid
