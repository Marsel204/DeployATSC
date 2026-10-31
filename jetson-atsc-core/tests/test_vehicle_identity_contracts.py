from dataclasses import FrozenInstanceError

import pytest

from src.experiment.identity_contracts import IdentityDetection, IdentityPolicy


@pytest.mark.parametrize('settings', [
    {'mode': 'other'}, {'occlusion_retention_sec': float('nan')},
    {'occlusion_retention_sec': True}, {'max_estimated_gap_sec': 11},
    {'capacity': 0}, {'capacity': 1.5}, {'confirmation_observations': True},
    {'exit_persistence_sec': -1}, {'ambiguity_margin': 0},
])
def test_invalid_identity_policy_is_rejected(settings):
    with pytest.raises((ValueError, TypeError)):
        IdentityPolicy(**settings)


def test_default_is_native_and_evidence_is_immutable():
    assert IdentityPolicy().mode == 'native'
    item = IdentityDetection((0, 0, 20, 20), (10, 20))
    with pytest.raises(FrozenInstanceError):
        item.native_id = 4
    with pytest.raises(ValueError):
        IdentityDetection((0, 0, float('nan'), 20), (10, 20))
