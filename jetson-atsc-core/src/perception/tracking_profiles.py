"""Named tracking-floor choices. Still-image and fallback confidence stay separate."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class TrackingProfile:
    name: str
    yaml_path: str
    confidence_floor: float | None = None

    @property
    def tracker_argument(self) -> str:
        return 'bytetrack.yaml' if self.name == 'baseline' else str(ROOT / self.yaml_path)

    def track_confidence(self, conf_threshold: float) -> float:
        return conf_threshold if self.confidence_floor is None else self.confidence_floor

    def provenance(self, conf_threshold: float, iou_threshold: float) -> dict:
        if self.name == 'baseline':
            path = Path(find_spec('ultralytics').origin).parent / 'cfg/trackers/bytetrack.yaml'
            public_path = 'ultralytics/cfg/trackers/bytetrack.yaml'
        else:
            path = ROOT / self.yaml_path
            public_path = self.yaml_path
        content = path.read_bytes()
        return {
            'name': self.name, 'status': 'applied',
            'tracker_yaml': {'path': public_path, 'sha256': hashlib.sha256(content).hexdigest()},
            'effective_thresholds': {**yaml.safe_load(content),
                'conf': self.track_confidence(conf_threshold), 'predict_conf': conf_threshold,
                'iou': iou_threshold, 'agnostic_nms': False},
        }


TRACKING_PROFILES = MappingProxyType({
    'baseline': TrackingProfile('baseline', 'bytetrack.yaml'),
    'low_score': TrackingProfile('low_score', 'configs/trackers/low_score.yaml', .10),
})


def get_tracking_profile(name: str) -> TrackingProfile:
    if not isinstance(name, str) or name not in TRACKING_PROFILES:
        raise ValueError('Unknown tracking profile; choose baseline or low_score')
    return TRACKING_PROFILES[name]


def tracking_profile_provenance(component) -> dict:
    describe = getattr(component, 'tracking_profile_provenance', None)
    if callable(describe):
        return describe()
    return {'name': 'baseline', 'status': 'injected_component',
            'tracker_yaml': {'path': None, 'sha256': None}, 'effective_thresholds': None}


def apply_tracking_profile(component, name: str) -> dict:
    """Preserve injected factory signatures; capability means setter plus descriptor."""
    get_tracking_profile(name)
    setter = getattr(component, 'set_tracking_profile', None)
    describe = getattr(component, 'tracking_profile_provenance', None)
    if callable(setter) and callable(describe):
        setter(name)
    return tracking_profile_provenance(component)
