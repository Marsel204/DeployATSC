"""Immutable evidence and run-level policy for application-owned vehicle IDs."""
from dataclasses import dataclass
import math

ASSOCIATION_VERSION = 'joint_image_motion_appearance_v2'


@dataclass(frozen=True)
class IdentityPolicy:
    mode: str = 'native'
    occlusion_retention_sec: float = 10.
    max_estimated_gap_sec: float = 5.
    gap_accounting: str = 'bounded_stationary_v1'
    confirmation_observations: int = 3
    confirmation_sec: float = .2
    capacity: int = 256
    appearance_threshold: float = .55
    ambiguity_margin: float = .12
    exit_persistence_sec: float = .3

    def __post_init__(self):
        if self.mode not in ('native', 'persistent_v1'):
            raise ValueError('Identity mode must be native or persistent_v1')
        if self.gap_accounting != 'bounded_stationary_v1':
            raise ValueError('Unsupported gap accounting policy')
        for key, bounds in {'occlusion_retention_sec': (.1, 60),
                'max_estimated_gap_sec': (0, 30), 'confirmation_sec': (0, 5),
                'appearance_threshold': (.01, 1), 'ambiguity_margin': (.001, .5),
                'exit_persistence_sec': (0, 5)}.items():
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'Invalid identity setting: {key}')
        for key, bounds in {'capacity': (1, 1024), 'confirmation_observations': (2, 30)}.items():
            value = getattr(self, key)
            if type(value) is not int or not bounds[0] <= value <= bounds[1]:
                raise ValueError(f'Invalid identity setting: {key}')
        if self.max_estimated_gap_sec > self.occlusion_retention_sec:
            raise ValueError('Estimated gap cannot exceed identity retention')


@dataclass(frozen=True)
class IdentityDetection:
    bbox: tuple[float, float, float, float]
    point: tuple[float, float]
    class_name: str = 'car'
    native_id: int | None = None
    native_epoch: int = 0
    native_stable: bool = True
    ground: tuple[float, float] | None = None
    appearance: tuple[float, ...] | None = None
    appearance_reliable: bool = False
    in_region: bool = True

    def __post_init__(self):
        for key, length in (('bbox', 4), ('point', 2), ('ground', 2)):
            value = getattr(self, key)
            if value is None and key == 'ground':
                continue
            if len(value) != length or not all(math.isfinite(x) for x in value):
                raise ValueError(f'Invalid identity {key}')
            object.__setattr__(self, key, tuple(float(x) for x in value))
        if self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError('Identity box must have positive area')
        if self.appearance is not None:
            if not 1 <= len(self.appearance) <= 256 or not all(math.isfinite(x) and x >= 0 for x in self.appearance):
                raise ValueError('Invalid appearance descriptor')
            norm = math.sqrt(sum(x*x for x in self.appearance))
            if norm <= 0:
                raise ValueError('Empty appearance descriptor')
            object.__setattr__(self, 'appearance', tuple(float(x)/norm for x in self.appearance))
        if self.native_id is not None and (type(self.native_id) is not int or self.native_id < 0):
            raise ValueError('Invalid native reference')


@dataclass(frozen=True)
class ResolvedVehicle:
    logical_id: int
    generation: int
    detection: IdentityDetection
    speed_mps: float | None = None
    velocity_valid: bool = False
    recovered: bool = False
    association: str = ASSOCIATION_VERSION
    exit_pending: bool = False
    birth_timestamp: float | None = None
    first_native_reference: tuple[int,int] | None = None


@dataclass(frozen=True)
class IdentityResolution:
    vehicles: tuple[ResolvedVehicle, ...] = ()
    labels: tuple[ResolvedVehicle | None, ...] = ()
    unresolved: tuple[int, ...] = ()
    duplicate: bool = False
    capacity_exhausted: bool = False
    ambiguous_logical_ids: tuple[int,...] = ()


@dataclass(frozen=True)
class FrameIdentityEvidence:
    detections: tuple[IdentityDetection, ...]
    camera_stable: bool = True
    camera_reason: str | None = None
