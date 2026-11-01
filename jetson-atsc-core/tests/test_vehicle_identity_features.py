from types import SimpleNamespace

import cv2
import numpy as np

from src.perception.identity_features import IdentityFeatureExtractor


def track(box=(10, 10, 40, 60)):
    return SimpleNamespace(bbox=box, bottom_center=((box[0]+box[2])/2, box[3]),
        class_name='car', track_id=4, epoch=0, stable_id=True)


def test_normalized_cpu_crop_descriptor_and_occluder_guard():
    extractor = IdentityFeatureExtractor()
    frame = np.zeros((100, 160, 3), np.uint8)
    frame[10:60, 10:40] = (10, 40, 240)
    data = extractor.extract(frame, [track()], lambda x, y: (x/10, y/10), lambda x, y: True)
    item = data.detections[0]
    assert item.appearance_reliable and np.linalg.norm(item.appearance) > .99
    assert item.ground == (2.5, 6.)
    data = extractor.extract(frame, [track(), track((15, 5, 60, 70))], None, lambda x, y: True)
    assert not data.detections[0].appearance_reliable


def test_material_camera_shift_invalidates_recovery_evidence():
    rng = np.random.default_rng(14)
    frame = rng.integers(0, 256, (180, 320, 3), dtype=np.uint8)
    extractor = IdentityFeatureExtractor()
    assert extractor.extract(frame, [], None, lambda x, y: True).camera_stable
    moved = cv2.warpAffine(frame, np.float32([[1, 0, 18], [0, 1, 0]]), (320, 180))
    assert not extractor.extract(moved, [], None, lambda x, y: True).camera_stable


def test_a_changing_corner_timestamp_is_not_evidence_of_global_camera_motion():
    extractor = IdentityFeatureExtractor()
    for number in range(20):
        frame = np.full((240, 320, 3), (0, 0, 255), np.uint8)
        cv2.putText(frame, str(number), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        result = extractor.extract(frame, [], None, lambda x, y: True)
        assert result.camera_stable
    assert result.camera_reason == 'background_unobservable'
