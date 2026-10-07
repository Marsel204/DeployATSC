"""
Tests for vehicle class mapping and non-vehicle filtering in DetectorTracker.
Ensures COCO models (yolo11s.pt) and custom models (Final.pt) correctly map
cars to 'car' (NOT 'bus') and filter out non-vehicles (e.g. traffic light class_9).
"""

import os
import sys
import numpy as np
import pytest

CORE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if CORE_ROOT not in sys.path:
    sys.path.insert(0, CORE_ROOT)

from src.perception.detector_tracker import DetectorTracker, TrackedVehicle


def test_coco_model_class_mapping_and_filtering():
    """Verifies that YOLO11s (COCO) maps class 2 to 'car' (not 'bus') and filters non-vehicles."""
    model_path = os.path.join(CORE_ROOT, "models", "yolo11s.pt")
    if not os.path.exists(model_path):
        pytest.skip("models/yolo11s.pt not found")

    tracker = DetectorTracker(model_path=model_path)

    # Class 2 in COCO is 'car'
    # Class 3 in COCO is 'motorcycle'
    # Class 5 in COCO is 'bus'
    # Class 7 in COCO is 'truck'
    # Class 9 in COCO is 'traffic light' (must be filtered out)
    # Class 0 in COCO is 'person' (must be filtered out)

    mapped_car = tracker.map_class(2)
    assert mapped_car == "car", f"Expected class 2 in COCO to map to 'car', got '{mapped_car}'"

    mapped_mc = tracker.map_class(3)
    assert mapped_mc == "motorcycle", f"Expected class 3 in COCO to map to 'motorcycle', got '{mapped_mc}'"

    mapped_bus = tracker.map_class(5)
    assert mapped_bus == "bus", f"Expected class 5 in COCO to map to 'bus', got '{mapped_bus}'"

    mapped_truck = tracker.map_class(7)
    assert mapped_truck == "truck", f"Expected class 7 in COCO to map to 'truck', got '{mapped_truck}'"

    # Non-vehicles should map to None (or be marked as non-vehicle)
    assert tracker.is_valid_vehicle_class(0) is False, "COCO 'person' (0) should not be a valid vehicle"
    assert tracker.is_valid_vehicle_class(9) is False, "COCO 'traffic light' (9) should not be a valid vehicle"


def test_custom_final_pt_class_mapping():
    """Verifies that Final.pt (custom 4-class) maps 0=motorcycle, 1=car, 2=bus, 3=truck."""
    model_path = os.path.join(CORE_ROOT, "models", "Final.pt")
    if not os.path.exists(model_path):
        pytest.skip("models/Final.pt not found")

    tracker = DetectorTracker(model_path=model_path)

    assert tracker.map_class(0) == "motorcycle"
    assert tracker.map_class(1) == "car"
    assert tracker.map_class(2) == "bus"
    assert tracker.map_class(3) == "truck"
