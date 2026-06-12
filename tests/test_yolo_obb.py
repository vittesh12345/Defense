"""Fast-tier tests for the OBB->Detection conversion (no weights)."""

from __future__ import annotations

import pytest

from proving_ground.adapters.base import Detection
from proving_ground.adapters.yolo_obb import obb_to_detections

NAMES = ["plane", "ship", "small vehicle"]


def test_converts_to_axis_aligned_detections():
    dets = obb_to_detections([[10, 20, 40, 60], [0, 0, 5, 5]], [2, 0], [0.9, 0.5], NAMES)
    assert len(dets) == 2
    assert all(isinstance(d, Detection) for d in dets)
    assert dets[0].box_xyxy == (10.0, 20.0, 40.0, 60.0)
    assert dets[0].class_id == 2 and dets[0].class_name == "small vehicle"
    assert dets[1].class_name == "plane"


def test_empty_input():
    assert obb_to_detections([], [], [], NAMES) == []


def test_length_mismatch_raises():
    with pytest.raises(ValueError):  # zip(strict=True)
        obb_to_detections([[0, 0, 1, 1]], [0, 1], [0.5], NAMES)


def test_out_of_range_score_rejected():
    # Detection validates score in [0, 1].
    with pytest.raises(ValueError):
        obb_to_detections([[0, 0, 1, 1]], [0], [1.5], NAMES)
