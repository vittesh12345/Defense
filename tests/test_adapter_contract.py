"""The shared adapter contract — anything claiming to be a Detector must pass.

Run against FakeDetector here; the real YOLO adapter reuses the same checks in
the (skipped-by-default) integration test.
"""

from __future__ import annotations

import numpy as np
import pytest

from proving_ground.adapters.base import Detection, Detector
from proving_ground.adapters.fake import FakeDetector


@pytest.fixture
def image() -> np.ndarray:
    img = np.full((64, 80, 3), 20, dtype=np.uint8)
    img[10:30, 15:40] = (200, 40, 40)  # a bright red patch
    return img


def assert_detector_contract(detector: Detector, image: np.ndarray) -> None:
    assert isinstance(detector.class_names, list)
    assert all(isinstance(n, str) for n in detector.class_names)

    dets = detector.predict(image)
    assert isinstance(dets, list)
    h, w = image.shape[:2]
    for d in dets:
        assert isinstance(d, Detection)
        x1, y1, x2, y2 = d.box_xyxy
        assert x2 >= x1 and y2 >= y1  # xyxy invariant
        assert 0.0 <= x1 <= w and 0.0 <= x2 <= w
        assert 0.0 <= y1 <= h and 0.0 <= y2 <= h
        assert 0.0 <= d.score <= 1.0
        assert 0 <= d.class_id < len(detector.class_names)
        assert d.class_name == detector.class_names[d.class_id]


def test_fake_detector_satisfies_contract(image):
    assert_detector_contract(FakeDetector(), image)


def test_fake_detector_is_deterministic(image):
    a = FakeDetector().predict(image)
    b = FakeDetector().predict(image)
    assert [d.box_xyxy for d in a] == [d.box_xyxy for d in b]
    assert [d.score for d in a] == [d.score for d in b]


def test_fake_detector_runtime_checkable():
    assert isinstance(FakeDetector(), Detector)


def test_rejects_non_uint8():
    with pytest.raises(ValueError):
        FakeDetector().predict(np.zeros((10, 10, 3), dtype=np.float32))


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        FakeDetector().predict(np.zeros((10, 10), dtype=np.uint8))


def test_detection_rejects_bad_box():
    with pytest.raises(ValueError):
        Detection(box_xyxy=(10, 10, 0, 0), score=0.5, class_id=0, class_name="x")


def test_detection_rejects_bad_score():
    with pytest.raises(ValueError):
        Detection(box_xyxy=(0, 0, 1, 1), score=1.5, class_id=0, class_name="x")
