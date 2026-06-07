"""Fast-tier tests for the figure-drawing helpers (no weights)."""

from __future__ import annotations

import numpy as np

from proving_ground.adapters.base import Detection
from proving_ground.viz import draw_detections, side_by_side


def _img(h=40, w=60):
    return np.full((h, w, 3), 30, dtype=np.uint8)


def _dets():
    return [Detection(box_xyxy=(5, 5, 25, 30), score=0.9, class_id=0, class_name="person")]


def test_draw_detections_preserves_shape_and_dtype():
    img = _img()
    out = draw_detections(img, _dets())
    assert out.shape == img.shape and out.dtype == np.uint8


def test_draw_detections_does_not_mutate_input():
    img = _img()
    before = img.copy()
    draw_detections(img, _dets())
    assert np.array_equal(img, before)


def test_draw_detections_actually_draws():
    img = _img()
    out = draw_detections(img, _dets())
    assert not np.array_equal(out, img)  # something was drawn


def test_draw_empty_detections_is_unchanged():
    img = _img()
    out = draw_detections(img, [])
    assert np.array_equal(out, img)


def test_side_by_side_dimensions():
    left = _img(40, 60)
    right = _img(40, 60)
    fig = side_by_side(left, right, "A", "B")
    # wider than both panels combined (separator) and taller (title bar).
    assert fig.shape[1] >= left.shape[1] + right.shape[1]
    assert fig.shape[0] > left.shape[0]
    assert fig.dtype == np.uint8
