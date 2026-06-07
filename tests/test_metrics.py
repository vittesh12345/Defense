"""Metric correctness against hand-computed values.

Float comparisons use ``pytest.approx`` per the project's hard rules.
"""

from __future__ import annotations

import pytest

from proving_ground.adapters.base import Detection
from proving_ground.eval.metrics import (
    average_precision,
    iou_xyxy,
    mean_average_precision,
    per_class_ap,
)


def det(box, score=1.0, class_id=0, name="red"):
    return Detection(box_xyxy=box, score=score, class_id=class_id, class_name=name)


# --- IoU ---------------------------------------------------------------------


def test_iou_identical_boxes_is_one():
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_disjoint_boxes_is_zero():
    assert iou_xyxy((0, 0, 10, 10), (100, 100, 110, 110)) == pytest.approx(0.0)


def test_iou_half_overlap():
    # Two 10x10 boxes overlapping in a 5x10 strip.
    # inter = 50, union = 100 + 100 - 50 = 150 -> 1/3.
    assert iou_xyxy((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1.0 / 3.0)


# --- AP ----------------------------------------------------------------------


def test_ap_perfect_single_box():
    preds = [[det((0, 0, 10, 10), score=0.9)]]
    gts = [[det((0, 0, 10, 10))]]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_ap_known_value_two_gt_one_fp():
    # GT: box1, box2. Preds: TP(box1, .9), FP(.8), TP(box2, .7).
    # Hand-computed all-points AP = 5/6.
    preds = [[
        det((0, 0, 10, 10), score=0.9),
        det((50, 50, 60, 60), score=0.8),
        det((20, 20, 30, 30), score=0.7),
    ]]
    gts = [[det((0, 0, 10, 10)), det((20, 20, 30, 30))]]
    assert average_precision(preds, gts) == pytest.approx(5.0 / 6.0)


def test_ap_no_predictions_is_zero():
    gts = [[det((0, 0, 10, 10))]]
    assert average_precision([[]], gts) == pytest.approx(0.0)


def test_ap_no_gt_no_pred_is_perfect():
    assert average_precision([[]], [[]]) == pytest.approx(1.0)


def test_ap_no_gt_with_pred_is_zero():
    assert average_precision([[det((0, 0, 10, 10), score=0.5)]], [[]]) == pytest.approx(0.0)


def test_ap_duplicate_prediction_is_false_positive():
    # Two preds on one GT: first TP, second can't re-match -> FP.
    # recall hits 1 on the TP, so AP is still 1.0, but the second must be FP
    # (verified indirectly: adding the dup must not raise / double-count).
    preds = [[det((0, 0, 10, 10), score=0.9), det((0, 0, 10, 10), score=0.8)]]
    gts = [[det((0, 0, 10, 10))]]
    assert average_precision(preds, gts) == pytest.approx(1.0)


def test_ap_matches_per_image_not_globally():
    # A pred in image1 geometrically matches a GT in image0, but they are in
    # different images, so it must be a false positive -> AP 0, not 1.
    image_preds = [[], [det((0, 0, 10, 10), score=0.9)]]
    image_gts = [[det((0, 0, 10, 10))], []]
    assert average_precision(image_preds, image_gts) == pytest.approx(0.0)


def test_ap_below_iou_threshold_is_false_positive():
    # IoU 1/3 < 0.5 -> not a match.
    preds = [[det((5, 0, 15, 10), score=0.9)]]
    gts = [[det((0, 0, 10, 10))]]
    assert average_precision(preds, gts, iou_threshold=0.5) == pytest.approx(0.0)


# --- per-class AP and mAP ----------------------------------------------------


def test_per_class_only_scores_present_classes():
    classes = ["red", "green", "blue"]
    preds = [[det((0, 0, 10, 10), score=0.9, class_id=0, name="red")]]
    gts = [[det((0, 0, 10, 10), class_id=0, name="red")]]
    aps = per_class_ap(preds, gts, classes)
    assert set(aps) == {"red"}  # green/blue absent -> not scored
    assert aps["red"] == pytest.approx(1.0)


def test_map_is_mean_over_scored_classes():
    classes = ["red", "green", "blue"]
    # red: perfect (AP 1.0). green: predicted where none exist (AP 0.0).
    preds = [[
        det((0, 0, 10, 10), score=0.9, class_id=0, name="red"),
        det((40, 40, 50, 50), score=0.9, class_id=1, name="green"),
    ]]
    gts = [[det((0, 0, 10, 10), class_id=0, name="red")]]
    assert mean_average_precision(preds, gts, classes) == pytest.approx(0.5)
