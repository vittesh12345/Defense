"""Detection metrics: IoU, per-class AP and mAP at a fixed IoU threshold.

This is the measurement core, so the algorithm is the standard, well-understood
one (Pascal-VOC-style AP) rather than anything clever:

* predictions are matched to ground truth **within each image** (never across
  images), greedily, highest score first;
* one ground-truth box may be matched at most once (later matches are
  false positives);
* TP/FP decisions are then pooled across all images, ranked by score, and AP is
  the area under the precision-recall curve using the all-points (COCO-style)
  interpolation, i.e. precision is made monotonically non-increasing from the
  right before integrating.

Everything is deterministic given the inputs.

The dataset-level functions take *aligned per-image* sequences: element ``i`` of
``image_preds`` and element ``i`` of ``image_gts`` belong to the same image.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from proving_ground.adapters.base import Detection

PerImage = Sequence[Sequence[Detection]]


def iou_xyxy(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float]
) -> float:
    """Intersection-over-union of two xyxy boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _ap_from_pr(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """All-points AP: area under the PR curve after right-max smoothing."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def average_precision(
    image_preds: PerImage,
    image_gts: PerImage,
    iou_threshold: float = 0.5,
) -> float:
    """AP for a single class across a dataset of images.

    ``image_preds`` and ``image_gts`` are aligned per-image sequences, already
    filtered to the class of interest.
    """
    if len(image_preds) != len(image_gts):
        raise ValueError("image_preds and image_gts must be aligned (same length)")

    n_gt = sum(len(g) for g in image_gts)
    n_pred = sum(len(p) for p in image_preds)

    if n_gt == 0:
        # No objects to find: AP is 0 if anything was predicted, else perfect.
        return 0.0 if n_pred else 1.0
    if n_pred == 0:
        return 0.0

    scores: list[float] = []
    is_tp: list[float] = []

    for preds, gts in zip(image_preds, image_gts, strict=True):
        matched = [False] * len(gts)
        for pred in sorted(preds, key=lambda d: d.score, reverse=True):
            best_iou, best_j = 0.0, -1
            for j, gt in enumerate(gts):
                if matched[j]:
                    continue
                iou = iou_xyxy(pred.box_xyxy, gt.box_xyxy)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            scores.append(pred.score)
            if best_j >= 0 and best_iou >= iou_threshold:
                matched[best_j] = True
                is_tp.append(1.0)
            else:
                is_tp.append(0.0)

    order = np.argsort(-np.asarray(scores), kind="stable")
    tp = np.asarray(is_tp)[order]
    fp = 1.0 - tp

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / n_gt
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    return _ap_from_pr(recalls, precisions)


def _split_by_class(
    image_dets: PerImage, class_id: int
) -> list[list[Detection]]:
    return [[d for d in dets if d.class_id == class_id] for dets in image_dets]


def per_class_ap(
    image_preds: PerImage,
    image_gts: PerImage,
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """AP per class. A class is scored if it appears in GT or predictions."""
    result: dict[str, float] = {}
    for cid, name in enumerate(class_names):
        preds_c = _split_by_class(image_preds, cid)
        gts_c = _split_by_class(image_gts, cid)
        if not any(preds_c) and not any(gts_c):
            continue  # class not present at all -> not scored
        result[name] = average_precision(preds_c, gts_c, iou_threshold)
    return result


def mean_average_precision(
    image_preds: PerImage,
    image_gts: PerImage,
    class_names: Sequence[str],
    iou_threshold: float = 0.5,
) -> float:
    """mAP: mean of per-class AP over scored classes."""
    aps = per_class_ap(image_preds, image_gts, class_names, iou_threshold)
    if not aps:
        return 0.0
    return float(np.mean(list(aps.values())))
