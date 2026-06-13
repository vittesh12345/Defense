"""Loader for COCO-format detection annotations (instances_*.json).

This is the path to a *credible* cross-model scorecard: COCO val2017 is
exhaustively annotated, so a higher-recall model is not penalised for detecting
real-but-unlabelled objects (the bias that makes our salient-annotated fixtures
unfair across models). The loader maps each COCO category to the detector's class
index **by name** — robust to COCO's gappy category ids (1..90) — so ground-truth
class ids line up with what an 80-class COCO detector (e.g. YOLOv8) outputs.

COCO images carry mixed third-party licenses, so none are committed here; point
the loader at your own COCO download (see README).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import cv2

from proving_ground.adapters.base import Detection
from proving_ground.data.loaders import Sample

# The 80 COCO classes in the canonical order used by YOLOv8 / ultralytics.
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def load_coco(
    images_dir: str | Path,
    instances_json: str | Path,
    limit: int | None = None,
) -> tuple[list[Sample], list[str]]:
    """Load a COCO instances file. Returns (samples, COCO_CLASSES).

    ``limit`` keeps the first N images (sorted by id) for a reproducible subset.
    """
    images_dir = Path(images_dir)
    with Path(instances_json).open() as f:
        coco = json.load(f)

    # COCO category_id -> our class index, mapped by name (handles gappy ids).
    name_by_cat = {c["id"]: c["name"] for c in coco["categories"]}
    idx_by_cat = {
        cid: COCO_CLASSES.index(name)
        for cid, name in name_by_cat.items()
        if name in COCO_CLASSES
    }

    anns_by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    images = sorted(coco["images"], key=lambda im: im["id"])
    if limit is not None:
        images = images[:limit]

    samples: list[Sample] = []
    for im in images:
        path = images_dir / im["file_name"]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt: list[Detection] = []
        for ann in anns_by_image.get(im["id"], []):
            if ann.get("iscrowd"):
                continue  # crowd regions aren't single-instance GT
            cid = idx_by_cat.get(ann["category_id"])
            if cid is None:
                continue
            x, y, w, h = ann["bbox"]
            gt.append(Detection(
                box_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                score=1.0, class_id=cid, class_name=COCO_CLASSES[cid],
            ))
        samples.append(Sample(image_id=im["file_name"], image=rgb, ground_truth=gt))

    return samples, list(COCO_CLASSES)
