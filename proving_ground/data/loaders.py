"""Load images + ground-truth boxes into a canonical in-memory form.

The annotation format is a deliberately tiny COCO-ish JSON so fixtures stay
small and human-readable::

    {
      "classes": ["red", "green", "blue"],
      "images": [
        {"file": "img0.png",
         "boxes": [{"box_xyxy": [x1, y1, x2, y2], "class_id": 0}]}
      ]
    }

Ground-truth boxes become ``Detection`` objects with ``score=1.0`` so the eval
code treats predictions and ground truth uniformly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from proving_ground.adapters.base import Detection


@dataclass(frozen=True)
class Sample:
    """One image plus its ground-truth detections."""

    image_id: str
    image: np.ndarray  # RGB uint8 HWC
    ground_truth: list[Detection]


def load_dataset(
    images_dir: str | Path, annotations_path: str | Path
) -> tuple[list[Sample], list[str]]:
    """Load all annotated images. Returns (samples, class_names)."""
    images_dir = Path(images_dir)
    annotations_path = Path(annotations_path)

    with annotations_path.open() as f:
        ann = json.load(f)

    classes: list[str] = ann["classes"]
    samples: list[Sample] = []

    for entry in ann["images"]:
        img_path = images_dir / entry["file"]
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(f"could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt = [
            Detection(
                box_xyxy=tuple(float(v) for v in b["box_xyxy"]),  # type: ignore[arg-type]
                score=1.0,
                class_id=int(b["class_id"]),
                class_name=classes[int(b["class_id"])],
            )
            for b in entry.get("boxes", [])
        ]
        samples.append(Sample(image_id=entry["file"], image=rgb, ground_truth=gt))

    return samples, classes
