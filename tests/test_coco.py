"""Fast-tier tests for the COCO instances-JSON loader (synthetic, no download)."""

from __future__ import annotations

import json

import cv2
import numpy as np

from proving_ground.data.coco import COCO_CLASSES, load_coco


def _setup(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    for fn in ["a.jpg", "b.jpg"]:
        cv2.imwrite(str(img_dir / fn), np.zeros((20, 30, 3), dtype=np.uint8))
    coco = {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 30, "height": 20},
            {"id": 2, "file_name": "b.jpg", "width": 30, "height": 20},
        ],
        # gappy COCO category ids (1, 3, 18) — must map by NAME
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"},
                       {"id": 18, "name": "dog"}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [5, 4, 10, 8]},
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 6, 6]},
            {"image_id": 1, "category_id": 1, "bbox": [2, 2, 3, 3], "iscrowd": 1},  # skip
            {"image_id": 2, "category_id": 18, "bbox": [1, 1, 4, 4]},
        ],
    }
    aj = tmp_path / "inst.json"
    aj.write_text(json.dumps(coco))
    return img_dir, aj


def test_load_coco_maps_classes_and_converts_boxes(tmp_path):
    img_dir, aj = _setup(tmp_path)
    samples, classes = load_coco(img_dir, aj)
    assert classes == COCO_CLASSES
    assert len(samples) == 2

    a = samples[0]
    assert {d.class_name for d in a.ground_truth} == {"person", "car"}  # crowd skipped
    person = next(d for d in a.ground_truth if d.class_name == "person")
    assert person.class_id == COCO_CLASSES.index("person") == 0
    assert person.box_xyxy == (5.0, 4.0, 15.0, 12.0)  # xywh -> xyxy

    dog = samples[1].ground_truth[0]  # gappy id 18 -> mapped by name
    assert dog.class_name == "dog" and dog.class_id == COCO_CLASSES.index("dog")


def test_load_coco_limit(tmp_path):
    img_dir, aj = _setup(tmp_path)
    samples, _ = load_coco(img_dir, aj, limit=1)
    assert len(samples) == 1 and samples[0].image_id == "a.jpg"
