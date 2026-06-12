"""Real OBB (aerial) adapter — opt-in integration.

Loads the DOTA-trained ``yolov8n-obb`` model (downloads on first use) and checks
the adapter contract: it returns valid axis-aligned ``Detection``s and exposes
the DOTA class space. Runs on an existing committed image — this verifies the
adapter wiring, not detection quality on aerial scenes.
"""

from __future__ import annotations

import cv2
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_obb_adapter_loads_and_satisfies_contract(fixtures_dir):
    from proving_ground.adapters.base import Detection
    from proving_ground.adapters.yolo_obb import UltralyticsOBBAdapter

    adapter = UltralyticsOBBAdapter("yolov8n-obb.pt", device="cpu")
    assert "small vehicle" in adapter.class_names  # DOTA aerial class space

    img_path = fixtures_dir / "coco_sample" / "images" / "sf_cable_car.jpg"
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    dets = adapter.predict(img)

    assert isinstance(dets, list)
    h, w = img.shape[:2]
    for d in dets:
        assert isinstance(d, Detection)
        x1, y1, x2, y2 = d.box_xyxy
        assert x2 >= x1 and y2 >= y1
        assert 0 <= d.class_id < len(adapter.class_names)
        assert d.class_name == adapter.class_names[d.class_id]


def test_cli_routes_obb_model():
    from proving_ground.adapters.yolo_obb import UltralyticsOBBAdapter
    from proving_ground.cli import _build_detector

    detector, label = _build_detector("yolov8n-obb.pt")
    assert isinstance(detector, UltralyticsOBBAdapter)
    assert label == "yolov8n-obb.pt"
