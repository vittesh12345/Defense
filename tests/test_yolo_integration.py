"""Real-YOLO integration — opt-in only.

Excluded from the default ``pytest -q`` run (see ``addopts`` in pyproject); run
explicitly with ``pytest -m integration``. Loading weights downloads them on
first use, which is why this never runs in the default suite.

Reuses the shared adapter-contract checks against the real model.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def yolo():
    from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

    return UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")


def test_yolo_satisfies_adapter_contract(yolo, fixtures_dir):
    from test_adapter_contract import assert_detector_contract

    img = np.full((96, 128, 3), 30, dtype=np.uint8)
    img[20:70, 30:90] = (200, 60, 60)
    assert_detector_contract(yolo, img)


def test_yolo_fgsm_runs_and_bounds_perturbation(yolo, fixtures_dir):
    from proving_ground.adapters.base import Detection, image_to_tensor
    from proving_ground.attacks.fgsm import FGSM

    img = np.full((96, 128, 3), 30, dtype=np.uint8)
    img[20:70, 30:90] = (200, 60, 60)
    targets = [Detection(box_xyxy=(30, 20, 90, 70), score=1.0, class_id=0, class_name="person")]

    eps = 0.03
    adv = FGSM(eps=eps).apply(yolo, img, targets)
    assert adv.shape == img.shape and adv.dtype == np.uint8

    linf = (image_to_tensor(adv) - image_to_tensor(img)).abs().max().item()
    assert linf <= eps + 1.0 / 255.0 + 1e-6


def test_yolo_cli_end_to_end(fixtures_dir, tmp_path):
    from proving_ground.cli import main

    out = tmp_path / "report.json"
    rc = main([
        "run",
        "--images", str(fixtures_dir / "images"),
        "--ann", str(fixtures_dir / "annotations.json"),
        "--model", "yolov8n.pt", "--eps", "0.03", "--seed", "0",
        "--out", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["meta"]["model"] == "yolov8n.pt"
    assert report["attacks"][0]["name"] == "fgsm"
