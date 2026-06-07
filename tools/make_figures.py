"""Render CLEAN vs ATTACKED detection-overlay figures (one per attack).

Writes small PNGs to figures/. Uses real YOLO weights (downloaded on first use),
so it's a manual tool, not part of the test suite.

    .venv/bin/python tools/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

import cv2
import numpy as np

from proving_ground.adapters.yolo import UltralyticsYOLOAdapter
from proving_ground.attacks.degradation import DegradationAttack
from proving_ground.attacks.eot_patch import EOTPatchAttack
from proving_ground.attacks.fgsm import FGSM
from proving_ground.attacks.patch import PatchAttack
from proving_ground.data.loaders import load_dataset
from proving_ground.seeding import set_seed
from proving_ground.viz import draw_detections, side_by_side

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "proving_ground" / "data" / "fixtures" / "coco_scenes"
OUT = ROOT / "figures"
PANEL_W = 300  # downscale each panel to keep PNGs small

CLEAN_C = (0, 200, 0)
ATK_C = (255, 60, 60)

# (attack, label, representative image) — patch/eot need a centre patch over objects.
SPECS = [
    (FGSM(eps=0.03), "fgsm", "nyc_crossing.jpg"),
    (PatchAttack(size=0.4, location="center", steps=20, step_size=0.1),
     "patch", "hanoi_market.jpg"),
    (EOTPatchAttack(size=0.4, location="center", steps=15, step_size=0.1, eot_samples=4,
                    scale_min=0.8, scale_max=1.2, rot_deg=12.0, trans=0.05,
                    brightness=0.1, contrast=0.2, seed=0), "eot-patch", "hanoi_market.jpg"),
    (DegradationAttack(mode="low_light", severity=0.8, seed=0),
     "low_light", "nyc_crossing.jpg"),
]


def _scaled(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = PANEL_W / w
    return cv2.resize(img, (PANEL_W, round(h * s)), interpolation=cv2.INTER_AREA)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    set_seed(0)
    detector = UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")
    samples, _ = load_dataset(FIX / "images", FIX / "annotations.json")
    by_id = {s.image_id: s for s in samples}

    for attack, label, image_id in SPECS:
        set_seed(0)
        s = by_id[image_id]
        clean_dets = detector.predict(s.image)
        attacked_img = attack.apply(detector, s.image, s.ground_truth)
        attacked_dets = detector.predict(attacked_img)

        left = _scaled(draw_detections(s.image, clean_dets, CLEAN_C))
        right = _scaled(draw_detections(attacked_img, attacked_dets, ATK_C))
        fig = side_by_side(
            left, right,
            f"CLEAN  ({len(clean_dets)} det)",
            f"{label.upper()}  ({len(attacked_dets)} det)",
        )
        out_path = OUT / f"{label}_{Path(image_id).stem}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
        print(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
