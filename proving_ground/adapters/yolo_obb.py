"""Ultralytics YOLO-OBB (oriented bounding box) behind the Detector contract.

This wraps an aerial-trained OBB model (e.g. ``yolov8n-obb.pt``, trained on the
DOTA aerial dataset) so the rest of the pipeline can run on overhead imagery.
COCO-pretrained detectors are out-of-distribution for nadir aerial views (they
mislabel cars as boats and detect almost nothing); a DOTA model detects aerial
classes (``small vehicle``, ``large vehicle``, ``ship``, ``plane`` ...).

This is a **black-box** adapter: it implements ``predict`` only (no ``WhiteBox``
/ ``compute_loss``), which is all the degradation (DVE) attacks need. The model
emits oriented boxes; we return the axis-aligned enclosing ``xyxy`` rectangle so
they slot into the existing ``Detection`` / mAP machinery unchanged.

Loading weights downloads them on first use, so this adapter is only touched by
the CLI and the opt-in integration tests — never the default suite.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from proving_ground.adapters.base import Detection


def obb_to_detections(
    boxes_xyxy: Sequence[Sequence[float]],
    class_ids: Sequence[float],
    scores: Sequence[float],
    class_names: Sequence[str],
) -> list[Detection]:
    """Convert OBB model output (axis-aligned enclosing boxes) to ``Detection``s.

    Pulled out as a pure function so the conversion is unit-testable without
    loading any weights.
    """
    dets: list[Detection] = []
    for box, cls, score in zip(boxes_xyxy, class_ids, scores, strict=True):
        x1, y1, x2, y2 = (float(v) for v in box)
        cid = int(cls)
        dets.append(
            Detection(
                box_xyxy=(x1, y1, x2, y2),
                score=float(score),
                class_id=cid,
                class_name=class_names[cid],
            )
        )
    return dets


class UltralyticsOBBAdapter:
    def __init__(
        self, weights: str = "yolov8n-obb.pt", conf: float = 0.25, device: str = "cpu"
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.model.model.to(device)
        self.conf = conf
        self.device = device
        names = self.model.names  # {id: name}
        self._class_names = [names[i] for i in range(len(names))]

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def predict(self, image: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            image, conf=self.conf, device=self.device, verbose=False
        )
        obb = results[0].obb
        if obb is None:
            return []
        # obb.xyxy is the axis-aligned enclosing rectangle of each oriented box.
        return obb_to_detections(
            obb.xyxy.tolist(), obb.cls.tolist(), obb.conf.tolist(), self._class_names
        )
