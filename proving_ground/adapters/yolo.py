"""Ultralytics YOLO behind the universal Detector + WhiteBox contract.

``predict`` uses the high-level inference path (boxes returned in the original
image's pixel coordinates). ``compute_loss`` drives the underlying torch module's
training-mode loss so FGSM can take gradients w.r.t. the input image.

Loading weights here will download them on first use, so this adapter is only
touched by the CLI and the opt-in integration test — never the default suite.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from proving_ground.adapters.base import Detection, image_to_tensor


class UltralyticsYOLOAdapter:
    def __init__(
        self, weights: str = "yolov8n.pt", conf: float = 0.25, device: str = "cpu"
    ) -> None:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG

        self.model = YOLO(weights)
        self.model.model.to(device)
        # An inference-loaded model carries a bare-dict ``args`` without the loss
        # gains (box/cls/dfl); the loss criterion needs them as a namespace.
        self.model.model.args = get_cfg(DEFAULT_CFG)
        self.conf = conf
        self.device = device
        names = self.model.names  # {id: name}
        self._class_names = [names[i] for i in range(len(names))]

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def predict(self, image: np.ndarray) -> list[Detection]:
        self.model.model.eval()  # a prior compute_loss may have left it in train mode
        results = self.model.predict(
            image, conf=self.conf, device=self.device, verbose=False
        )
        dets: list[Detection] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cid = int(box.cls[0])
            dets.append(
                Detection(
                    box_xyxy=(x1, y1, x2, y2),
                    score=float(box.conf[0]),
                    class_id=cid,
                    class_name=self._class_names[cid],
                )
            )
        return dets

    # --- WhiteBox ---------------------------------------------------------

    def to_input_tensor(self, image: np.ndarray) -> torch.Tensor:
        return image_to_tensor(image).to(self.device)

    def compute_loss(self, image_tensor: torch.Tensor, targets: list[Detection]) -> torch.Tensor:
        net = self.model.model
        net.train()  # loss path needs training-mode forward (raw feature maps)

        h, w = image_tensor.shape[-2:]
        # YOLO strides require H, W divisible by 32; resize differentiably.
        new_h = max(32, round(h / 32) * 32)
        new_w = max(32, round(w / 32) * 32)
        if (new_h, new_w) != (h, w):
            x = F.interpolate(
                image_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
            )
        else:
            x = image_tensor

        batch = self._build_batch(x, targets, h, w)
        try:
            loss, _ = net.loss(batch)
        finally:
            net.eval()  # restore inference mode for any later predict()
        return loss.sum()

    def _build_batch(self, img: torch.Tensor, targets: list[Detection], h: int, w: int) -> dict:
        # Normalised xywh is scale-invariant, so compute it from native h/w.
        cls, boxes, batch_idx = [], [], []
        for t in targets:
            x1, y1, x2, y2 = t.box_xyxy
            cls.append([float(t.class_id)])
            boxes.append([((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h, (x2 - x1) / w, (y2 - y1) / h])
            batch_idx.append(0.0)

        if boxes:
            cls_t = torch.tensor(cls, dtype=torch.float32, device=img.device)
            box_t = torch.tensor(boxes, dtype=torch.float32, device=img.device)
            idx_t = torch.tensor(batch_idx, dtype=torch.float32, device=img.device)
        else:
            cls_t = torch.zeros((0, 1), dtype=torch.float32, device=img.device)
            box_t = torch.zeros((0, 4), dtype=torch.float32, device=img.device)
            idx_t = torch.zeros((0,), dtype=torch.float32, device=img.device)

        return {"img": img, "cls": cls_t, "bboxes": box_t, "batch_idx": idx_t}
