"""A deterministic, weight-free detector.

Powers the fast test suite and the CLI smoke test: it never downloads or loads
real weights, yet satisfies the full ``Detector`` contract. Its predictions are
a fixed function of the image content so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from proving_ground.adapters.base import Detection, image_to_tensor


class FakeDetector:
    """Returns one box per image, derived deterministically from the pixels.

    The box is placed at the centroid of the brightest region and sized as a
    fixed fraction of the image; the score is the normalised mean brightness.
    Class is chosen from the dominant colour channel. This gives non-trivial,
    content-dependent, fully deterministic output for pipeline tests.

    It also implements ``WhiteBox`` with a simple differentiable surrogate loss
    so the full FGSM code path can be exercised without loading real weights.
    The surrogate is a stand-in for a real detector's loss, not a faithful model
    of ``predict``; it exists so the orchestration is testable end-to-end.
    """

    def __init__(self, class_names: list[str] | None = None) -> None:
        self._class_names = class_names or ["red", "green", "blue"]
        # Fixed, seed-independent channel weights -> determinate gradient sign.
        self._w = torch.linspace(-1.0, 1.0, steps=3).reshape(1, 3, 1, 1)

    @property
    def class_names(self) -> list[str]:
        return list(self._class_names)

    def predict(self, image: np.ndarray) -> list[Detection]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"expected RGB HWC image, got shape {image.shape}")
        if image.dtype != np.uint8:
            raise ValueError(f"expected uint8 image, got {image.dtype}")

        h, w = image.shape[:2]
        gray = image.mean(axis=2)

        # Centroid of the brightest pixels (top 10% by brightness).
        thresh = np.quantile(gray, 0.9)
        ys, xs = np.where(gray >= thresh)
        cy = float(ys.mean()) if ys.size else h / 2.0
        cx = float(xs.mean()) if xs.size else w / 2.0

        bw, bh = w * 0.25, h * 0.25
        x1 = max(0.0, cx - bw / 2.0)
        y1 = max(0.0, cy - bh / 2.0)
        x2 = min(float(w), cx + bw / 2.0)
        y2 = min(float(h), cy + bh / 2.0)

        score = float(gray.mean() / 255.0)
        class_id = int(np.argmax(image.reshape(-1, 3).mean(axis=0)))

        return [
            Detection(
                box_xyxy=(x1, y1, x2, y2),
                score=score,
                class_id=class_id,
                class_name=self._class_names[class_id],
            )
        ]

    # --- WhiteBox surrogate (for exercising white-box attacks weight-free) ---

    def to_input_tensor(self, image: np.ndarray) -> torch.Tensor:
        return image_to_tensor(image)

    def compute_loss(self, image_tensor: torch.Tensor, targets) -> torch.Tensor:
        return (image_tensor * self._w).mean()
