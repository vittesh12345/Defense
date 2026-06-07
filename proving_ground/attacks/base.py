"""Attack interface.

Attacks consume an RGB uint8 image and return a perturbed RGB uint8 image, so
the result can be fed straight back into any ``Detector.predict``. White-box
attacks additionally require the detector to implement ``WhiteBox``.

The canonical ``[0, 1]`` image<->tensor conversions live in ``adapters.base``
(the image-space contract's home) and are re-exported here for convenience.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from proving_ground.adapters.base import (
    Detection,
    Detector,
    image_to_tensor,
    tensor_to_image,
)

__all__ = ["Attack", "rect_for", "image_to_tensor", "tensor_to_image"]


def rect_for(
    size: float, location: str | tuple[float, float], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    """Axis-aligned patch rectangle (x1, y1, x2, y2) in pixels, clamped to image.

    Shared by the patch-style attacks so they place patches identically.
    ``location`` is ``"center"`` or top-left fractions ``(fx, fy)``.
    """
    h, w = image_shape[:2]
    pw = max(1, round(size * w))
    ph = max(1, round(size * h))
    if location == "center":
        x1 = round((w - pw) / 2)
        y1 = round((h - ph) / 2)
    else:
        fx, fy = location
        x1 = round(fx * w)
        y1 = round(fy * h)
    x1 = min(max(0, x1), w - pw)
    y1 = min(max(0, y1), h - ph)
    return x1, y1, x1 + pw, y1 + ph


@runtime_checkable
class Attack(Protocol):
    """Maps a clean image to a perturbed image of the same shape/dtype."""

    name: str

    def apply(
        self, detector: Detector, image: np.ndarray, targets: list[Detection]
    ) -> np.ndarray:
        ...
