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

__all__ = ["Attack", "image_to_tensor", "tensor_to_image"]


@runtime_checkable
class Attack(Protocol):
    """Maps a clean image to a perturbed image of the same shape/dtype."""

    name: str

    def apply(
        self, detector: Detector, image: np.ndarray, targets: list[Detection]
    ) -> np.ndarray:
        ...
