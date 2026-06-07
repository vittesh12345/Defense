"""Weight-free test doubles shared across the fast-tier attack tests."""

from __future__ import annotations

import numpy as np
import torch

from proving_ground.adapters.base import image_to_tensor


class BlackBoxOnly:
    """A detector that does NOT implement WhiteBox."""

    class_names = ["red"]

    def predict(self, image):
        return []


class DummyWhiteBox:
    """Differentiable toy detector. Loss = mean of a fixed linear projection.

    Weights are seed-independent so the gradient sign is determinate, which makes
    gradient-ascent attacks (FGSM, patch) behave predictably in tests.
    """

    class_names = ["red", "green", "blue"]

    def __init__(self) -> None:
        self._w = torch.linspace(-1.0, 1.0, steps=3).reshape(1, 3, 1, 1)

    def to_input_tensor(self, image: np.ndarray) -> torch.Tensor:
        return image_to_tensor(image)

    def compute_loss(self, image_tensor: torch.Tensor, targets) -> torch.Tensor:
        return (image_tensor * self._w).mean()

    def predict(self, image):  # present for completeness; not used by attacks
        return []


class SpatialWhiteBox:
    """Differentiable toy detector whose loss depends on WHERE pixels land.

    The loss weights pixels by a 2-D spatial ramp (and per channel), so the
    sampled transform (scale/rotation/translation) changes the gradient. That
    makes EOT behave non-trivially in tests: different sampled transforms -> a
    different trained patch.
    """

    class_names = ["red", "green", "blue"]

    def to_input_tensor(self, image: np.ndarray) -> torch.Tensor:
        return image_to_tensor(image)

    def compute_loss(self, image_tensor: torch.Tensor, targets) -> torch.Tensor:
        h, w = image_tensor.shape[-2:]
        ramp_w = torch.linspace(-1.0, 1.0, w).reshape(1, 1, 1, w)
        ramp_h = torch.linspace(-1.0, 1.0, h).reshape(1, 1, h, 1)
        cw = torch.linspace(-1.0, 1.0, 3).reshape(1, 3, 1, 1)
        return (image_tensor * ramp_w * ramp_h * cw).mean()

    def predict(self, image):
        return []
