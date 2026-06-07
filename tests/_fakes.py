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
