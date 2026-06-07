"""Localized adversarial patch (white-box, optimized).

A physical-patch-style attack: pick a rectangular region, fill it with a solid
mid-gray "sticker", then optimize ONLY those pixels over N steps to *maximize*
the detector's loss. Everything outside the rectangle is left untouched.

Optimization is PGD-style sign-gradient ascent in the canonical ``[0, 1]`` image
space, clamped to ``[0, 1]`` each step. No optimizer state and no RNG, so the
result is fully deterministic for a given input.

Containment is the defining property: the gradient is masked to the patch, and
after every step the outside region is re-set to the original. Because the base
image came from ``image / 255``, those outside pixels round-trip back to the
exact input bytes — so only patch pixels can ever change.
"""

from __future__ import annotations

import numpy as np
import torch

from proving_ground.adapters.base import Detection, Detector, WhiteBox, tensor_to_image
from proving_ground.attacks.base import rect_for

GRAY = 0.5  # mid-gray sticker init


class PatchAttack:
    name = "patch"

    def __init__(
        self,
        size: float = 0.25,
        location: str | tuple[float, float] = "center",
        steps: int = 10,
        step_size: float = 0.05,
    ) -> None:
        if not 0.0 < size <= 1.0:
            raise ValueError(f"size must be a fraction in (0, 1]; got {size}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1; got {steps}")
        if not 0.0 < step_size <= 1.0:
            raise ValueError(f"step_size must be in (0, 1]; got {step_size}")
        if location != "center":
            fx, fy = location
            if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                raise ValueError(f"location fractions must be in [0, 1]; got {location}")
        self.size = size
        self.location = location
        self.steps = steps
        self.step_size = step_size

    def region(self, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        """Patch rectangle (x1, y1, x2, y2) in pixels, clamped to the image."""
        return rect_for(self.size, self.location, image_shape)

    def apply(self, detector: Detector, image: np.ndarray, targets: list[Detection]) -> np.ndarray:
        if not isinstance(detector, WhiteBox):
            raise TypeError(
                f"{type(detector).__name__} does not implement WhiteBox; "
                "PatchAttack requires a white-box (gradient-capable) detector"
            )

        base = detector.to_input_tensor(image)
        x1, y1, x2, y2 = self.region(image.shape)

        mask = torch.zeros_like(base)
        mask[..., y1:y2, x1:x2] = 1.0

        adv = base.clone()
        adv[..., y1:y2, x1:x2] = GRAY  # mid-gray sticker init

        for _ in range(self.steps):
            adv = adv.detach().requires_grad_(True)
            loss = detector.compute_loss(adv, targets)
            (grad,) = torch.autograd.grad(loss, adv)
            with torch.no_grad():
                adv = adv + self.step_size * grad.sign() * mask  # ascent on patch only
                adv = adv.clamp(0.0, 1.0)
                adv = base * (1.0 - mask) + adv * mask  # enforce containment exactly

        return tensor_to_image(adv)
