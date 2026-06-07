"""The one interface every detector plugs in behind.

Two levels:

* ``Detector`` — the universal contract. Takes an RGB ``uint8`` ``HWC`` image and
  returns canonical ``Detection`` objects. This is everything a black-box attack
  or a degradation needs.
* ``WhiteBox`` — an optional sub-protocol for gradient-based attacks (e.g. FGSM).
  Only adapters that can expose a differentiable loss implement it.

Coordinate convention is fixed once here: boxes are ``xyxy`` in absolute pixels
with a top-left origin, ``x2 >= x1`` and ``y2 >= y1``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch


def image_to_tensor(image: np.ndarray) -> torch.Tensor:
    """RGB uint8 HWC -> ``(1, 3, H, W)`` float32 tensor in ``[0, 1]``.

    This is the canonical image space white-box attacks operate in.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected RGB uint8 HWC image, got {image.shape}/{image.dtype}")
    t = torch.from_numpy(image.astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).unsqueeze(0).contiguous()


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """``(1, 3, H, W)`` float tensor in ``[0, 1]`` -> RGB uint8 HWC."""
    t = tensor.detach().clamp(0.0, 1.0).squeeze(0).permute(1, 2, 0)
    return (t.cpu().numpy() * 255.0).round().astype(np.uint8)


@dataclass(frozen=True)
class Detection:
    """One predicted (or ground-truth) box in canonical form."""

    box_xyxy: tuple[float, float, float, float]
    score: float
    class_id: int
    class_name: str

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.box_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError(f"box must be xyxy with x2>=x1, y2>=y1; got {self.box_xyxy}")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"score must be in [0, 1]; got {self.score}")


@runtime_checkable
class Detector(Protocol):
    """Universal detector contract."""

    @property
    def class_names(self) -> list[str]:
        """Ordered class names; index == class_id."""
        ...

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Run inference on a single RGB uint8 HWC image."""
        ...


@runtime_checkable
class WhiteBox(Protocol):
    """Optional contract for gradient-based (white-box) attacks.

    The attack works in a single, adapter-agnostic image space: a batched float
    tensor of shape ``(1, 3, H, W)`` with values in ``[0, 1]`` at the image's
    native resolution. This makes a perturbation budget like ``||delta||_inf <=
    eps`` well-defined in pixel space regardless of what the model does
    internally (resize, mean/std normalisation, etc.).

    ``compute_loss`` must be differentiable w.r.t. ``image_tensor`` so an attack
    can take ``d loss / d image``; any internal resizing/normalisation must be
    done with differentiable ops.
    """

    def to_input_tensor(self, image: np.ndarray) -> torch.Tensor:
        """RGB uint8 HWC image -> ``(1, 3, H, W)`` float tensor in ``[0, 1]``."""
        ...

    def compute_loss(
        self, image_tensor: torch.Tensor, targets: list[Detection]
    ) -> torch.Tensor:
        """Scalar detection loss, differentiable w.r.t. ``image_tensor``."""
        ...
