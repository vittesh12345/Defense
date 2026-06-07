"""Physically-realizable adversarial patch via Expectation Over Transformation.

A plain optimized patch fools the detector in clean pixel space but falls apart
once it is printed and re-photographed at a different scale, angle, or lighting.
EOT trains the patch to survive that: at every optimization step we sample a
batch of random transforms (scale, rotation, in-region translation, lighting),
composite the (warped, re-lit) patch onto the *unwarped* scene, and ascend the
detector loss AVERAGED over the batch. The patch is thus optimized to be
adversarial in expectation over how it might physically appear.

Determinism: all transforms are drawn from a ``torch.Generator`` seeded off the
run's global seed, so the same seed reproduces the same sampled transforms and
the same patch (up to the float determinism of the warp ops).

Containment: EOT happens only during optimization. The returned attacked image
places the trained patch at ONE canonical, axis-aligned location by direct pixel
assignment (no warp), so every pixel outside that rectangle round-trips back to
the exact input bytes. Warping is used only for training and for held-out
evaluation renderings via ``render``.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F

from proving_ground.adapters.base import Detection, Detector, WhiteBox, tensor_to_image
from proving_ground.attacks.base import rect_for

GRAY = 0.5


class Transform(NamedTuple):
    scale: float
    angle_deg: float
    tx: float  # normalized translation (units of half-image)
    ty: float
    brightness: float
    contrast: float


class EOTPatchAttack:
    name = "eot-patch"

    def __init__(
        self,
        size: float = 0.4,
        location: str | tuple[float, float] = "center",
        steps: int = 15,
        step_size: float = 0.1,
        eot_samples: int = 4,
        scale_min: float = 0.8,
        scale_max: float = 1.2,
        rot_deg: float = 12.0,
        trans: float = 0.05,
        brightness: float = 0.1,
        contrast: float = 0.2,
        seed: int = 0,
    ) -> None:
        if not 0.0 < size <= 1.0:
            raise ValueError(f"size must be in (0, 1]; got {size}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1; got {steps}")
        if not 0.0 < step_size <= 1.0:
            raise ValueError(f"step_size must be in (0, 1]; got {step_size}")
        if eot_samples < 1:
            raise ValueError(f"eot_samples must be >= 1; got {eot_samples}")
        if not 0.0 < scale_min <= scale_max:
            raise ValueError(f"need 0 < scale_min <= scale_max; got {scale_min}, {scale_max}")
        if rot_deg < 0 or trans < 0 or brightness < 0 or contrast < 0:
            raise ValueError("rot_deg/trans/brightness/contrast must be >= 0")
        if location != "center":
            fx, fy = location
            if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
                raise ValueError(f"location fractions must be in [0, 1]; got {location}")
        self.size = size
        self.location = location
        self.steps = steps
        self.step_size = step_size
        self.eot_samples = eot_samples
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.rot_deg = rot_deg
        self.trans = trans
        self.brightness = brightness
        self.contrast = contrast
        self.seed = seed
        self.loss_history: list[float] = []

    def region(self, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        return rect_for(self.size, self.location, image_shape)

    # --- transform sampling & warping ------------------------------------

    def _sample(self, gen: torch.Generator) -> Transform:
        r = torch.rand(6, generator=gen).tolist()
        return Transform(
            scale=self.scale_min + r[0] * (self.scale_max - self.scale_min),
            angle_deg=(r[1] * 2 - 1) * self.rot_deg,
            tx=(r[2] * 2 - 1) * self.trans,
            ty=(r[3] * 2 - 1) * self.trans,
            brightness=(r[4] * 2 - 1) * self.brightness,
            contrast=1.0 + (r[5] * 2 - 1) * self.contrast,
        )

    def _theta(
        self, tf: Transform, region: tuple[int, int, int, int], h: int, w: int, device
    ) -> torch.Tensor:
        x1, y1, x2, y2 = region
        # Patch centre in normalised [-1, 1] coords.
        pcx = 2 * ((x1 + x2) / 2) / w - 1
        pcy = 2 * ((y1 + y2) / 2) / h - 1
        a = math.radians(tf.angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        inv = 1.0 / tf.scale
        # Inverse (output->input) linear part: (1/s) R(-a).
        l00, l01 = ca * inv, sa * inv
        l10, l11 = -sa * inv, ca * inv
        # Transform about the patch centre, then translate: in = L*out + (pc - L*(pc+t)).
        px, py = pcx + tf.tx, pcy + tf.ty
        t0 = pcx - (l00 * px + l01 * py)
        t1 = pcy - (l10 * px + l11 * py)
        return torch.tensor([[[l00, l01, t0], [l10, l11, t1]]], dtype=torch.float32, device=device)

    def _composite(
        self,
        base: torch.Tensor,
        patch: torch.Tensor,
        region: tuple[int, int, int, int],
        tf: Transform,
    ) -> torch.Tensor:
        x1, y1, x2, y2 = region
        h, w = base.shape[-2:]
        patch_lit = (patch - 0.5) * tf.contrast + 0.5 + tf.brightness

        patch_layer = base.new_zeros((1, 3, h, w))
        patch_layer[..., y1:y2, x1:x2] = patch_lit
        mask_layer = base.new_zeros((1, 1, h, w))
        mask_layer[..., y1:y2, x1:x2] = 1.0

        theta = self._theta(tf, region, h, w, base.device)
        grid = F.affine_grid(theta, (1, 3, h, w), align_corners=False)
        warped_patch = F.grid_sample(patch_layer, grid, align_corners=False, padding_mode="zeros")
        warped_mask = F.grid_sample(mask_layer, grid, align_corners=False, padding_mode="zeros")

        # warped_patch already carries edge coverage, so blend with (1 - mask).
        composite = base * (1.0 - warped_mask) + warped_patch
        return composite.clamp(0.0, 1.0)

    # --- optimization & placement ----------------------------------------

    def optimize(
        self, detector: Detector, image: np.ndarray, targets: list[Detection]
    ) -> torch.Tensor:
        if not isinstance(detector, WhiteBox):
            raise TypeError(
                f"{type(detector).__name__} does not implement WhiteBox; "
                "EOTPatchAttack requires a white-box (gradient-capable) detector"
            )
        base = detector.to_input_tensor(image)
        region = self.region(image.shape)
        x1, y1, x2, y2 = region

        patch = base.new_full((1, 3, y2 - y1, x2 - x1), GRAY)
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        self.loss_history = []

        for _ in range(self.steps):
            patch = patch.detach().requires_grad_(True)
            total = base.new_zeros(())
            for _ in range(self.eot_samples):
                tf = self._sample(gen)
                composite = self._composite(base, patch, region, tf)
                total = total + detector.compute_loss(composite, targets)
            mean_loss = total / self.eot_samples
            (grad,) = torch.autograd.grad(mean_loss, patch)
            with torch.no_grad():
                patch = (patch + self.step_size * grad.sign()).clamp(0.0, 1.0)
            self.loss_history.append(float(mean_loss.detach()))

        return patch.detach()

    def apply(self, detector: Detector, image: np.ndarray, targets: list[Detection]) -> np.ndarray:
        patch = self.optimize(detector, image, targets)
        base = detector.to_input_tensor(image)
        x1, y1, x2, y2 = self.region(image.shape)
        adv = base.clone()
        adv[..., y1:y2, x1:x2] = patch.clamp(0.0, 1.0)  # exact canonical placement
        return tensor_to_image(adv)

    # --- evaluation helpers ----------------------------------------------

    def render(
        self,
        detector: Detector,
        image: np.ndarray,
        patch: torch.Tensor,
        scale: float = 1.0,
        rotation: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 1.0,
    ) -> np.ndarray:
        """Composite a trained patch onto the scene under a given transform."""
        base = detector.to_input_tensor(image)
        region = self.region(image.shape)
        tf = Transform(scale, rotation, 0.0, 0.0, brightness, contrast)
        with torch.no_grad():
            composite = self._composite(base, patch, region, tf)
        return tensor_to_image(composite)

    def eot_mean_loss(
        self,
        detector: Detector,
        image: np.ndarray,
        targets: list[Detection],
        patch: torch.Tensor,
        eval_seed: int = 12345,
        n: int | None = None,
    ) -> float:
        """Mean detector loss over a fixed-seed batch of sampled transforms."""
        base = detector.to_input_tensor(image)
        region = self.region(image.shape)
        gen = torch.Generator(device="cpu").manual_seed(eval_seed)
        k = n or self.eot_samples
        total = 0.0
        with torch.no_grad():
            for _ in range(k):
                tf = self._sample(gen)
                composite = self._composite(base, patch, region, tf)
                total += float(detector.compute_loss(composite, targets))
        return total / k
