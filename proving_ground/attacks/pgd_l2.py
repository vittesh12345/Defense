"""Projected Gradient Descent (untargeted) for object detectors, L2 flavor.

Multi-step ascent with an L2 budget. Each step takes a fixed-size step in the
unit-norm gradient direction and then projects back into the L2 ball of radius
``eps`` around ``x0`` before clipping to ``[0, 1]``::

    for _ in range(steps):
        g <- d loss / d x
        x <- x + step_size * g / ||g||_2
        delta <- x - x0
        if ||delta||_2 > eps:
            delta <- delta * eps / ||delta||_2     # L2 projection
        x <- clip( x0 + delta, 0, 1 )              # pixel space

This is the L2 mirror of ``PGDLinf``. The ``eps`` budget is measured as the L2
norm of the whole perturbation tensor (not per-pixel), so the literature default
``eps=3.0`` corresponds to an average per-pixel perturbation that scales as
``eps / sqrt(3 * H * W)`` -- much smaller than the L_inf ``eps=0.03`` for
typical image sizes.
"""

from __future__ import annotations

import torch

from proving_ground.adapters.base import Detection, Detector, WhiteBox, tensor_to_image


class PGDL2:
    name = "pgd-l2"

    def __init__(
        self,
        eps: float = 3.0,
        steps: int = 10,
        step_size: float = 0.75,
        random_init: bool = False,
        seed: int = 0,
    ) -> None:
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0 (L2 norm of perturbation); got {eps}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1; got {steps}")
        if step_size < 0.0:
            raise ValueError(f"step_size must be >= 0; got {step_size}")
        self.eps = eps
        self.steps = steps
        self.step_size = step_size
        self.random_init = random_init
        self.seed = seed

    def apply(
        self, detector: Detector, image, targets: list[Detection]
    ):
        if not isinstance(detector, WhiteBox):
            raise TypeError(
                f"{type(detector).__name__} does not implement WhiteBox; "
                "PGD-L2 requires a white-box (gradient-capable) detector"
            )

        x0 = detector.to_input_tensor(image).detach()
        if self.random_init and self.eps > 0.0:
            # Uniform sample inside the L2 ball: pick a direction from a Gaussian,
            # normalise to the unit sphere, then scale by a radius uniform in
            # [0, eps]. Seeded for reproducibility.
            gen = torch.Generator(device=x0.device).manual_seed(self.seed)
            noise = torch.empty_like(x0).normal_(generator=gen)
            noise_norm = noise.flatten().norm(p=2) + 1e-12
            radius = torch.empty(1, device=x0.device).uniform_(0.0, self.eps, generator=gen)
            x = (x0 + noise * (radius / noise_norm)).clamp(0.0, 1.0)
        else:
            x = x0.clone()

        for _ in range(self.steps):
            x = x.detach().requires_grad_(True)
            loss = detector.compute_loss(x, targets)
            loss.backward()
            assert x.grad is not None  # compute_loss must depend on x
            with torch.no_grad():
                g_norm = x.grad.flatten().norm(p=2) + 1e-12
                x = x + self.step_size * (x.grad / g_norm)
                # Project onto the L2 ball around x0, then clamp to [0, 1].
                delta = x - x0
                d_norm = delta.flatten().norm(p=2)
                if d_norm > self.eps:
                    delta = delta * (self.eps / d_norm)
                x = (x0 + delta).clamp(0.0, 1.0)

        return tensor_to_image(x)
