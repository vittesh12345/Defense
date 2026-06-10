"""Projected Gradient Descent (untargeted) for object detectors, L-inf flavor.

Multi-step ascent with an L-inf budget. Each step takes a fixed-size step in the
gradient sign direction and then projects back into the ``[x0 - eps, x0 + eps]``
ball before clipping to ``[0, 1]``::

    for _ in range(steps):
        x <- x + step_size * sign( d loss / d x )
        x <- clip( x, x0 - eps, x0 + eps )       # L-inf projection
        x <- clip( x, 0, 1 )                     # pixel space

Strictly stronger than single-step FGSM at the same ``eps`` budget. The
default form here is deterministic (no random init) so it doesn't need a seed;
``random_init=True`` picks a uniform start inside the eps-ball using ``seed`` for
reproducibility, matching the canonical PGD-K formulation.
"""

from __future__ import annotations

import torch

from proving_ground.adapters.base import Detection, Detector, WhiteBox, tensor_to_image


class PGDLinf:
    name = "pgd-linf"

    def __init__(
        self,
        eps: float = 0.03,
        steps: int = 10,
        step_size: float = 0.0075,
        random_init: bool = False,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1] pixel units; got {eps}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1; got {steps}")
        if not 0.0 <= step_size <= 1.0:
            raise ValueError(f"step_size must be in [0, 1] pixel units; got {step_size}")
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
                "PGD-Linf requires a white-box (gradient-capable) detector"
            )

        x0 = detector.to_input_tensor(image).detach()
        if self.random_init and self.eps > 0.0:
            gen = torch.Generator(device=x0.device).manual_seed(self.seed)
            noise = torch.empty_like(x0).uniform_(-self.eps, self.eps, generator=gen)
            x = (x0 + noise).clamp(0.0, 1.0)
        else:
            x = x0.clone()

        for _ in range(self.steps):
            x = x.detach().requires_grad_(True)
            loss = detector.compute_loss(x, targets)
            loss.backward()
            assert x.grad is not None  # compute_loss must depend on x
            with torch.no_grad():
                x = x + self.step_size * x.grad.sign()
                # Project onto the L_inf ball around x0, then back into [0, 1].
                x = torch.clamp(x, x0 - self.eps, x0 + self.eps).clamp(0.0, 1.0)

        return tensor_to_image(x)
