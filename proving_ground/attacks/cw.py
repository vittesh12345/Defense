"""Carlini & Wagner L2 attack (untargeted), adapted to a scalar detection loss.

C&W is the gold-standard *minimal-perturbation* white-box attack: rather than
spending a fixed budget like PGD, it searches for the **smallest** L2
perturbation that still breaks the model. Three signatures of the method:

  1. **tanh box-constraint.** Optimise an unconstrained variable ``w`` with
     ``x = ½(tanh(w) + 1)`` so the image stays in ``[0, 1]`` by construction —
     no projection/clipping step that would stall the optimiser at the boundary.
  2. **Adam** on a smooth objective ``‖x − x₀‖₂² + c · f(x)`` that trades
     perturbation size against attack strength.
  3. **Binary search on c.** Find the smallest ``c`` (least perturbation) for
     which the attack still succeeds, keeping the smallest successful adversary
     seen across the search.

Adaptation to object detection. Canonical C&W reads per-class logits to define
its margin ``f``. Our white-box contract exposes only a scalar detection loss
``L`` that *rises* as detection is disrupted (the quantity PGD ascends). So we
take attack *success* to mean raising that loss by a confidence margin ``κ``:

    f(x) = relu( (L(x₀) + κ) − L(x) )           # 0 once the margin is met

Minimising ``‖x − x₀‖₂² + c · f`` thus finds the smallest perturbation that
lifts the loss by at least ``κ``. ``κ`` (``confidence``) is expressed in the
detector's own loss units, so it is detector-specific — a value that bites on
one model's loss scale may be a no-op on another's. ``κ = 0`` is the degenerate
case (``x₀`` already "succeeds"), so a positive ``κ`` is required for a real
perturbation.

Deterministic: the optimiser starts from ``x₀`` (no random restarts), so two
runs are byte-identical under our seeding.
"""

from __future__ import annotations

import torch

from proving_ground.adapters.base import Detection, Detector, WhiteBox, tensor_to_image

_BOX_EPS = 1e-6  # keep x₀ inside (0, 1) so atanh is finite


class CarliniWagnerL2:
    name = "cw-l2"

    def __init__(
        self,
        confidence: float = 0.0,
        max_iter: int = 40,
        lr: float = 0.01,
        binary_search_steps: int = 4,
        initial_const: float = 1.0,
        seed: int = 0,
    ) -> None:
        if confidence < 0.0:
            raise ValueError(f"confidence (loss-increase margin) must be >= 0; got {confidence}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1; got {max_iter}")
        if lr <= 0.0:
            raise ValueError(f"lr must be > 0; got {lr}")
        if binary_search_steps < 1:
            raise ValueError(f"binary_search_steps must be >= 1; got {binary_search_steps}")
        if initial_const <= 0.0:
            raise ValueError(f"initial_const must be > 0; got {initial_const}")
        self.confidence = confidence
        self.max_iter = max_iter
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.initial_const = initial_const
        self.seed = seed

    def apply(self, detector: Detector, image, targets: list[Detection]):
        if not isinstance(detector, WhiteBox):
            raise TypeError(
                f"{type(detector).__name__} does not implement WhiteBox; "
                "C&W-L2 requires a white-box (gradient-capable) detector"
            )

        x0 = detector.to_input_tensor(image).detach()
        # tanh-space init: w0 such that ½(tanh(w0)+1) == clamp(x0).
        x0c = x0.clamp(_BOX_EPS, 1.0 - _BOX_EPS)
        w0 = torch.atanh(2.0 * x0c - 1.0)
        target = detector.compute_loss(x0, targets).detach() + self.confidence

        lo, hi = 0.0, 1e10
        const = self.initial_const
        best_l2 = float("inf")
        best_adv = tensor_to_image(x0)  # fallback: the clean image (no successful adversary)

        for _ in range(self.binary_search_steps):
            w = w0.clone().detach().requires_grad_(True)
            opt = torch.optim.Adam([w], lr=self.lr)
            succeeded = False
            for _ in range(self.max_iter):
                opt.zero_grad()
                x = 0.5 * (torch.tanh(w) + 1.0)
                l2_sq = ((x - x0) ** 2).sum()
                loss_det = detector.compute_loss(x, targets)
                hinge = torch.clamp(target - loss_det, min=0.0)
                (l2_sq + const * hinge).backward()
                opt.step()
                with torch.no_grad():
                    x_eval = 0.5 * (torch.tanh(w) + 1.0)
                    cur_loss = detector.compute_loss(x_eval, targets)
                    if cur_loss >= target:
                        cur_l2 = ((x_eval - x0) ** 2).sum().sqrt().item()
                        succeeded = True
                        if cur_l2 < best_l2:
                            best_l2 = cur_l2
                            best_adv = tensor_to_image(x_eval)

            # Smaller perturbation if it still succeeds; otherwise spend more.
            if succeeded:
                hi = min(hi, const)
                const = (lo + hi) / 2.0
            else:
                lo = max(lo, const)
                const = (lo + hi) / 2.0 if hi < 1e10 else const * 10.0

        return best_adv
