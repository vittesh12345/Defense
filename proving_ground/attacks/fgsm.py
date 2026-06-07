"""Fast Gradient Sign Method (untargeted) for object detectors.

Single step in normalised pixel space::

    x_adv = clip_[0,1]( x + eps * sign( d loss / d x ) )

Untargeted: we move *up* the detection loss to degrade predictions. ``eps`` is
the L-inf budget in ``[0, 1]`` pixel units, so before the final clip the
perturbation satisfies ``||x_adv - x||_inf <= eps`` exactly. The detector must
implement the ``WhiteBox`` protocol; a clear error is raised otherwise.
"""

from __future__ import annotations

import torch

from proving_ground.adapters.base import Detection, Detector, WhiteBox, tensor_to_image


class FGSM:
    name = "fgsm"

    def __init__(self, eps: float = 0.03) -> None:
        if not 0.0 <= eps <= 1.0:
            raise ValueError(f"eps must be in [0, 1] pixel units; got {eps}")
        self.eps = eps

    def apply(
        self, detector: Detector, image, targets: list[Detection]
    ):
        if not isinstance(detector, WhiteBox):
            raise TypeError(
                f"{type(detector).__name__} does not implement WhiteBox; "
                "FGSM requires a white-box (gradient-capable) detector"
            )

        x = detector.to_input_tensor(image).clone()
        x.requires_grad_(True)

        loss = detector.compute_loss(x, targets)
        if x.grad is not None:
            x.grad.zero_()
        loss.backward()
        assert x.grad is not None  # compute_loss must depend on x

        with torch.no_grad():
            x_adv = x + self.eps * x.grad.sign()
            x_adv = x_adv.clamp(0.0, 1.0)

        return tensor_to_image(x_adv)
