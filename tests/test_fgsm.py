"""FGSM behaviour, proven against a tiny differentiable dummy model.

No real weights are loaded, so this runs in the fast suite. The dummy satisfies
the WhiteBox protocol with a trivial differentiable "loss" so we can assert the
mathematical guarantees of FGSM: bounded perturbation, increased loss, shape and
dtype preservation, and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from proving_ground.adapters.base import Detection, WhiteBox
from proving_ground.attacks.base import image_to_tensor
from proving_ground.attacks.fgsm import FGSM


class BlackBoxOnly:
    """A detector that does NOT implement WhiteBox."""

    class_names = ["red"]

    def predict(self, image):
        return []


class DummyWhiteBox:
    """Differentiable toy detector. Loss = mean of a fixed linear projection."""

    class_names = ["red", "green", "blue"]

    def __init__(self) -> None:
        # Fixed (seed-independent) weights so the gradient sign is determinate.
        self._w = torch.linspace(-1.0, 1.0, steps=3).reshape(1, 3, 1, 1)

    def to_input_tensor(self, image: np.ndarray) -> torch.Tensor:
        return image_to_tensor(image)

    def compute_loss(self, image_tensor: torch.Tensor, targets) -> torch.Tensor:
        return (image_tensor * self._w).mean()

    def predict(self, image):  # not used by FGSM, present for completeness
        return []


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> list[Detection]:
    return [Detection(box_xyxy=(0, 0, 10, 10), score=1.0, class_id=0, class_name="red")]


def test_implements_whitebox():
    assert isinstance(DummyWhiteBox(), WhiteBox)


def test_perturbation_bounded_by_eps(image, targets):
    eps = 0.03
    adv = FGSM(eps=eps).apply(DummyWhiteBox(), image, targets)
    clean = image_to_tensor(image)
    perturbed = image_to_tensor(adv)
    linf = (perturbed - clean).abs().max().item()
    # Bound is eps in [0,1] space; allow one uint8 quantisation step (1/255).
    assert linf <= eps + 1.0 / 255.0 + 1e-6


def test_shape_and_dtype_preserved(image, targets):
    adv = FGSM(eps=0.05).apply(DummyWhiteBox(), image, targets)
    assert adv.shape == image.shape
    assert adv.dtype == np.uint8


def test_loss_increases(image, targets):
    model = DummyWhiteBox()
    clean_loss = model.compute_loss(image_to_tensor(image), targets).item()
    adv = FGSM(eps=0.1).apply(model, image, targets)
    adv_loss = model.compute_loss(image_to_tensor(adv), targets).item()
    assert adv_loss > clean_loss


def test_deterministic(image, targets):
    a = FGSM(eps=0.03).apply(DummyWhiteBox(), image, targets)
    b = FGSM(eps=0.03).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(a, b)


def test_zero_eps_is_noop(image, targets):
    adv = FGSM(eps=0.0).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(adv, image)


def test_requires_whitebox_detector(image, targets):
    with pytest.raises(TypeError):
        FGSM(eps=0.03).apply(BlackBoxOnly(), image, targets)


def test_bad_eps_rejected():
    with pytest.raises(ValueError):
        FGSM(eps=1.5)
