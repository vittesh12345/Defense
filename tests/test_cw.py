"""Carlini & Wagner L2 behaviour, proven against a tiny differentiable model.

Asserts the C&W guarantees on the toy linear loss (range ~[-1/3, 1/3]):
minimal-perturbation success to a confidence margin, monotonic perturbation
growth with the margin, shape/dtype preservation, determinism, the white-box
requirement, and degenerate / bad-param handling.
"""

from __future__ import annotations

import numpy as np
import pytest
from _fakes import BlackBoxOnly, DummyWhiteBox

from proving_ground.adapters.base import Detection
from proving_ground.attacks.base import image_to_tensor
from proving_ground.attacks.cw import CarliniWagnerL2


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(24, 32, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> list[Detection]:
    return [Detection(box_xyxy=(0, 0, 10, 10), score=1.0, class_id=0, class_name="red")]


def _l2(a: np.ndarray, b: np.ndarray) -> float:
    return float((image_to_tensor(a) - image_to_tensor(b)).flatten().norm(p=2))


def test_reaches_confidence_margin(image, targets):
    model = DummyWhiteBox()
    kappa = 0.08  # comfortably below the toy loss's ~0.333 ceiling
    clean_loss = model.compute_loss(image_to_tensor(image), targets).item()
    adv = CarliniWagnerL2(confidence=kappa, max_iter=120, lr=0.06).apply(model, image, targets)
    adv_loss = model.compute_loss(image_to_tensor(adv), targets).item()
    # The attack drives the loss up toward the clean + kappa target.
    assert adv_loss >= clean_loss + 0.5 * kappa


def test_larger_margin_costs_more_perturbation(image, targets):
    model = DummyWhiteBox()
    small = CarliniWagnerL2(confidence=0.04, max_iter=120, lr=0.06).apply(model, image, targets)
    large = CarliniWagnerL2(confidence=0.08, max_iter=120, lr=0.06).apply(model, image, targets)
    assert _l2(large, image) > _l2(small, image)


def test_shape_and_dtype_preserved(image, targets):
    adv = CarliniWagnerL2(confidence=0.1, max_iter=20).apply(DummyWhiteBox(), image, targets)
    assert adv.shape == image.shape and adv.dtype == np.uint8


def test_deterministic(image, targets):
    a = CarliniWagnerL2(confidence=0.1, max_iter=30, lr=0.05).apply(DummyWhiteBox(), image, targets)
    b = CarliniWagnerL2(confidence=0.1, max_iter=30, lr=0.05).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(a, b)


def test_zero_confidence_is_near_noop(image, targets):
    # kappa=0 -> x0 already "succeeds"; binary search drives c down to ~0 -> tiny perturbation.
    adv = CarliniWagnerL2(confidence=0.0, max_iter=40, lr=0.05).apply(
        DummyWhiteBox(), image, targets)
    # within uint8 round-trip slack of the clean image
    n = image_to_tensor(image).numel()
    slack = (n ** 0.5) * (1.0 / 255.0) + 1e-6
    assert _l2(adv, image) <= slack


def test_requires_whitebox_detector(image, targets):
    with pytest.raises(TypeError):
        CarliniWagnerL2(confidence=0.1).apply(BlackBoxOnly(), image, targets)


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        CarliniWagnerL2(confidence=-0.1)
    with pytest.raises(ValueError):
        CarliniWagnerL2(max_iter=0)
    with pytest.raises(ValueError):
        CarliniWagnerL2(lr=0.0)
    with pytest.raises(ValueError):
        CarliniWagnerL2(binary_search_steps=0)
    with pytest.raises(ValueError):
        CarliniWagnerL2(initial_const=0.0)
