"""PGD-Linf behaviour, proven against a tiny differentiable dummy model.

Same shape as the FGSM test (`test_fgsm.py`) but for the multi-step variant.
Asserts the mathematical guarantees of PGD-Linf: bounded perturbation, increased
loss, shape/dtype preservation, determinism, and that more steps at the same
budget never undershoot single-step FGSM.
"""

from __future__ import annotations

import numpy as np
import pytest
from _fakes import BlackBoxOnly, DummyWhiteBox

from proving_ground.adapters.base import Detection
from proving_ground.attacks.base import image_to_tensor
from proving_ground.attacks.fgsm import FGSM
from proving_ground.attacks.pgd import PGDLinf


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> list[Detection]:
    return [Detection(box_xyxy=(0, 0, 10, 10), score=1.0, class_id=0, class_name="red")]


def test_perturbation_bounded_by_eps(image, targets):
    eps = 0.03
    adv = PGDLinf(eps=eps, steps=10, step_size=0.0075).apply(DummyWhiteBox(), image, targets)
    clean = image_to_tensor(image)
    perturbed = image_to_tensor(adv)
    linf = (perturbed - clean).abs().max().item()
    # Bound is eps in [0,1] space; allow one uint8 quantisation step (1/255).
    assert linf <= eps + 1.0 / 255.0 + 1e-6


def test_shape_and_dtype_preserved(image, targets):
    adv = PGDLinf(eps=0.05, steps=5, step_size=0.02).apply(DummyWhiteBox(), image, targets)
    assert adv.shape == image.shape
    assert adv.dtype == np.uint8


def test_loss_increases(image, targets):
    model = DummyWhiteBox()
    clean_loss = model.compute_loss(image_to_tensor(image), targets).item()
    adv = PGDLinf(eps=0.1, steps=10, step_size=0.02).apply(model, image, targets)
    adv_loss = model.compute_loss(image_to_tensor(adv), targets).item()
    assert adv_loss > clean_loss


def test_deterministic_without_random_init(image, targets):
    a = PGDLinf(eps=0.03, steps=10, step_size=0.0075).apply(DummyWhiteBox(), image, targets)
    b = PGDLinf(eps=0.03, steps=10, step_size=0.0075).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(a, b)


def test_random_init_reproducible_under_same_seed(image, targets):
    a = PGDLinf(eps=0.03, steps=10, step_size=0.0075, random_init=True, seed=42).apply(
        DummyWhiteBox(), image, targets
    )
    b = PGDLinf(eps=0.03, steps=10, step_size=0.0075, random_init=True, seed=42).apply(
        DummyWhiteBox(), image, targets
    )
    assert np.array_equal(a, b)


def test_at_least_as_strong_as_fgsm(image, targets):
    """At the same eps budget, multi-step PGD must not undershoot single-step FGSM."""
    model = DummyWhiteBox()
    eps = 0.03
    fgsm_adv = FGSM(eps=eps).apply(model, image, targets)
    pgd_adv = PGDLinf(eps=eps, steps=10, step_size=eps / 4).apply(model, image, targets)
    fgsm_loss = model.compute_loss(image_to_tensor(fgsm_adv), targets).item()
    pgd_loss = model.compute_loss(image_to_tensor(pgd_adv), targets).item()
    # Saturation: with a linear toy loss both attacks hit the boundary; PGD must
    # not be worse than FGSM by more than uint8 round-trip noise.
    assert pgd_loss >= fgsm_loss - 1e-6


def test_zero_eps_is_noop(image, targets):
    adv = PGDLinf(eps=0.0, steps=5, step_size=0.0).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(adv, image)


def test_requires_whitebox_detector(image, targets):
    with pytest.raises(TypeError):
        PGDLinf(eps=0.03).apply(BlackBoxOnly(), image, targets)


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        PGDLinf(eps=1.5)
    with pytest.raises(ValueError):
        PGDLinf(eps=0.03, steps=0)
    with pytest.raises(ValueError):
        PGDLinf(eps=0.03, step_size=2.0)
