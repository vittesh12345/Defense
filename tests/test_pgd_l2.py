"""PGD-L2 behaviour, proven against a tiny differentiable dummy model.

The L2 mirror of `test_pgd_linf.py`. Asserts the mathematical guarantees of
PGD-L2: bounded L2 perturbation, increased loss, shape/dtype preservation,
determinism, and that more steps at the same budget never undershoot
single-step FGSM (loss-wise on the toy model).
"""

from __future__ import annotations

import numpy as np
import pytest
from _fakes import BlackBoxOnly, DummyWhiteBox

from proving_ground.adapters.base import Detection
from proving_ground.attacks.base import image_to_tensor
from proving_ground.attacks.fgsm import FGSM
from proving_ground.attacks.pgd_l2 import PGDL2


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(32, 48, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> list[Detection]:
    return [Detection(box_xyxy=(0, 0, 10, 10), score=1.0, class_id=0, class_name="red")]


def test_perturbation_bounded_by_eps(image, targets):
    eps = 2.0
    adv = PGDL2(eps=eps, steps=10, step_size=0.5).apply(DummyWhiteBox(), image, targets)
    clean = image_to_tensor(image)
    perturbed = image_to_tensor(adv)
    l2 = (perturbed - clean).flatten().norm(p=2).item()
    # uint8 round-trip can move every pixel by up to 1/255; allow a worst-case
    # quantisation slack of sqrt(N) * (1/255) on top of the L2 bound.
    n = clean.numel()
    slack = (n ** 0.5) * (1.0 / 255.0) + 1e-6
    assert l2 <= eps + slack


def test_shape_and_dtype_preserved(image, targets):
    adv = PGDL2(eps=2.0, steps=5, step_size=0.4).apply(DummyWhiteBox(), image, targets)
    assert adv.shape == image.shape
    assert adv.dtype == np.uint8


def test_loss_increases(image, targets):
    model = DummyWhiteBox()
    clean_loss = model.compute_loss(image_to_tensor(image), targets).item()
    adv = PGDL2(eps=5.0, steps=10, step_size=1.0).apply(model, image, targets)
    adv_loss = model.compute_loss(image_to_tensor(adv), targets).item()
    assert adv_loss > clean_loss


def test_deterministic_without_random_init(image, targets):
    a = PGDL2(eps=3.0, steps=10, step_size=0.75).apply(DummyWhiteBox(), image, targets)
    b = PGDL2(eps=3.0, steps=10, step_size=0.75).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(a, b)


def test_random_init_reproducible_under_same_seed(image, targets):
    a = PGDL2(eps=3.0, steps=10, step_size=0.75, random_init=True, seed=42).apply(
        DummyWhiteBox(), image, targets
    )
    b = PGDL2(eps=3.0, steps=10, step_size=0.75, random_init=True, seed=42).apply(
        DummyWhiteBox(), image, targets
    )
    assert np.array_equal(a, b)


def test_at_least_as_strong_as_fgsm(image, targets):
    """PGD-L2 at the L2 budget that matches FGSM's L_inf damage must not
    undershoot FGSM on the toy linear loss."""
    model = DummyWhiteBox()
    fgsm_eps = 0.03
    fgsm_adv = FGSM(eps=fgsm_eps).apply(model, image, targets)
    # FGSM at L_inf=eps moves every pixel by exactly eps in [0,1], so its L2
    # perturbation is sqrt(N) * eps. Match that as the L2 budget for PGD.
    n = image_to_tensor(image).numel()
    pgd_eps = (n ** 0.5) * fgsm_eps
    pgd_adv = PGDL2(eps=pgd_eps, steps=10, step_size=pgd_eps / 4).apply(
        model, image, targets
    )
    fgsm_loss = model.compute_loss(image_to_tensor(fgsm_adv), targets).item()
    pgd_loss = model.compute_loss(image_to_tensor(pgd_adv), targets).item()
    assert pgd_loss >= fgsm_loss - 1e-6


def test_zero_eps_is_noop(image, targets):
    adv = PGDL2(eps=0.0, steps=5, step_size=0.0).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(adv, image)


def test_requires_whitebox_detector(image, targets):
    with pytest.raises(TypeError):
        PGDL2(eps=3.0).apply(BlackBoxOnly(), image, targets)


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        PGDL2(eps=-0.1)
    with pytest.raises(ValueError):
        PGDL2(eps=3.0, steps=0)
    with pytest.raises(ValueError):
        PGDL2(eps=3.0, step_size=-0.1)
