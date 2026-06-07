"""EOTPatchAttack behaviour, proven against tiny differentiable dummy models.

No real weights, so fast-tier. Containment of the FINAL placed patch is the
defining invariant (EOT happens during optimization, not in the placement).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _fakes import BlackBoxOnly, DummyWhiteBox, SpatialWhiteBox

from proving_ground.adapters.base import Detection, image_to_tensor
from proving_ground.attacks.eot_patch import EOTPatchAttack


def make(**kw):
    """Small, fast EOT attack for tests."""
    defaults = dict(size=0.3, location="center", steps=4, step_size=0.1, eot_samples=3, seed=0)
    defaults.update(kw)
    return EOTPatchAttack(**defaults)


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(40, 64, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> list[Detection]:
    return [Detection(box_xyxy=(0, 0, 10, 10), score=1.0, class_id=0, class_name="red")]


def _outside_mask(image, region):
    x1, y1, x2, y2 = region
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    return outside


def test_containment_center(image, targets):
    atk = make(location="center")
    adv = atk.apply(DummyWhiteBox(), image, targets)
    outside = _outside_mask(image, atk.region(image.shape))
    assert np.array_equal(adv[outside], image[outside])


def test_containment_offcenter(image, targets):
    atk = make(location=(0.0, 0.0))
    adv = atk.apply(DummyWhiteBox(), image, targets)
    outside = _outside_mask(image, atk.region(image.shape))
    assert np.array_equal(adv[outside], image[outside])


def test_patch_region_actually_changes(image, targets):
    atk = make()
    adv = atk.apply(DummyWhiteBox(), image, targets)
    x1, y1, x2, y2 = atk.region(image.shape)
    assert not np.array_equal(adv[y1:y2, x1:x2], image[y1:y2, x1:x2])


def test_shape_and_dtype_preserved(image, targets):
    adv = make().apply(DummyWhiteBox(), image, targets)
    assert adv.shape == image.shape and adv.dtype == np.uint8


def test_patch_pixels_in_range(image, targets):
    adv = make(step_size=0.2).apply(DummyWhiteBox(), image, targets)
    t = image_to_tensor(adv)
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_same_seed_identical(image, targets):
    a = make(seed=0).apply(SpatialWhiteBox(), image, targets)
    b = make(seed=0).apply(SpatialWhiteBox(), image, targets)
    assert np.array_equal(a, b)


def test_different_seed_differs(image, targets):
    a = make(seed=0).apply(SpatialWhiteBox(), image, targets)
    b = make(seed=1).apply(SpatialWhiteBox(), image, targets)
    assert not np.array_equal(a, b)


def test_eot_objective_increases(image, targets):
    model = SpatialWhiteBox()
    atk = make(steps=8)
    trained = atk.optimize(model, image, targets)

    x1, y1, x2, y2 = atk.region(image.shape)
    gray = torch.full((1, 3, y2 - y1, x2 - x1), 0.5)

    # Same held-out transform batch for a fair comparison.
    gray_loss = atk.eot_mean_loss(model, image, targets, gray, eval_seed=999)
    trained_loss = atk.eot_mean_loss(model, image, targets, trained, eval_seed=999)
    assert trained_loss > gray_loss


def test_loss_history_recorded(image, targets):
    atk = make(steps=5)
    atk.apply(SpatialWhiteBox(), image, targets)
    assert len(atk.loss_history) == 5


def test_requires_whitebox_detector(image, targets):
    with pytest.raises(TypeError):
        make().apply(BlackBoxOnly(), image, targets)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": 0.0}, {"size": 1.5}, {"steps": 0}, {"step_size": 0.0},
        {"eot_samples": 0}, {"scale_min": 0.0}, {"scale_min": 1.5, "scale_max": 1.0},
        {"rot_deg": -1.0}, {"location": (1.2, 0.0)},
    ],
)
def test_bad_params_rejected(kwargs):
    with pytest.raises(ValueError):
        EOTPatchAttack(**kwargs)
