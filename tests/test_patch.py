"""PatchAttack behaviour, proven against a tiny differentiable dummy model.

No real weights, so this is fast-tier. The defining property is CONTAINMENT:
only pixels inside the patch rectangle may change.
"""

from __future__ import annotations

import numpy as np
import pytest
from _fakes import BlackBoxOnly, DummyWhiteBox

from proving_ground.adapters.base import Detection
from proving_ground.attacks.base import image_to_tensor
from proving_ground.attacks.patch import PatchAttack


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(40, 64, 3), dtype=np.uint8)


@pytest.fixture
def targets() -> list[Detection]:
    return [Detection(box_xyxy=(0, 0, 10, 10), score=1.0, class_id=0, class_name="red")]


def test_containment_only_patch_pixels_change(image, targets):
    attack = PatchAttack(size=0.3, location="center", steps=5, step_size=0.1)
    adv = attack.apply(DummyWhiteBox(), image, targets)
    x1, y1, x2, y2 = attack.region(image.shape)

    # Build a boolean mask of the patch region and assert every pixel OUTSIDE it
    # is byte-identical to the input.
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(adv[outside], image[outside])


def test_containment_non_center_location(image, targets):
    attack = PatchAttack(size=0.25, location=(0.0, 0.0), steps=3, step_size=0.1)
    adv = attack.apply(DummyWhiteBox(), image, targets)
    x1, y1, x2, y2 = attack.region(image.shape)
    outside = np.ones(image.shape[:2], dtype=bool)
    outside[y1:y2, x1:x2] = False
    assert np.array_equal(adv[outside], image[outside])


def test_patch_region_actually_changes(image, targets):
    # Containment must not be vacuous: something inside the patch must change.
    attack = PatchAttack(size=0.3, location="center", steps=5, step_size=0.1)
    adv = attack.apply(DummyWhiteBox(), image, targets)
    x1, y1, x2, y2 = attack.region(image.shape)
    assert not np.array_equal(adv[y1:y2, x1:x2], image[y1:y2, x1:x2])


def test_shape_and_dtype_preserved(image, targets):
    adv = PatchAttack(size=0.25, steps=4).apply(DummyWhiteBox(), image, targets)
    assert adv.shape == image.shape
    assert adv.dtype == np.uint8


def test_patch_pixels_in_range(image, targets):
    adv = PatchAttack(size=0.3, steps=5, step_size=0.2).apply(DummyWhiteBox(), image, targets)
    t = image_to_tensor(adv)
    assert float(t.min()) >= 0.0 and float(t.max()) <= 1.0


def test_deterministic(image, targets):
    a = PatchAttack(size=0.3, steps=5, step_size=0.1).apply(DummyWhiteBox(), image, targets)
    b = PatchAttack(size=0.3, steps=5, step_size=0.1).apply(DummyWhiteBox(), image, targets)
    assert np.array_equal(a, b)


def test_loss_increases_over_optimization(image, targets):
    model = DummyWhiteBox()
    attack = PatchAttack(size=0.4, location="center", steps=8, step_size=0.1)
    x1, y1, x2, y2 = attack.region(image.shape)

    # Loss at the optimization's starting point (gray-init patch).
    start = image_to_tensor(image).clone()
    start[..., y1:y2, x1:x2] = 0.5
    start_loss = model.compute_loss(start, targets).item()

    adv = attack.apply(model, image, targets)
    final_loss = model.compute_loss(image_to_tensor(adv), targets).item()

    assert final_loss > start_loss


def test_requires_whitebox_detector(image, targets):
    with pytest.raises(TypeError):
        PatchAttack(size=0.25, steps=3).apply(BlackBoxOnly(), image, targets)


@pytest.mark.parametrize(
    "kwargs",
    [{"size": 0.0}, {"size": 1.5}, {"steps": 0}, {"step_size": 0.0}, {"location": (1.5, 0.0)}],
)
def test_bad_params_rejected(kwargs):
    with pytest.raises(ValueError):
        PatchAttack(**kwargs)
