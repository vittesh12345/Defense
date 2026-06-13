"""Fast-tier tests for the cross-modality domain-shift probes (no weights).

Asserts the transform contract for thermal_ir and sar: shape/dtype preserved,
severity 0 is identity, output is deterministic for a fixed seed, the full-shift
output is (near-)grayscale, sar speckle responds to content + seed, and bad
params are rejected. Black-box: the detector argument is ignored.
"""

from __future__ import annotations

import numpy as np
import pytest

from proving_ground.adapters.fake import FakeDetector
from proving_ground.attacks.modality import MODES, ModalityShift


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(40, 56, 3), dtype=np.uint8)


@pytest.fixture
def det() -> FakeDetector:
    return FakeDetector()


@pytest.mark.parametrize("mode", MODES)
def test_shape_dtype_preserved(mode, image, det):
    out = ModalityShift(mode, severity=0.8).apply(det, image, [])
    assert out.shape == image.shape and out.dtype == np.uint8


@pytest.mark.parametrize("mode", MODES)
def test_severity_zero_is_identity(mode, image, det):
    out = ModalityShift(mode, severity=0.0).apply(det, image, [])
    assert np.array_equal(out, image)


@pytest.mark.parametrize("mode", MODES)
def test_deterministic(mode, image, det):
    a = ModalityShift(mode, severity=0.8, seed=0).apply(det, image, [])
    b = ModalityShift(mode, severity=0.8, seed=0).apply(det, image, [])
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mode", MODES)
def test_full_shift_is_near_grayscale(mode, image, det):
    # at severity 1 both modalities collapse chroma to a single intensity channel
    out = ModalityShift(mode, severity=1.0, seed=0).apply(det, image, [])
    spread = out.astype(np.int32).max(axis=2) - out.astype(np.int32).min(axis=2)
    assert spread.mean() < 2.0  # R == G == B up to uint8 rounding


def test_sar_speckle_depends_on_seed(image, det):
    a = ModalityShift("sar", severity=0.8, seed=0).apply(det, image, [])
    b = ModalityShift("sar", severity=0.8, seed=1).apply(det, image, [])
    assert not np.array_equal(a, b)  # multiplicative speckle is seed-driven


def test_thermal_is_deterministic_without_rng(image, det):
    # thermal_ir has no stochastic component; seeds must not matter
    a = ModalityShift("thermal_ir", severity=0.8, seed=0).apply(det, image, [])
    b = ModalityShift("thermal_ir", severity=0.8, seed=99).apply(det, image, [])
    assert np.array_equal(a, b)


def test_bad_params_rejected():
    with pytest.raises(ValueError):
        ModalityShift("not_a_mode")
    with pytest.raises(ValueError):
        ModalityShift("sar", severity=1.5)
