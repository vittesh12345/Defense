"""DegradationAttack (black-box DVE) behaviour — fast tier, no weights."""

from __future__ import annotations

import numpy as np
import pytest
from _fakes import BlackBoxOnly

from proving_ground.attacks.degradation import MODES, DegradationAttack


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)


@pytest.fixture
def smooth_image() -> np.ndarray:
    # Low-frequency image: JPEG at max quality is near-identity here (a pure-noise
    # image is pathological for JPEG even at quality 100).
    h, w = 64, 80
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([xx / w * 255, yy / h * 255, (xx + yy) / (w + h) * 255], axis=-1)
    img[10:30, 15:45] = (200, 40, 40)
    return img.astype(np.uint8)


# Run against a detector that does NOT implement WhiteBox — degradations are
# black-box and must not require it.
DET = BlackBoxOnly()


@pytest.mark.parametrize("mode", MODES)
def test_shape_dtype_and_range(image, mode):
    out = DegradationAttack(mode, severity=0.7).apply(DET, image, [])
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


@pytest.mark.parametrize("mode", MODES)
def test_severity_zero_is_near_identity(smooth_image, mode):
    out = DegradationAttack(mode, severity=0.0).apply(DET, smooth_image, [])
    # JPEG at max quality isn't bit-exact; everything else is identity.
    assert np.mean(np.abs(out.astype(int) - smooth_image.astype(int))) < 2.0


@pytest.mark.parametrize("mode", MODES)
def test_deterministic(image, mode):
    a = DegradationAttack(mode, severity=0.6, seed=0).apply(DET, image, [])
    b = DegradationAttack(mode, severity=0.6, seed=0).apply(DET, image, [])
    assert np.array_equal(a, b)


@pytest.mark.parametrize("mode", MODES)
def test_high_severity_changes_image(image, mode):
    out = DegradationAttack(mode, severity=0.85).apply(DET, image, [])
    assert not np.array_equal(out, image)


def test_stochastic_modes_depend_on_seed(image):
    # noise / low_light / smoke / dust should differ across seeds; deterministic
    # modes need not.
    for mode in ("gaussian_noise", "low_light", "smoke", "dust"):
        a = DegradationAttack(mode, severity=0.6, seed=0).apply(DET, image, [])
        b = DegradationAttack(mode, severity=0.6, seed=1).apply(DET, image, [])
        assert not np.array_equal(a, b)


def test_bad_mode_rejected():
    with pytest.raises(ValueError):
        DegradationAttack("bogus")


@pytest.mark.parametrize("sev", [-0.1, 1.5])
def test_bad_severity_rejected(sev):
    with pytest.raises(ValueError):
        DegradationAttack("fog", severity=sev)


def test_rejects_non_uint8(image):
    with pytest.raises(ValueError):
        DegradationAttack("fog").apply(DET, image.astype(np.float32), [])


def test_cli_degradation_smoke(fixtures_dir, tmp_path):
    import json

    from proving_ground.cli import main

    out = tmp_path / "r.json"
    rc = main([
        "run",
        "--images", str(fixtures_dir / "images"),
        "--ann", str(fixtures_dir / "annotations.json"),
        "--model", "fake", "--attack", "degradation",
        "--mode", "fog", "--severity", "0.6", "--out", str(out),
    ])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["attacks"][0]["name"] == "degradation-fog"
    assert report["attacks"][0]["params"]["severity"] == 0.6
