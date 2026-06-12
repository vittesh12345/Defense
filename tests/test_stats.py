"""Fast-tier tests for the CI summary stats (no weights)."""

from __future__ import annotations

import numpy as np
import pytest

from proving_ground.stats import Z95, summarize_bootstrap, summarize_seed


def test_seed_constant_is_zero_width():
    s = summarize_seed([0.5, 0.5, 0.5])
    assert s["mean"] == 0.5 and s["std"] == 0.0
    assert s["ci_lo"] == 0.5 and s["ci_hi"] == 0.5
    assert s["min"] == 0.5 and s["max"] == 0.5


def test_seed_known_values():
    s = summarize_seed([0.1, 0.2, 0.3])
    assert s["mean"] == pytest.approx(0.2)
    assert s["std"] == pytest.approx(0.1)  # sample std (ddof=1) of [.1,.2,.3]
    assert s["min"] == 0.1 and s["max"] == 0.3
    half = Z95 * 0.1 / np.sqrt(3)
    assert s["ci_lo"] == pytest.approx(0.2 - half)
    assert s["ci_hi"] == pytest.approx(0.2 + half)


def test_seed_single_value_is_degenerate():
    s = summarize_seed([0.7])
    assert s["std"] == 0.0 and s["ci_lo"] == 0.7 and s["ci_hi"] == 0.7


def test_bootstrap_percentiles():
    vals = list(np.linspace(0.0, 1.0, 1001))
    b = summarize_bootstrap(vals)
    assert b["mean"] == pytest.approx(0.5, abs=1e-6)
    assert b["ci_lo"] == pytest.approx(0.025, abs=0.01)
    assert b["ci_hi"] == pytest.approx(0.975, abs=0.01)


def test_ci_bounds_ordered():
    s = summarize_seed([0.2, 0.4, 0.9])
    assert s["min"] <= s["mean"] <= s["max"]
    b = summarize_bootstrap([0.1, 0.2, 0.3, 0.9])
    assert b["ci_lo"] <= b["mean"] <= b["ci_hi"]
