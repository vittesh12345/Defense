"""Reproducibility rule: same seed => identical numbers, across RNGs."""

from __future__ import annotations

import numpy as np
import torch

from proving_ground.seeding import set_seed


def test_numpy_reproducible():
    set_seed(123)
    a = np.random.rand(5)
    set_seed(123)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_torch_reproducible():
    set_seed(7)
    a = torch.randn(4, 3)
    set_seed(7)
    b = torch.randn(4, 3)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.randn(8)
    set_seed(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)


def test_deterministic_algorithms_enabled():
    set_seed(0)
    assert torch.are_deterministic_algorithms_enabled()
