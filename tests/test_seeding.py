"""Reproducibility rule: same seed => identical numbers, across RNGs."""

from __future__ import annotations

import os

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


def test_blas_pinned_to_one_thread():
    # Multi-threaded BLAS reductions are non-associative and their scheduling
    # varies across processes, which drifts iterative attacks past the snapshot
    # tolerance. The pin is what keeps the locked baselines reproducible.
    set_seed(0)
    assert torch.get_num_threads() == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"


def test_blas_pin_overrides_preexisting_env():
    os.environ["OMP_NUM_THREADS"] = "8"
    os.environ["MKL_NUM_THREADS"] = "8"
    try:
        set_seed(0)
        assert os.environ["OMP_NUM_THREADS"] == "1"
        assert os.environ["MKL_NUM_THREADS"] == "1"
    finally:
        set_seed(0)
