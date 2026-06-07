"""Single entry point for reproducibility.

Every numeric result in this project must be reproducible. `set_seed` is the one
place that pins Python, NumPy and Torch RNGs and flips Torch into deterministic
mode. The CLI calls it at the start of a run; the test suite calls it from
`conftest.py` so every test starts from the same state.
"""

from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = 0


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Pin all RNGs and enable deterministic algorithms.

    Torch is imported lazily so that modules which only need NumPy (e.g. the
    metrics tests) don't pull torch in transitively.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:  # torch is optional for the pure-numpy code paths
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuBLAS needs this set before deterministic algorithms can be enabled.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
