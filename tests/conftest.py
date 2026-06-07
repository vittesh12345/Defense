"""Shared test setup.

Every test starts from a pinned RNG state so results are reproducible, honoring
the project's hard rule on determinism.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from proving_ground.seeding import set_seed

FIXTURES = Path(__file__).resolve().parent.parent / "proving_ground" / "data" / "fixtures"


@pytest.fixture(autouse=True)
def _seed_everything():
    set_seed(0)


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES
