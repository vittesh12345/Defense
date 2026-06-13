"""Fast-tier tests for the worst-case ensemble (no weights).

Uses FakeDetector (image-dependent predictions, so attacks genuinely move its
mAP) to assert the ensemble's structural guarantees: the per-image selection is
the minimum over attacks, win counts cover every image, the ensemble is no weaker
than the strongest single attack, and the result is reproducible.
"""

from __future__ import annotations

import pytest
from _fakes import DummyWhiteBox  # noqa: F401  (kept available for future variants)

from proving_ground.adapters.fake import FakeDetector
from proving_ground.benchmark import white_box_attacks
from proving_ground.ensemble import ensemble_table, run_ensemble


@pytest.fixture
def setup(fixtures_dir):
    from proving_ground.data.loaders import load_dataset
    samples, classes = load_dataset(fixtures_dir / "images", fixtures_dir / "annotations.json")
    return samples, classes


def test_structure_and_keys(setup):
    samples, classes = setup
    r = run_ensemble(FakeDetector(), samples, classes, white_box_attacks(seed=0), seed=0)
    assert r["kind"] == "ensemble"
    assert r["num_images"] == len(samples)
    assert set(r["attacks"]) == set(r["single_attack_maps"]) == set(r["win_counts"])
    # every image is assigned to exactly one winning attack
    assert sum(r["win_counts"].values()) == r["num_images"]


def test_ensemble_is_a_lower_bound(setup):
    samples, classes = setup
    r = run_ensemble(FakeDetector(), samples, classes, white_box_attacks(seed=0), seed=0)
    # worst-case-per-image is no weaker than the strongest single attack
    assert r["ensemble_map"] <= r["strongest_single_map"] + 1e-9
    assert r["ensemble_map"] <= r["clean_map"] + 1e-9
    assert r["ensemble_gain_over_single"] >= -1e-9
    # strongest single = the minimum of the per-attack pooled maps
    assert r["strongest_single_map"] == pytest.approx(min(r["single_attack_maps"].values()))


def test_deterministic(setup):
    samples, classes = setup
    a = run_ensemble(FakeDetector(), samples, classes, white_box_attacks(seed=0), seed=0)
    b = run_ensemble(FakeDetector(), samples, classes, white_box_attacks(seed=0), seed=0)
    assert a == b


def test_empty_attacks_rejected(setup):
    samples, classes = setup
    with pytest.raises(ValueError):
        run_ensemble(FakeDetector(), samples, classes, [], seed=0)


def test_table_renders(setup):
    samples, classes = setup
    r = run_ensemble(FakeDetector(), samples, classes, white_box_attacks(seed=0), seed=0)
    t = ensemble_table(r)
    assert "worst-case ensemble" in t and "strongest single attack" in t
