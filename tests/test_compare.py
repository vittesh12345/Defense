"""Fast-tier tests for the multi-model robustness scorecard (no weights)."""

from __future__ import annotations

from _fakes import DummyWhiteBox  # noqa: F401  (kept available for future variants)

from proving_ground.adapters.fake import FakeDetector
from proving_ground.benchmark import default_attacks
from proving_ground.compare import comparison_table, run_comparison
from proving_ground.data.loaders import load_dataset


def test_comparison_structure_and_ranking(fixtures_dir):
    samples, classes = load_dataset(fixtures_dir / "images", fixtures_dir / "annotations.json")
    detectors = [(FakeDetector(), "fake-a"), (FakeDetector(), "fake-b")]
    res = run_comparison(detectors, samples, classes, default_attacks(seed=0), seed=0)

    assert res["num_images"] == 2
    assert len(res["models"]) == 2
    for m in res["models"]:
        assert 0.0 <= m["clean_map"] <= 1.0
        assert 0.0 <= m["mean_attacked_map"] <= 1.0
        assert m["weakest_attack"] is not None
        assert len(m["attacks"]) == res["num_attacks"]

    # Ranked most-robust first: retained fraction is non-increasing.
    retained = [m["robustness_retained"] for m in res["models"]]
    assert retained == sorted(retained, reverse=True)
    assert "Rank" in comparison_table(res)


def test_cli_compare_smoke(fixtures_dir, tmp_path):
    import json

    from proving_ground.cli import main

    out = tmp_path / "scorecard.json"
    rc = main([
        "compare",
        "--images", str(fixtures_dir / "images"),
        "--ann", str(fixtures_dir / "annotations.json"),
        "--models", "fake,fake",
        "--seed", "0", "--out", str(out),
    ])
    assert rc == 0
    res = json.loads(out.read_text())
    assert len(res["models"]) == 2
    assert all("robustness_retained" in m for m in res["models"])
