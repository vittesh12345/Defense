"""Locked snapshot of the benchmark confidence intervals (seed sweep + bootstrap).

Runs the canonical suite over K seeds and B image-bootstraps on coco_scenes and
asserts the CI result matches the committed golden within tolerance, the interval
bounds are ordered, and stochastic vs deterministic attacks behave as expected.

Opt-in: marked integration + slow (re-runs the suite K times). Regenerate the
baseline deliberately with:
    proving-ground bench --images ... --ann ... --model yolov8n.pt \
        --seeds 5 --bootstrap 1000 --out tests/baselines/coco_scenes_benchmark_ci.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline

from proving_ground.data.loaders import load_dataset
from proving_ground.seeding import set_seed
from proving_ground.stats import compute_ci

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_HERE = Path(__file__).resolve().parent
BASELINE = _HERE / "baselines" / "coco_scenes_benchmark_ci.json"
FIX = _HERE.parent / "proving_ground" / "data" / "fixtures" / "coco_scenes"
N_SEEDS, N_BOOT = 5, 1000


def _compute() -> dict:
    set_seed(0)
    from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

    yolo = UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")
    samples, classes = load_dataset(FIX / "images", FIX / "annotations.json")
    return compute_ci(yolo, samples, classes, n_seeds=N_SEEDS, n_bootstrap=N_BOOT, base_seed=0)


@pytest.fixture(scope="module")
def ci() -> dict:
    return _compute()


def test_ci_matches_locked_baseline(ci):
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(ci, baseline)
    assert not diffs, "CI drifted from locked baseline:\n" + "\n".join(diffs)


def test_ci_bounds_are_ordered(ci):
    cb = ci["clean_bootstrap_ci"]
    assert cb["ci_lo"] <= cb["mean"] <= cb["ci_hi"]
    eps = 1e-9  # mean = sum/n can land 1 ULP outside [min, max] for equal values
    for a in ci["attacks"]:
        s, b = a["seed_ci"], a["bootstrap_ci"]
        assert s["ci_lo"] - eps <= s["mean"] <= s["ci_hi"] + eps
        assert s["min"] - eps <= s["mean"] <= s["max"] + eps
        assert b["ci_lo"] <= b["mean"] <= b["ci_hi"]


def test_stochastic_vs_deterministic_seed_width(ci):
    by_name = {a["name"]: a for a in ci["attacks"]}
    # EOT samples random transforms -> non-zero seed-CI width.
    assert by_name["eot-patch"]["seed_ci"]["std"] > 0
    # The plain patch is deterministic -> zero-width seed CI.
    assert by_name["patch"]["seed_ci"]["std"] == 0.0
    # Smoke and dust draw from a seeded RNG -> non-zero seed-CI width.
    assert by_name["degradation-smoke"]["seed_ci"]["std"] > 0
    assert by_name["degradation-dust"]["seed_ci"]["std"] > 0
