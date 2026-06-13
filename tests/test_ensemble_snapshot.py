"""Locked snapshot of the real worst-case ensemble over the coco_scenes set.

Runs the FULL real pipeline (real YOLO + the white-box attack ensemble) over the
realistic fixture set and asserts:

  (a) two runs are byte-identical (reproducible under our seeding);
  (b) results match the committed golden baseline within tolerance (drift lock);
  (c) the ensemble is a genuine lower bound: ensemble mAP <= the strongest single
      attack's pooled mAP, and well below clean.

Opt-in: marked integration + slow. Regenerate the baseline deliberately:
    proving-ground ensemble --images ... --ann ... --model yolov8n.pt \
        --out tests/baselines/coco_scenes_ensemble.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline

from proving_ground.benchmark import white_box_attacks
from proving_ground.data.loaders import load_dataset
from proving_ground.ensemble import run_ensemble
from proving_ground.seeding import set_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASELINE = Path(__file__).resolve().parent / "baselines" / "coco_scenes_ensemble.json"
MARGIN = 1e-6


def _run(fixtures_dir: Path) -> dict:
    set_seed(0)
    from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

    detector = UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")
    samples, classes = load_dataset(
        fixtures_dir / "coco_scenes" / "images",
        fixtures_dir / "coco_scenes" / "annotations.json",
    )
    return run_ensemble(detector, samples, classes, white_box_attacks(seed=0), seed=0)


def test_ensemble_reproducible(fixtures_dir):
    a = _run(fixtures_dir)
    b = _run(fixtures_dir)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_ensemble_matches_locked_baseline(fixtures_dir):
    produced = _run(fixtures_dir)
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(produced, baseline)
    assert not diffs, "ensemble drifted from locked baseline:\n" + "\n".join(diffs)


def test_ensemble_is_a_genuine_lower_bound(fixtures_dir):
    r = _run(fixtures_dir)
    assert r["ensemble_map"] <= r["strongest_single_map"] + MARGIN
    assert r["ensemble_map"] < r["clean_map"] - MARGIN
    assert sum(r["win_counts"].values()) == r["num_images"]
