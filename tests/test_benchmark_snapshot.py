"""Locked snapshot of the aggregate benchmark over the coco_scenes set.

Runs the FULL real pipeline (real YOLO + the full attack suite) over the realistic
fixture set and asserts:

  (a) two runs are byte-identical (reproducible under our seeding);
  (b) results match the committed golden baseline within tolerance (drift lock);
  (c) the clean pooled mAP is believable: 0 < clean < 1 (not the 1.0 of the
      single-image anchor);
  (d) every attack degrades the pooled mAP.

Opt-in: marked integration + slow. Regenerate the baseline deliberately by
re-running `proving_ground.cli bench ... --out tests/baselines/coco_scenes_benchmark.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline

from proving_ground.benchmark import default_attacks, run_benchmark
from proving_ground.data.loaders import load_dataset
from proving_ground.seeding import set_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASELINE = Path(__file__).resolve().parent / "baselines" / "coco_scenes_benchmark.json"
MARGIN = 1e-6


def _run(fixtures_dir: Path) -> dict:
    set_seed(0)
    from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

    detector = UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")
    samples, classes = load_dataset(
        fixtures_dir / "coco_scenes" / "images",
        fixtures_dir / "coco_scenes" / "annotations.json",
    )
    return run_benchmark(detector, samples, classes, default_attacks(seed=0), seed=0)


def test_benchmark_reproducible(fixtures_dir):
    a = _run(fixtures_dir)
    b = _run(fixtures_dir)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_benchmark_matches_locked_baseline(fixtures_dir):
    produced = _run(fixtures_dir)
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(produced, baseline)
    assert not diffs, "benchmark drifted from locked baseline:\n" + "\n".join(diffs)


def test_clean_map_is_believable(fixtures_dir):
    results = _run(fixtures_dir)
    assert 0.0 < results["clean_map"] < 1.0  # realistic, not the single-image 1.0


def test_all_attacks_degrade(fixtures_dir):
    results = _run(fixtures_dir)
    clean = results["clean_map"]
    assert results["attacks"], "no attacks ran"
    for a in results["attacks"]:
        assert a["attacked_map"] < clean - MARGIN, f"{a['name']} did not degrade"
