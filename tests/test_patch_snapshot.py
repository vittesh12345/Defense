"""Locked snapshot of the real patch-attack measurement.

Runs the FULL real pipeline (real YOLO weights + real white-box patch
optimization, not the FakeDetector surrogate) on the committed CC0 fixture, and
asserts:

  (a) two runs produce byte-identical reports after dropping meta.timestamp_utc
      (the multi-step patch optimization is bit-stable under our seeding);
  (b) the produced metrics match the committed golden baseline within tolerance
      (drift lock);
  (c) clean mAP > 0 and attacked mAP < clean mAP (the patch genuinely degrades).

Opt-in: marked integration + slow, excluded from the default `pytest -q`.
Regenerate the baseline deliberately via tests/baselines/_regen.py (see
fixtures/coco_sample/SOURCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline, normalize_report

from proving_ground.cli import main

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASELINE = Path(__file__).resolve().parent / "baselines" / "coco_sample_patch_report.json"
PATCH_SIZE, STEPS, STEP_SIZE = 0.4, 20, 0.1
MARGIN = 1e-6  # guard the strict inequality against float-equality flakiness


def _run(fixtures_dir: Path, out: Path) -> dict:
    rc = main([
        "run",
        "--images", str(fixtures_dir / "coco_sample" / "images"),
        "--ann", str(fixtures_dir / "coco_sample" / "annotations.json"),
        "--model", "yolov8n.pt",
        "--attack", "patch",
        "--patch-size", str(PATCH_SIZE),
        "--steps", str(STEPS),
        "--step-size", str(STEP_SIZE),
        "--seed", "0",
        "--out", str(out),
    ])
    assert rc == 0
    return json.loads(out.read_text())


def test_real_patch_is_byte_identical_across_runs(fixtures_dir, tmp_path):
    a = _run(fixtures_dir, tmp_path / "a.json")
    b = _run(fixtures_dir, tmp_path / "b.json")
    a["meta"].pop("timestamp_utc")
    b["meta"].pop("timestamp_utc")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_real_patch_matches_locked_baseline(fixtures_dir, tmp_path):
    produced = normalize_report(_run(fixtures_dir, tmp_path / "r.json"))
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(produced, baseline)
    assert not diffs, "patch report drifted from locked baseline:\n" + "\n".join(diffs)


def test_patch_genuinely_degrades(fixtures_dir, tmp_path):
    report = _run(fixtures_dir, tmp_path / "r.json")
    clean = report["clean_map"]
    attacked = report["attacks"][0]["attacked_map"]
    assert report["attacks"][0]["name"] == "patch"
    assert clean > 0.0
    assert attacked < clean - MARGIN
