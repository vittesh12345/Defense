"""Locked snapshot of the real clean->attacked measurement.

Runs the FULL real pipeline (real YOLO weights + real white-box FGSM, not the
FakeDetector surrogate) on the committed CC0 fixture, and asserts:

  (a) two runs produce byte-identical reports after dropping meta.timestamp_utc
      (reproducibility);
  (b) the produced metrics match the committed golden baseline within tolerance
      (drift lock -- a silent metric change fails here);
  (c) clean mAP > 0 and attacked mAP < clean mAP (the attack genuinely degrades).

Opt-in: marked integration + slow, excluded from the default `pytest -q`.
Weights download once on first run. Regenerate the baseline deliberately via
tests/baselines/_regen.py (see fixtures/coco_sample/SOURCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline, normalize_report

from proving_ground.cli import main

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASELINE = Path(__file__).resolve().parent / "baselines" / "coco_sample_report.json"
EPS = 0.03
MARGIN = 1e-6  # guard the strict inequality against float-equality flakiness


def _run(fixtures_dir: Path, out: Path) -> dict:
    rc = main([
        "run",
        "--images", str(fixtures_dir / "coco_sample" / "images"),
        "--ann", str(fixtures_dir / "coco_sample" / "annotations.json"),
        "--model", "yolov8n.pt",
        "--eps", str(EPS),
        "--seed", "0",
        "--out", str(out),
    ])
    assert rc == 0
    return json.loads(out.read_text())


def test_real_pipeline_is_byte_identical_across_runs(fixtures_dir, tmp_path):
    a = _run(fixtures_dir, tmp_path / "a.json")
    b = _run(fixtures_dir, tmp_path / "b.json")
    a["meta"].pop("timestamp_utc")
    b["meta"].pop("timestamp_utc")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_real_pipeline_matches_locked_baseline(fixtures_dir, tmp_path):
    produced = normalize_report(_run(fixtures_dir, tmp_path / "r.json"))
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(produced, baseline)
    assert not diffs, "report drifted from locked baseline:\n" + "\n".join(diffs)


def test_attack_genuinely_degrades(fixtures_dir, tmp_path):
    report = _run(fixtures_dir, tmp_path / "r.json")
    clean = report["clean_map"]
    attacked = report["attacks"][0]["attacked_map"]
    assert clean > 0.0
    assert attacked < clean - MARGIN
