"""Locked snapshot of the DVE degradation family over coco_scenes.

Runs each black-box degradation mode (real YOLO) at several severities over the
realistic set and asserts:

  (a) two runs are byte-identical (deterministic under our seeding);
  (b) the per-mode/per-severity mAP grid matches the committed golden baseline
      within tolerance (drift lock);
  (c) every mode degrades at high severity (attacked mAP < clean);
  (d) each mode is monotonic-ish: mAP is non-increasing as severity rises,
      within tolerance.

Opt-in: marked integration + slow. Regenerate the baseline deliberately (see
CONTRIBUTING.md / this module's compute helper).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline

from proving_ground.attacks.degradation import MODES, DegradationAttack
from proving_ground.data.loaders import load_dataset
from proving_ground.eval.metrics import mean_average_precision
from proving_ground.seeding import set_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_HERE = Path(__file__).resolve().parent
BASELINE = _HERE / "baselines" / "coco_scenes_degradation.json"
FIX = _HERE.parent / "proving_ground" / "data" / "fixtures" / "coco_scenes"
SEVERITIES = [0.25, 0.5, 0.8]
DEGRADE_MARGIN = 1e-6
MONO_TOL = 0.05  # mAP is steppy on a small set; allow small non-monotonic bumps


def _compute() -> dict:
    set_seed(0)
    from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

    yolo = UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")
    samples, classes = load_dataset(FIX / "images", FIX / "annotations.json")
    gts = [s.ground_truth for s in samples]
    clean = mean_average_precision([yolo.predict(s.image) for s in samples], gts, classes, 0.5)

    modes: dict[str, list[float]] = {}
    for mode in MODES:
        row = []
        for sev in SEVERITIES:
            set_seed(0)
            atk = DegradationAttack(mode, severity=sev, seed=0)
            preds = [yolo.predict(atk.apply(yolo, s.image, s.ground_truth)) for s in samples]
            row.append(mean_average_precision(preds, gts, classes, 0.5))
        modes[mode] = row
    return {"clean_map": clean, "severities": SEVERITIES, "modes": modes}


@pytest.fixture(scope="module")
def results() -> dict:
    return _compute()


def test_degradation_reproducible():
    a = _compute()
    b = _compute()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_degradation_matches_locked_baseline(results):
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(results, baseline)
    assert not diffs, "degradation grid drifted from locked baseline:\n" + "\n".join(diffs)


def test_every_mode_degrades_at_high_severity(results):
    clean = results["clean_map"]
    for mode, row in results["modes"].items():
        assert row[-1] < clean - DEGRADE_MARGIN, f"{mode} did not degrade at high severity"


def test_each_mode_is_monotonic_ish(results):
    for mode, row in results["modes"].items():
        for lo, hi in zip(row[:-1], row[1:], strict=True):
            assert hi <= lo + MONO_TOL, f"{mode} not monotonic-ish: {row}"
