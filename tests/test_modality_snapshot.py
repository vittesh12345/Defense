"""Locked snapshot of the cross-modality domain-shift probes over coco_scenes.

Runs each modality (real YOLO) at several severities over the realistic set and
asserts:

  (a) two runs are byte-identical (deterministic under our seeding);
  (b) the per-modality/per-severity mAP grid matches the committed golden baseline
      within tolerance (drift lock);
  (c) every modality degrades the RGB detector at full shift (the domain-shift
      probe shows a real OOD penalty);
  (d) each modality is monotonic-ish: mAP is non-increasing as severity rises.

This measures how far an OPTICAL detector falls on simulated thermal/SAR-like
input — an OOD robustness probe, not a real IR/SAR sensor evaluation.

Opt-in: marked integration + slow. Regenerate the baseline deliberately via this
module's _compute helper (see tests/baselines/_regen-style flow).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline

from proving_ground.attacks.modality import MODES, ModalityShift
from proving_ground.data.loaders import load_dataset
from proving_ground.eval.metrics import mean_average_precision
from proving_ground.seeding import set_seed

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_HERE = Path(__file__).resolve().parent
BASELINE = _HERE / "baselines" / "coco_scenes_modality.json"
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
            atk = ModalityShift(mode, severity=sev, seed=0)
            preds = [yolo.predict(atk.apply(yolo, s.image, s.ground_truth)) for s in samples]
            row.append(mean_average_precision(preds, gts, classes, 0.5))
        modes[mode] = row
    return {"clean_map": clean, "severities": SEVERITIES, "modes": modes}


@pytest.fixture(scope="module")
def results() -> dict:
    return _compute()


def test_modality_reproducible():
    a = _compute()
    b = _compute()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_modality_matches_locked_baseline(results):
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(results, baseline)
    assert not diffs, "modality grid drifted from locked baseline:\n" + "\n".join(diffs)


def test_every_modality_degrades_at_full_shift(results):
    clean = results["clean_map"]
    for mode, row in results["modes"].items():
        assert row[-1] < clean - DEGRADE_MARGIN, f"{mode} did not degrade at high severity"


def test_each_modality_is_monotonic_ish(results):
    for mode, row in results["modes"].items():
        for lo, hi in zip(row[:-1], row[1:], strict=True):
            assert hi <= lo + MONO_TOL, f"{mode} not monotonic-ish: {row}"
