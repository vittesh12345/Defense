"""Locked snapshot + held-out robustness of the real EOT patch.

Runs the FULL real pipeline (real YOLO weights + real EOT optimization) on the
committed CC0 fixture, and asserts:

  (a) two runs produce byte-identical reports after dropping meta.timestamp_utc
      (the EOT transform sampling + warps are bit-stable under our seeding);
  (b) the produced metrics match the committed golden baseline within tolerance;
  (c) clean mAP > 0 and the canonical-placement attacked mAP < clean mAP;
  (d) ROBUSTNESS GATE: the trained patch, re-rendered at scales/rotations HELD
      OUT from training (range edges + slightly beyond), still drops mAP on every
      rendering -- the patch survives transformation, not just clean pixels.

It also REPORTS (without gating) how the EOT patch compares to a plain patch off
its trained placement -- single-image, so informational only.

Opt-in: marked integration + slow. Regenerate the baseline deliberately via
tests/baselines/_regen.py (see fixtures/coco_sample/SOURCE.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _snapshot_util import diff_against_baseline, normalize_report

from proving_ground.cli import main

pytestmark = [pytest.mark.integration, pytest.mark.slow]

BASELINE = Path(__file__).resolve().parent / "baselines" / "coco_sample_eot_patch_report.json"

# Canonical EOT config — kept in sync with the committed baseline.
CFG = dict(
    size=0.4, location="center", steps=15, step_size=0.1, eot_samples=4,
    scale_min=0.8, scale_max=1.2, rot_deg=12.0, trans=0.05,
    brightness=0.1, contrast=0.2, seed=0,
)
MARGIN = 1e-6

# Scales/rotations held out from the training ranges (edges + slightly beyond).
HELD_OUT = [(1.0, 0.0), (0.75, 0.0), (1.25, 0.0), (1.0, 18.0), (1.0, -18.0),
            (0.75, 18.0), (1.25, -18.0)]


def _run(fixtures_dir: Path, out: Path) -> dict:
    rc = main([
        "run",
        "--images", str(fixtures_dir / "coco_sample" / "images"),
        "--ann", str(fixtures_dir / "coco_sample" / "annotations.json"),
        "--model", "yolov8n.pt", "--attack", "eot-patch",
        "--patch-size", str(CFG["size"]), "--steps", str(CFG["steps"]),
        "--step-size", str(CFG["step_size"]), "--eot-samples", str(CFG["eot_samples"]),
        "--eot-scale-min", str(CFG["scale_min"]), "--eot-scale-max", str(CFG["scale_max"]),
        "--eot-rot-deg", str(CFG["rot_deg"]), "--eot-trans", str(CFG["trans"]),
        "--eot-bright", str(CFG["brightness"]), "--eot-contrast", str(CFG["contrast"]),
        "--seed", "0", "--out", str(out),
    ])
    assert rc == 0
    return json.loads(out.read_text())


def test_real_eot_is_byte_identical_across_runs(fixtures_dir, tmp_path):
    a = _run(fixtures_dir, tmp_path / "a.json")
    b = _run(fixtures_dir, tmp_path / "b.json")
    a["meta"].pop("timestamp_utc")
    b["meta"].pop("timestamp_utc")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_real_eot_matches_locked_baseline(fixtures_dir, tmp_path):
    produced = normalize_report(_run(fixtures_dir, tmp_path / "r.json"))
    baseline = json.loads(BASELINE.read_text())
    diffs = diff_against_baseline(produced, baseline)
    assert not diffs, "EOT report drifted from locked baseline:\n" + "\n".join(diffs)


def test_canonical_placement_degrades(fixtures_dir, tmp_path):
    report = _run(fixtures_dir, tmp_path / "r.json")
    assert report["attacks"][0]["name"] == "eot-patch"
    clean = report["clean_map"]
    attacked = report["attacks"][0]["attacked_map"]
    assert clean > 0.0
    assert attacked < clean - MARGIN


def test_robust_across_held_out_renderings(fixtures_dir):
    """The defining EOT gate: drop persists across transformed renderings."""
    from proving_ground.adapters.yolo import UltralyticsYOLOAdapter
    from proving_ground.attacks.eot_patch import EOTPatchAttack
    from proving_ground.attacks.patch import PatchAttack
    from proving_ground.data.loaders import load_dataset
    from proving_ground.eval.metrics import mean_average_precision
    from proving_ground.seeding import set_seed

    set_seed(0)
    yolo = UltralyticsYOLOAdapter("yolov8n.pt", device="cpu")
    samples, classes = load_dataset(
        fixtures_dir / "coco_sample" / "images",
        fixtures_dir / "coco_sample" / "annotations.json",
    )
    s = samples[0]
    gts = [s.ground_truth]
    clean = mean_average_precision([yolo.predict(s.image)], gts, classes, 0.5)
    assert clean > 0.0

    eot = EOTPatchAttack(**CFG)
    patch = eot.optimize(yolo, s.image, s.ground_truth)

    # GATE: every held-out rendering must drop mAP below clean.
    for scale, rot in HELD_OUT:
        rendered = eot.render(yolo, s.image, patch, scale=scale, rotation=rot)
        m = mean_average_precision([yolo.predict(rendered)], gts, classes, 0.5)
        assert m < clean - MARGIN, f"no drop at scale={scale} rot={rot}: mAP={m}"

    # INFORMATIONAL (not gated): plain patch rendered off its trained placement.
    set_seed(0)
    plain = PatchAttack(size=CFG["size"], location="center", steps=20, step_size=0.1)
    # Reuse the EOT renderer to view the plain patch under a held-out transform.
    plain_img = plain.apply(yolo, s.image, s.ground_truth)
    plain_map = mean_average_precision([yolo.predict(plain_img)], gts, classes, 0.5)
    print(f"[info] plain patch @ canonical: mAP={plain_map:.3f}; "
          f"EOT held-out range observed across {len(HELD_OUT)} renders")
