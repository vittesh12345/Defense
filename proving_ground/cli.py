"""Orchestrate one robustness run end-to-end.

    proving-ground run \
        --images DIR --ann FILE \
        --model fake --attack fgsm --eps 0.03 \
        --seed 0 --iou 0.5 --out report.json

Flow: seed -> load data -> clean eval (mAP) -> one attack -> attacked eval ->
robustness deltas -> JSON report. The seed is pinned first so every number in
the report is reproducible.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

from proving_ground.adapters.base import Detection, Detector
from proving_ground.adapters.fake import FakeDetector
from proving_ground.attacks.base import Attack
from proving_ground.attacks.cw import CarliniWagnerL2
from proving_ground.attacks.degradation import MODES as DEGRADATION_MODES
from proving_ground.attacks.degradation import DegradationAttack
from proving_ground.attacks.eot_patch import EOTPatchAttack
from proving_ground.attacks.fgsm import FGSM
from proving_ground.attacks.modality import MODES as MODALITY_MODES
from proving_ground.attacks.modality import ModalityShift
from proving_ground.attacks.patch import PatchAttack
from proving_ground.attacks.pgd import PGDLinf
from proving_ground.attacks.pgd_l2 import PGDL2
from proving_ground.data.loaders import Sample, load_dataset
from proving_ground.eval.metrics import mean_average_precision, per_class_ap
from proving_ground.eval.robustness import robustness_delta
from proving_ground.report.generator import build_report, write_report
from proving_ground.report.schema import DatasetInfo
from proving_ground.seeding import set_seed


def _build_detector(model: str) -> tuple[Detector, str]:
    """Return (detector, model_label). 'fake' is weight-free; '*.pt' loads YOLO."""
    if model == "fake":
        return FakeDetector(), "fake"
    if "obb" in model:
        from proving_ground.adapters.yolo_obb import UltralyticsOBBAdapter

        return UltralyticsOBBAdapter(model), model
    if model.endswith(".pt") or model.startswith("yolo"):
        from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

        return UltralyticsYOLOAdapter(model), model
    raise ValueError(f"unknown model: {model!r} (use 'fake' or a YOLO weights path)")


def _predict_all(detector: Detector, samples: list[Sample]) -> list[list[Detection]]:
    return [detector.predict(s.image) for s in samples]


def _load_samples(args: argparse.Namespace):
    """Load our annotation format, or COCO instances JSON when --coco is set."""
    if getattr(args, "coco", False):
        from proving_ground.data.coco import load_coco

        return load_coco(args.images, args.ann, limit=getattr(args, "limit", None))
    return load_dataset(args.images, args.ann)


def _parse_loc(loc: str) -> str | tuple[float, float]:
    if loc == "center":
        return "center"
    try:
        fx, fy = (float(v) for v in loc.split(","))
    except ValueError as e:
        raise ValueError(f"--patch-loc must be 'center' or 'fx,fy'; got {loc!r}") from e
    return (fx, fy)


def _build_attack(args: argparse.Namespace) -> tuple[Attack, dict[str, float]]:
    """Return (attack, numeric params for the report)."""
    if args.attack == "fgsm":
        return FGSM(eps=args.eps), {"eps": args.eps}
    if args.attack == "pgd-linf":
        attack = PGDLinf(
            eps=args.eps,
            steps=args.pgd_steps,
            step_size=args.pgd_step_size,
            random_init=args.pgd_random_init,
            seed=args.seed,
        )
        params = {
            "eps": args.eps,
            "steps": float(args.pgd_steps),
            "step_size": args.pgd_step_size,
            "random_init": float(args.pgd_random_init),
        }
        return attack, params
    if args.attack == "pgd-l2":
        attack = PGDL2(
            eps=args.pgd_l2_eps,
            steps=args.pgd_l2_steps,
            step_size=args.pgd_l2_step_size,
            random_init=args.pgd_l2_random_init,
            seed=args.seed,
        )
        params = {
            "eps": args.pgd_l2_eps,
            "steps": float(args.pgd_l2_steps),
            "step_size": args.pgd_l2_step_size,
            "random_init": float(args.pgd_l2_random_init),
        }
        return attack, params
    if args.attack == "patch":
        loc = _parse_loc(args.patch_loc)
        attack = PatchAttack(
            size=args.patch_size, location=loc, steps=args.steps, step_size=args.step_size
        )
        # Record resolved top-left fractions so the run is fully described.
        if loc == "center":
            loc_x = loc_y = (1.0 - args.patch_size) / 2.0
        else:
            loc_x, loc_y = loc
        params = {
            "patch_size": args.patch_size,
            "loc_x": loc_x,
            "loc_y": loc_y,
            "steps": float(args.steps),
            "step_size": args.step_size,
        }
        return attack, params
    if args.attack == "eot-patch":
        loc = _parse_loc(args.patch_loc)
        attack = EOTPatchAttack(
            size=args.patch_size, location=loc, steps=args.steps, step_size=args.step_size,
            eot_samples=args.eot_samples, scale_min=args.eot_scale_min,
            scale_max=args.eot_scale_max, rot_deg=args.eot_rot_deg, trans=args.eot_trans,
            brightness=args.eot_bright, contrast=args.eot_contrast, seed=args.seed,
        )
        if loc == "center":
            loc_x = loc_y = (1.0 - args.patch_size) / 2.0
        else:
            loc_x, loc_y = loc
        params = {
            "patch_size": args.patch_size, "loc_x": loc_x, "loc_y": loc_y,
            "steps": float(args.steps), "step_size": args.step_size,
            "eot_samples": float(args.eot_samples),
            "scale_min": args.eot_scale_min, "scale_max": args.eot_scale_max,
            "rot_deg": args.eot_rot_deg, "trans": args.eot_trans,
            "brightness": args.eot_bright, "contrast": args.eot_contrast,
        }
        return attack, params
    if args.attack == "cw-l2":
        attack = CarliniWagnerL2(
            confidence=args.cw_confidence, max_iter=args.cw_max_iter, lr=args.cw_lr,
            binary_search_steps=args.cw_bsteps, initial_const=args.cw_initial_const,
            seed=args.seed,
        )
        params = {
            "confidence": args.cw_confidence, "max_iter": float(args.cw_max_iter),
            "lr": args.cw_lr, "binary_search_steps": float(args.cw_bsteps),
            "initial_const": args.cw_initial_const,
        }
        return attack, params
    if args.attack == "degradation":
        attack = DegradationAttack(mode=args.mode, severity=args.severity, seed=args.seed)
        return attack, {"severity": args.severity}
    if args.attack == "modality":
        attack = ModalityShift(mode=args.modality, severity=args.severity, seed=args.seed)
        return attack, {"severity": args.severity}
    raise ValueError(f"unknown attack: {args.attack!r}")


def run(args: argparse.Namespace) -> int:
    set_seed(args.seed)

    samples, classes = load_dataset(args.images, args.ann)
    detector, model_label = _build_detector(args.model)
    gts = [s.ground_truth for s in samples]

    # Clean evaluation.
    clean_preds = _predict_all(detector, samples)
    clean_map = mean_average_precision(clean_preds, gts, classes, args.iou)
    clean_pc = per_class_ap(clean_preds, gts, classes, args.iou)

    # One attack, then re-evaluate on the perturbed images.
    attack, attack_params = _build_attack(args)
    attacked_images = [
        attack.apply(detector, s.image, s.ground_truth) for s in samples
    ]
    attacked_preds = [detector.predict(img) for img in attacked_images]
    attacked_map = mean_average_precision(attacked_preds, gts, classes, args.iou)
    attacked_pc = per_class_ap(attacked_preds, gts, classes, args.iou)

    rob = robustness_delta(clean_map, attacked_map, clean_pc, attacked_pc)

    dataset = DatasetInfo(
        images_dir=str(args.images),
        annotations_path=str(args.ann),
        num_images=len(samples),
        classes=classes,
    )
    report = build_report(
        model=model_label,
        seed=args.seed,
        iou_threshold=args.iou,
        dataset=dataset,
        clean_map=clean_map,
        clean_per_class=clean_pc,
        attack_name=attack.name,
        attack_params=attack_params,
        robustness=rob,
        timestamp_utc=datetime.now(UTC).isoformat(),
    )
    out = write_report(report, args.out)

    print(f"clean mAP@{args.iou}:    {clean_map:.4f}")
    print(f"attacked mAP@{args.iou}: {attacked_map:.4f}  (delta {rob.map_delta:+.4f})")
    print(f"report written: {out}")
    return 0


def bench(args: argparse.Namespace) -> int:
    """Run the canonical attack suite over an image set; emit a results table.

    ``--seeds 1`` (default) is the single-seed point estimate. ``--seeds K`` with
    K > 1 instead emits confidence intervals (seed sweep + image bootstrap).
    """
    import json
    from pathlib import Path

    set_seed(args.seed)
    samples, classes = _load_samples(args)
    detector, _ = _build_detector(args.model)

    if args.seeds > 1:
        from proving_ground.stats import ci_table, compute_ci

        results = compute_ci(
            detector, samples, classes, n_seeds=args.seeds,
            n_bootstrap=args.bootstrap, base_seed=args.seed, iou_threshold=args.iou,
        )
        table = ci_table(results)
    else:
        from proving_ground.benchmark import default_attacks, results_table, run_benchmark

        results = run_benchmark(
            detector, samples, classes, default_attacks(seed=args.seed),
            iou_threshold=args.iou, seed=args.seed,
        )
        table = results_table(results)

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(table)
    if args.out:
        print(f"\nresults written: {args.out}")
    return 0


def ensemble(args: argparse.Namespace) -> int:
    """Worst-case-per-image ensemble: a robustness lower bound over the attacks."""
    import json
    from pathlib import Path

    from proving_ground.benchmark import default_attacks, white_box_attacks
    from proving_ground.ensemble import ensemble_table, run_ensemble

    set_seed(args.seed)
    samples, classes = _load_samples(args)
    detector, _ = _build_detector(args.model)
    attacks = (default_attacks(seed=args.seed) if args.full_suite
               else white_box_attacks(seed=args.seed))
    results = run_ensemble(
        detector, samples, classes, attacks, iou_threshold=args.iou, seed=args.seed,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(ensemble_table(results))
    if args.out:
        print(f"\nensemble results written: {args.out}")
    return 0


def compare(args: argparse.Namespace) -> int:
    """Run the canonical attack suite across several models; emit a scorecard."""
    import json
    from pathlib import Path

    from proving_ground.benchmark import default_attacks
    from proving_ground.compare import comparison_table, run_comparison

    set_seed(args.seed)
    samples, classes = _load_samples(args)
    labels = [m.strip() for m in args.models.split(",") if m.strip()]
    detectors = [(_build_detector(m)[0], m) for m in labels]
    results = run_comparison(
        detectors, samples, classes, default_attacks(seed=args.seed),
        iou_threshold=args.iou, seed=args.seed,
    )
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(comparison_table(results))
    if args.out:
        print(f"\nscorecard written: {args.out}")
    return 0


def report(args: argparse.Namespace) -> int:
    """Render a bench / compare results JSON into a self-contained HTML report."""
    import json
    from pathlib import Path

    from proving_ground.report.html import render_html

    results = json.loads(Path(args.input).read_text())
    Path(args.out).write_text(render_html(results, title=args.title))
    print(f"report written: {args.out}")
    return 0


def tevv(args: argparse.Namespace) -> int:
    """Judge a bench result against acceptance criteria; emit a TEVV assurance case."""
    import json
    from pathlib import Path

    from proving_ground.report.tevv import assess, render_tevv_html

    results = json.loads(Path(args.input).read_text())
    assessment = assess(results, min_clean_map=args.min_clean, min_retained=args.min_retained)
    Path(args.out).write_text(render_tevv_html(assessment, model=args.model, title=args.title))
    print(f"VERDICT: {assessment['verdict']}  "
          f"({assessment['n_pass']}/{len(assessment['conditions'])} conditions pass) — "
          f"{assessment['rationale']}")
    print(f"TEVV report written: {args.out}")
    return 0


def monitor(args: argparse.Namespace) -> int:
    """Diff a current bench result against a baseline; gate on regression."""
    import json
    from pathlib import Path

    from proving_ground.report.monitor import diff_results, render_monitor_html

    baseline = json.loads(Path(args.baseline).read_text())
    current = json.loads(Path(args.current).read_text())
    diff = diff_results(baseline, current, abs_floor=args.abs_floor, rel_floor=args.rel_floor)

    if args.out:
        Path(args.out).write_text(render_monitor_html(diff, model=args.model, title=args.title))
        print(f"monitor report written: {args.out}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(diff, indent=2, sort_keys=True))
        print(f"monitor diff JSON written: {args.out_json}")

    print(f"VERDICT: {diff['verdict']}  "
          f"({diff['n_regressed']}/{diff['n_conditions']} conditions regressed) — "
          f"{diff['rationale']}")

    if diff["verdict"] == "REGRESSED" and not args.no_gate:
        return 1
    return 0


def video(args: argparse.Namespace) -> int:
    """Run a degradation attack over sampled video frames (GT-free stability)."""
    import json
    from pathlib import Path

    from proving_ground.attacks.degradation import DegradationAttack
    from proving_ground.video import run_video

    set_seed(args.seed)
    detector, _ = _build_detector(args.model)
    attack = DegradationAttack(mode=args.mode, severity=args.severity, seed=args.seed)
    res = run_video(detector, attack, args.video, n_frames=args.frames)
    retained = res["detection_retained"]
    retained_str = f"{retained * 100:.0f}%" if retained is not None else "n/a"
    print(f"{res['n_frames']} frames | attack {res['attack']} | "
          f"clean {res['clean_detections']} det -> attacked {res['attacked_detections']} det "
          f"(detections retained {retained_str})")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"results written: {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proving-ground")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one robustness evaluation")
    r.add_argument("--images", required=True, help="directory of images")
    r.add_argument("--ann", required=True, help="annotations JSON")
    r.add_argument("--model", default="fake", help="'fake' or a YOLO weights path/name")
    r.add_argument("--attack", default="fgsm",
                   choices=["fgsm", "pgd-linf", "pgd-l2", "cw-l2", "patch", "eot-patch",
                            "degradation", "modality"],
                   help="attack to run")
    r.add_argument("--mode", default="gaussian_blur", choices=list(DEGRADATION_MODES),
                   help="degradation: which degradation mode")
    r.add_argument("--modality", default="thermal_ir", choices=list(MODALITY_MODES),
                   help="modality: simulated sensor domain shift (uses --severity)")
    r.add_argument("--severity", type=float, default=0.5,
                   help="degradation: severity in [0,1] (0=identity)")
    r.add_argument("--eps", type=float, default=0.03,
                   help="FGSM / PGD-Linf L-inf budget in [0,1]")
    r.add_argument("--pgd-steps", type=int, default=10,
                   help="pgd-linf: number of gradient-sign steps")
    r.add_argument("--pgd-step-size", type=float, default=0.0075,
                   help="pgd-linf: per-step size in [0,1]")
    r.add_argument("--pgd-random-init", action="store_true",
                   help="pgd-linf: start from a uniform point inside the eps-ball")
    r.add_argument("--pgd-l2-eps", type=float, default=3.0,
                   help="pgd-l2: L2 budget for the whole perturbation tensor")
    r.add_argument("--pgd-l2-steps", type=int, default=10,
                   help="pgd-l2: number of unit-norm gradient steps")
    r.add_argument("--pgd-l2-step-size", type=float, default=0.75,
                   help="pgd-l2: per-step L2 size (literature rule: 2.5 * eps / steps)")
    r.add_argument("--pgd-l2-random-init", action="store_true",
                   help="pgd-l2: start from a uniform point inside the L2 eps-ball")
    r.add_argument("--cw-confidence", type=float, default=0.0,
                   help="cw-l2: required loss-increase margin kappa (in the model's loss "
                        "units; detector-specific). 0 = degenerate no-op")
    r.add_argument("--cw-max-iter", type=int, default=40,
                   help="cw-l2: Adam iterations per binary-search step")
    r.add_argument("--cw-lr", type=float, default=0.01, help="cw-l2: Adam learning rate")
    r.add_argument("--cw-bsteps", type=int, default=4,
                   help="cw-l2: binary-search steps over the c trade-off constant")
    r.add_argument("--cw-initial-const", type=float, default=1.0,
                   help="cw-l2: initial c trade-off constant")
    r.add_argument("--patch-size", type=float, default=0.4,
                   help="patch attack: patch side as a fraction of each image dim")
    r.add_argument("--patch-loc", default="center",
                   help="patch attack: 'center' or top-left fractions 'fx,fy'")
    r.add_argument("--steps", type=int, default=20, help="patch attack: optimization steps")
    r.add_argument("--step-size", type=float, default=0.1,
                   help="patch attack: per-step size in [0,1]")
    r.add_argument("--eot-samples", type=int, default=4,
                   help="eot-patch: random transforms averaged per step")
    r.add_argument("--eot-scale-min", type=float, default=0.8, help="eot-patch: min patch scale")
    r.add_argument("--eot-scale-max", type=float, default=1.2, help="eot-patch: max patch scale")
    r.add_argument("--eot-rot-deg", type=float, default=12.0,
                   help="eot-patch: rotation range +/- degrees")
    r.add_argument("--eot-trans", type=float, default=0.05,
                   help="eot-patch: translation jitter (normalized half-image units)")
    r.add_argument("--eot-bright", type=float, default=0.1,
                   help="eot-patch: brightness jitter range +/-")
    r.add_argument("--eot-contrast", type=float, default=0.2,
                   help="eot-patch: contrast jitter range +/-")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP")
    r.add_argument("--out", default="report.json", help="output JSON path")
    r.set_defaults(func=run)

    b = sub.add_parser("bench", help="run the canonical attack suite over an image set")
    b.add_argument("--images", required=True, help="directory of images")
    b.add_argument("--ann", required=True, help="annotations JSON")
    b.add_argument("--model", default="fake", help="'fake' or a YOLO weights path/name")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--seeds", type=int, default=1,
                   help="number of seeds; >1 emits confidence intervals (seed sweep + bootstrap)")
    b.add_argument("--bootstrap", type=int, default=1000,
                   help="image-bootstrap resamples (CI mode only)")
    b.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP")
    b.add_argument("--out", default=None, help="optional results JSON path")
    b.add_argument("--coco", action="store_true",
                   help="treat --images/--ann as a COCO val dir + instances JSON")
    b.add_argument("--limit", type=int, default=None, help="COCO: keep first N images")
    b.set_defaults(func=bench)

    e = sub.add_parser("ensemble",
                       help="worst-case-per-image ensemble: a robustness lower bound")
    e.add_argument("--images", required=True, help="directory of images")
    e.add_argument("--ann", required=True, help="annotations JSON")
    e.add_argument("--model", default="fake", help="'fake' or a YOLO weights path/name")
    e.add_argument("--full-suite", dest="full_suite", action="store_true",
                   help="ensemble over the full suite (white-box + DVE) instead of "
                        "white-box only")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP")
    e.add_argument("--out", default=None, help="optional results JSON path")
    e.add_argument("--coco", action="store_true",
                   help="treat --images/--ann as a COCO val dir + instances JSON")
    e.add_argument("--limit", type=int, default=None, help="COCO: keep first N images")
    e.set_defaults(func=ensemble)

    c = sub.add_parser("compare", help="rank several models by robustness under the attack suite")
    c.add_argument("--images", required=True, help="directory of images")
    c.add_argument("--ann", required=True, help="annotations JSON")
    c.add_argument("--models", required=True,
                   help="comma-separated model weights, e.g. yolov8n.pt,yolov8s.pt,yolov8m.pt")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP")
    c.add_argument("--out", default=None, help="optional scorecard JSON path")
    c.add_argument("--coco", action="store_true",
                   help="treat --images/--ann as a COCO val dir + instances JSON")
    c.add_argument("--limit", type=int, default=None, help="COCO: keep first N images")
    c.set_defaults(func=compare)

    rp = sub.add_parser("report", help="render a bench/compare results JSON to an HTML report")
    rp.add_argument("--in", dest="input", required=True, help="results JSON (from bench/compare)")
    rp.add_argument("--out", required=True, help="output HTML path")
    rp.add_argument("--title", default="Robustness Assurance Report", help="report title")
    rp.set_defaults(func=report)

    t = sub.add_parser("tevv", help="judge a bench result vs acceptance criteria (TEVV verdict)")
    t.add_argument("--in", dest="input", required=True, help="bench results JSON")
    t.add_argument("--out", required=True, help="output HTML assurance-case path")
    t.add_argument("--model", default="model under test", help="model name for the claim")
    t.add_argument("--title", default="TEVV Robustness Assurance Case", help="report title")
    t.add_argument("--min-clean", type=float, default=0.30,
                   help="baseline competence: required clean mAP")
    t.add_argument("--min-retained", type=float, default=0.50,
                   help="per-condition robustness floor (fraction of clean mAP retained)")
    t.set_defaults(func=tevv)

    m = sub.add_parser("monitor",
                       help="diff a current bench result vs a baseline; gate on regression")
    m.add_argument("--baseline", required=True, help="locked baseline bench results JSON")
    m.add_argument("--current", required=True, help="current bench results JSON to check")
    m.add_argument("--out", default=None, help="optional output HTML diff report path")
    m.add_argument("--out-json", dest="out_json", default=None,
                   help="optional output diff JSON path")
    m.add_argument("--model", default="model under test", help="model name for the report")
    m.add_argument("--title", default="Robustness Regression Monitor", help="report title")
    m.add_argument("--abs-floor", dest="abs_floor", type=float, default=0.02,
                   help="absolute mAP-drop floor to flag a regression")
    m.add_argument("--rel-floor", dest="rel_floor", type=float, default=0.05,
                   help="relative mAP-drop floor (fraction of baseline) to flag a regression")
    m.add_argument("--no-gate", dest="no_gate", action="store_true",
                   help="report only; always exit 0 (don't fail CI on regression)")
    m.set_defaults(func=monitor)

    v = sub.add_parser("video", help="run a degradation over sampled video frames (GT-free)")
    v.add_argument("--video", required=True, help="path to a video file")
    v.add_argument("--frames", type=int, default=8, help="number of frames to sample")
    v.add_argument("--model", default="yolov8n.pt", help="'fake' or a YOLO weights path/name")
    v.add_argument("--mode", default="fog", choices=list(DEGRADATION_MODES),
                   help="degradation mode to apply per frame")
    v.add_argument("--severity", type=float, default=0.8, help="degradation severity in [0,1]")
    v.add_argument("--seed", type=int, default=0)
    v.add_argument("--out", default=None, help="optional results JSON path")
    v.set_defaults(func=video)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
