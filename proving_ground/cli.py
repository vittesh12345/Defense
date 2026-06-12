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
from proving_ground.attacks.degradation import MODES as DEGRADATION_MODES
from proving_ground.attacks.degradation import DegradationAttack
from proving_ground.attacks.eot_patch import EOTPatchAttack
from proving_ground.attacks.fgsm import FGSM
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
    if args.attack == "degradation":
        attack = DegradationAttack(mode=args.mode, severity=args.severity, seed=args.seed)
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
    samples, classes = load_dataset(args.images, args.ann)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proving-ground")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one robustness evaluation")
    r.add_argument("--images", required=True, help="directory of images")
    r.add_argument("--ann", required=True, help="annotations JSON")
    r.add_argument("--model", default="fake", help="'fake' or a YOLO weights path/name")
    r.add_argument("--attack", default="fgsm",
                   choices=["fgsm", "pgd-linf", "pgd-l2", "patch", "eot-patch", "degradation"],
                   help="attack to run")
    r.add_argument("--mode", default="gaussian_blur", choices=list(DEGRADATION_MODES),
                   help="degradation: which degradation mode")
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
    b.set_defaults(func=bench)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
