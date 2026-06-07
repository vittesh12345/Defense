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
from proving_ground.attacks.fgsm import FGSM
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
    if model.endswith(".pt") or model.startswith("yolo"):
        from proving_ground.adapters.yolo import UltralyticsYOLOAdapter

        return UltralyticsYOLOAdapter(model), model
    raise ValueError(f"unknown model: {model!r} (use 'fake' or a YOLO weights path)")


def _predict_all(detector: Detector, samples: list[Sample]) -> list[list[Detection]]:
    return [detector.predict(s.image) for s in samples]


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
    attack = FGSM(eps=args.eps)
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
        attack_params={"eps": args.eps},
        robustness=rob,
        timestamp_utc=datetime.now(UTC).isoformat(),
    )
    out = write_report(report, args.out)

    print(f"clean mAP@{args.iou}:    {clean_map:.4f}")
    print(f"attacked mAP@{args.iou}: {attacked_map:.4f}  (delta {rob.map_delta:+.4f})")
    print(f"report written: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="proving-ground")
    sub = parser.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="run one robustness evaluation")
    r.add_argument("--images", required=True, help="directory of images")
    r.add_argument("--ann", required=True, help="annotations JSON")
    r.add_argument("--model", default="fake", help="'fake' or a YOLO weights path/name")
    r.add_argument("--attack", default="fgsm", choices=["fgsm"], help="attack to run")
    r.add_argument("--eps", type=float, default=0.03, help="FGSM L-inf budget in [0,1]")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--iou", type=float, default=0.5, help="IoU threshold for mAP")
    r.add_argument("--out", default="report.json", help="output JSON path")
    r.set_defaults(func=run)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
