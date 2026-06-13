"""Multi-model robustness comparison — the client-facing scorecard.

The product's value is not "detect objects" but "rate how robust a model is".
This runs the identical canonical attack suite across several detectors and ranks
them by how much detection performance they *retain* under attack — an
independent, cross-vendor robustness scorecard.

Retained fraction = mean(attacked mAP over the suite) / clean mAP. A model can
have high clean accuracy yet a low retained fraction (fragile under attack); the
scorecard surfaces exactly that trade-off.
"""

from __future__ import annotations

from collections.abc import Sequence

from proving_ground.adapters.base import Detector
from proving_ground.attacks.base import Attack
from proving_ground.benchmark import run_benchmark
from proving_ground.data.loaders import Sample


def run_comparison(
    detectors: Sequence[tuple[Detector, str]],
    samples: Sequence[Sample],
    classes: Sequence[str],
    attacks: Sequence[tuple[Attack, dict[str, float]]],
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Run the attack suite on each detector; return a ranked scorecard."""
    models = []
    for detector, label in detectors:
        bench = run_benchmark(detector, samples, classes, attacks, iou_threshold, seed)
        clean = bench["clean_map"]
        attacked = [a["attacked_map"] for a in bench["attacks"]]
        mean_attacked = sum(attacked) / len(attacked) if attacked else 0.0
        retained = mean_attacked / clean if clean > 0 else 0.0
        weakest = (
            min(bench["attacks"], key=lambda a: a["attacked_map"]) if bench["attacks"] else None
        )
        models.append({
            "model": label,
            "clean_map": clean,
            "mean_attacked_map": mean_attacked,
            "robustness_retained": retained,
            "weakest_attack": weakest["name"] if weakest else None,
            "attacks": bench["attacks"],
        })
    # Rank by retained fraction (most robust first); clean mAP breaks ties.
    models.sort(key=lambda m: (m["robustness_retained"], m["clean_map"]), reverse=True)
    return {
        "num_images": len(samples),
        "iou_threshold": iou_threshold,
        "num_attacks": len(attacks),
        "models": models,
    }


def comparison_table(result: dict) -> str:
    """Render the scorecard as a markdown table (ranked, most robust first)."""
    lines = [
        f"Robustness scorecard — {result['num_attacks']} attacks over "
        f"{result['num_images']} images, mAP@{result['iou_threshold']} "
        "(ranked by performance retained under attack):",
        "",
        "| Rank | Model | Clean mAP | Mean attacked mAP | Retained | Weakest vs |",
        "|---|---|---|---|---|---|",
    ]
    for i, m in enumerate(result["models"], 1):
        lines.append(
            f"| {i} | {m['model']} | {m['clean_map']:.3f} | "
            f"{m['mean_attacked_map']:.3f} | {m['robustness_retained'] * 100:.0f}% | "
            f"{m['weakest_attack']} |"
        )
    return "\n".join(lines)
