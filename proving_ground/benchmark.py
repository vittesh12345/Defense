"""Aggregate benchmark: run several attacks over an image set, pooled mAP.

Produces the headline results table — clean mAP and attacked mAP per attack,
pooled over the whole set (the same dataset-level mAP the rest of the project
uses). The run is deterministic: the seed is re-pinned before the clean pass and
before each attack, so attack order can't change the numbers.
"""

from __future__ import annotations

from collections.abc import Sequence

from proving_ground.adapters.base import Detector
from proving_ground.attacks.base import Attack
from proving_ground.attacks.degradation import MODES as DEGRADATION_MODES
from proving_ground.attacks.degradation import DegradationAttack
from proving_ground.attacks.eot_patch import EOTPatchAttack
from proving_ground.attacks.fgsm import FGSM
from proving_ground.attacks.patch import PatchAttack
from proving_ground.attacks.pgd import PGDLinf
from proving_ground.data.loaders import Sample
from proving_ground.eval.metrics import mean_average_precision
from proving_ground.seeding import set_seed

# Severity used for the DVE entries in the canonical suite — matches the README
# headline drop table and the high-severity row of the degradation snapshot.
DVE_SEVERITY = 0.8


def default_attacks(seed: int = 0) -> list[tuple[Attack, dict[str, float]]]:
    """The canonical attack suite (params kept in sync with the locked baselines).

    White-box attacks first, then the six DVE degradations at severity 0.8 —
    one ``bench`` invocation now produces the full README headline.
    """
    suite: list[tuple[Attack, dict[str, float]]] = [
        (FGSM(eps=0.03), {"eps": 0.03}),
        (PGDLinf(eps=0.03, steps=10, step_size=0.0075),
         {"eps": 0.03, "steps": 10.0, "step_size": 0.0075, "random_init": 0.0}),
        (PatchAttack(size=0.4, location="center", steps=20, step_size=0.1),
         {"patch_size": 0.4, "steps": 20.0, "step_size": 0.1}),
        (EOTPatchAttack(size=0.4, location="center", steps=15, step_size=0.1, eot_samples=4,
                        scale_min=0.8, scale_max=1.2, rot_deg=12.0, trans=0.05,
                        brightness=0.1, contrast=0.2, seed=seed),
         {"patch_size": 0.4, "steps": 15.0, "step_size": 0.1, "eot_samples": 4.0,
          "scale_min": 0.8, "scale_max": 1.2, "rot_deg": 12.0, "trans": 0.05,
          "brightness": 0.1, "contrast": 0.2}),
    ]
    for mode in DEGRADATION_MODES:
        suite.append((
            DegradationAttack(mode=mode, severity=DVE_SEVERITY, seed=seed),
            {"severity": DVE_SEVERITY},
        ))
    return suite


def run_benchmark(
    detector: Detector,
    samples: Sequence[Sample],
    classes: Sequence[str],
    attacks: Sequence[tuple[Attack, dict[str, float]]],
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Run clean + each attack over the set; return a pooled-mAP results dict."""
    gts = [s.ground_truth for s in samples]

    set_seed(seed)
    clean_preds = [detector.predict(s.image) for s in samples]
    clean_map = mean_average_precision(clean_preds, gts, classes, iou_threshold)

    results: dict = {
        "num_images": len(samples),
        "iou_threshold": iou_threshold,
        "clean_map": clean_map,
        "attacks": [],
    }
    for attack, params in attacks:
        set_seed(seed)  # each attack starts from the same RNG state
        attacked = [attack.apply(detector, s.image, s.ground_truth) for s in samples]
        attacked_preds = [detector.predict(img) for img in attacked]
        attacked_map = mean_average_precision(attacked_preds, gts, classes, iou_threshold)
        results["attacks"].append({
            "name": attack.name,
            "params": params,
            "attacked_map": attacked_map,
            "map_delta": clean_map - attacked_map,
        })
    return results


def results_table(results: dict) -> str:
    """Render the results as a markdown table."""
    clean = results["clean_map"]
    lines = [
        f"Pooled mAP@{results['iou_threshold']} over {results['num_images']} images "
        f"(clean = {clean:.3f}):",
        "",
        "| Attack | Clean mAP | Attacked mAP | Δ (drop) |",
        "|---|---|---|---|",
    ]
    for a in results["attacks"]:
        lines.append(
            f"| {a['name']} | {clean:.3f} | {a['attacked_map']:.3f} | {a['map_delta']:.3f} |"
        )
    return "\n".join(lines)
