"""AutoAttack-style worst-case ensemble.

A single attack reports *one* model's-eye view of robustness; a defender who
quotes "we survive PGD" has said nothing about C&W or patches. AutoAttack's idea
is to stop cherry-picking: run an ensemble and, **per example**, keep whichever
attack hurt the most. The resulting accuracy is a *lower bound* — the robustness
you can actually count on against an adversary free to choose their method.

Adapted to detection: for each image we run every attack, score that image's
mAP under each, and select the attack that drove the image's mAP lowest. Pooling
those per-image worst cases gives ``ensemble_map`` — by construction no higher
than any single image could manage against its own worst attacker, and in
practice a tighter bound than the strongest single attack's pooled mAP.

We also report which attack "won" each image (``win_counts``): a diverse winner
spread is evidence the ensemble is doing real work and no single attack dominates.

Seeding mirrors ``run_benchmark`` (each attack re-seeded from ``seed`` before it
sweeps the image set), so the numbers line up with the per-attack benchmark and
are reproducible.
"""

from __future__ import annotations

from collections.abc import Sequence

from proving_ground.adapters.base import Detector
from proving_ground.attacks.base import Attack
from proving_ground.data.loaders import Sample
from proving_ground.eval.metrics import mean_average_precision
from proving_ground.seeding import set_seed


def run_ensemble(
    detector: Detector,
    samples: Sequence[Sample],
    classes: Sequence[str],
    attacks: Sequence[tuple[Attack, dict[str, float]]],
    iou_threshold: float = 0.5,
    seed: int = 0,
) -> dict:
    """Worst-case-per-image ensemble over ``attacks``; returns a results dict."""
    if not attacks:
        raise ValueError("ensemble needs at least one attack")
    gts = [s.ground_truth for s in samples]
    n = len(samples)

    set_seed(seed)
    clean_preds = [detector.predict(s.image) for s in samples]
    clean_map = mean_average_precision(clean_preds, gts, classes, iou_threshold)

    names: list[str] = []
    pooled: list[float] = []           # each attack's own pooled mAP
    per_image: list[list[float]] = []  # [attack][image] mAP
    preds_by_attack: list[list] = []   # [attack][image] predictions
    for attack, _ in attacks:
        names.append(attack.name)
        set_seed(seed)  # same RNG state as run_benchmark, for matching numbers
        advs = [attack.apply(detector, s.image, s.ground_truth) for s in samples]
        preds = [detector.predict(a) for a in advs]
        preds_by_attack.append(preds)
        pooled.append(mean_average_precision(preds, gts, classes, iou_threshold))
        per_image.append(
            [mean_average_precision([preds[i]], [gts[i]], classes, iou_threshold) for i in range(n)]
        )

    # Per image, pick the attack with the lowest mAP (stable: first on ties).
    selected_preds, winners, worst_per_image = [], [], []
    for i in range(n):
        best_a, best_m = 0, per_image[0][i]
        for a in range(1, len(attacks)):
            if per_image[a][i] < best_m:
                best_a, best_m = a, per_image[a][i]
        selected_preds.append(preds_by_attack[best_a][i])
        winners.append(names[best_a])
        worst_per_image.append(best_m)
    ensemble_map = mean_average_precision(selected_preds, gts, classes, iou_threshold)

    # Strongest single attack = lowest pooled mAP (stable on ties).
    strongest_idx = min(range(len(attacks)), key=lambda a: (pooled[a], a))

    return {
        "kind": "ensemble",
        "num_images": n,
        "iou_threshold": iou_threshold,
        "clean_map": clean_map,
        "ensemble_map": ensemble_map,
        "map_delta": clean_map - ensemble_map,
        "attacks": names,
        "single_attack_maps": dict(zip(names, pooled, strict=True)),
        "win_counts": {name: winners.count(name) for name in names},
        "strongest_single_attack": names[strongest_idx],
        "strongest_single_map": pooled[strongest_idx],
        "ensemble_gain_over_single": pooled[strongest_idx] - ensemble_map,
    }


def ensemble_table(results: dict) -> str:
    """One-screen console summary of an ensemble result."""
    r = results
    lines = [
        f"clean mAP@{r['iou_threshold']}:           {r['clean_map']:.4f}",
        f"worst-case ensemble mAP:    {r['ensemble_map']:.4f}  "
        f"(drop {r['map_delta']:+.4f})",
        f"strongest single attack:    {r['strongest_single_attack']} "
        f"@ {r['strongest_single_map']:.4f}  "
        f"(ensemble is {r['ensemble_gain_over_single']:+.4f} tighter)",
        "per-attack pooled mAP / images-won:",
    ]
    for name in r["attacks"]:
        lines.append(
            f"  {name:28s} {r['single_attack_maps'][name]:.4f}   "
            f"won {r['win_counts'][name]}/{r['num_images']}"
        )
    return "\n".join(lines)
