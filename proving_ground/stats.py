"""Confidence intervals for the benchmark via two independent variance sources.

Single-seed point estimates are easy to challenge. This adds error bars two ways:

* **Seed sweep** — run each attack over K seeds and summarise the spread of its
  attacked mAP. Captures sensitivity to the attack's own randomness. Deterministic
  attacks (FGSM, patch, PGD without random-init) collapse to a zero-width interval;
  stochastic ones (EOT, gaussian-noise, low-light) get real intervals.
* **Image bootstrap** — resample the image set with replacement and recompute the
  pooled mAP. Captures "would a different sample of images change the number",
  which gives an interval for *every* attack and surfaces small-fixture uncertainty.

Both are deterministic given fixed seeds (the bootstrap RNG is seeded), so the
result is lockable.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np

from proving_ground.adapters.base import Detector
from proving_ground.benchmark import default_attacks
from proving_ground.data.loaders import Sample
from proving_ground.eval.metrics import mean_average_precision
from proving_ground.seeding import set_seed

Z95 = 1.959963984540054  # normal-approx 95% two-sided


def summarize_seed(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample std and a normal-approx 95% CI over seed repeats."""
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    half = Z95 * std / np.sqrt(arr.size) if arr.size > 1 else 0.0
    return {
        "mean": mean, "std": std,
        "ci_lo": mean - half, "ci_hi": mean + half,
        "min": float(arr.min()), "max": float(arr.max()),
    }


def summarize_bootstrap(values: Sequence[float]) -> dict[str, float]:
    """Mean and a 2.5/97.5 percentile CI over bootstrap resamples."""
    arr = np.asarray(values, dtype=float)
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"mean": float(arr.mean()), "ci_lo": float(lo), "ci_hi": float(hi)}


def compute_ci(
    detector: Detector,
    samples: Sequence[Sample],
    classes: Sequence[str],
    n_seeds: int = 5,
    n_bootstrap: int = 1000,
    base_seed: int = 0,
    iou_threshold: float = 0.5,
) -> dict:
    """Seed-sweep + image-bootstrap CIs for the canonical attack suite."""
    gts = [s.ground_truth for s in samples]
    seeds = [base_seed + i for i in range(n_seeds)]

    set_seed(base_seed)
    clean_per_image = [detector.predict(s.image) for s in samples]
    clean_map = mean_average_precision(clean_per_image, gts, classes, iou_threshold)

    # Seed sweep: each attack's pooled attacked mAP at each seed. Capture the
    # base-seed per-image predictions to feed the (cheap) image bootstrap.
    seed_maps: dict[str, list[float]] = defaultdict(list)
    base_preds: dict[str, list] = {}
    for i, seed in enumerate(seeds):
        for attack, _params in default_attacks(seed=seed):
            set_seed(seed)
            preds = [detector.predict(attack.apply(detector, s.image, s.ground_truth))
                     for s in samples]
            seed_maps[attack.name].append(
                mean_average_precision(preds, gts, classes, iou_threshold)
            )
            if i == 0:
                base_preds[attack.name] = preds

    # Image bootstrap: resample image indices with replacement; recompute pooled
    # mAP from the cached base-seed predictions (no attacks re-run).
    rng = np.random.default_rng(base_seed)
    n = len(samples)
    resamples = rng.integers(0, n, size=(n_bootstrap, n))
    boot: dict[str, list[float]] = defaultdict(list)
    for idx in resamples.tolist():
        bgts = [gts[j] for j in idx]
        boot["__clean__"].append(
            mean_average_precision([clean_per_image[j] for j in idx], bgts, classes, iou_threshold)
        )
        for name, preds in base_preds.items():
            boot[name].append(
                mean_average_precision([preds[j] for j in idx], bgts, classes, iou_threshold)
            )

    attacks = [
        {
            "name": name,
            "seed_ci": summarize_seed(seed_maps[name]),
            "bootstrap_ci": summarize_bootstrap(boot[name]),
        }
        for name in seed_maps
    ]
    return {
        "seeds": seeds,
        "n_bootstrap": n_bootstrap,
        "iou_threshold": iou_threshold,
        "clean_map": clean_map,
        "clean_bootstrap_ci": summarize_bootstrap(boot["__clean__"]),
        "attacks": attacks,
    }


def ci_table(ci: dict) -> str:
    """Render the CI result as a markdown table."""
    lines = [
        f"Pooled mAP@{ci['iou_threshold']} over {len(ci['seeds'])} seeds and "
        f"{ci['n_bootstrap']} image-bootstraps (clean = {ci['clean_map']:.3f}, "
        f"bootstrap 95% CI [{ci['clean_bootstrap_ci']['ci_lo']:.3f}, "
        f"{ci['clean_bootstrap_ci']['ci_hi']:.3f}]):",
        "",
        "| Attack | Attacked mAP (mean) | Seed 95% CI | Image-bootstrap 95% CI |",
        "|---|---|---|---|",
    ]
    for a in ci["attacks"]:
        s, b = a["seed_ci"], a["bootstrap_ci"]
        lines.append(
            f"| {a['name']} | {s['mean']:.3f} | "
            f"[{s['ci_lo']:.3f}, {s['ci_hi']:.3f}] | [{b['ci_lo']:.3f}, {b['ci_hi']:.3f}] |"
        )
    return "\n".join(lines)
