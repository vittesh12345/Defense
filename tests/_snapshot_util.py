"""Helpers for the locked real-pipeline snapshot.

Normalisation drops the only environment-dependent fields (wall-clock timestamp
and absolute filesystem paths) so a baseline is comparable across runs/machines.
Comparison is exact for structure and within-tolerance for numeric leaves, so a
genuine metric drift fails loudly while harmless float noise does not.
"""

from __future__ import annotations

import copy
import math

FLOAT_TOL = 1e-4


def normalize_report(report: dict) -> dict:
    """Return a copy with non-reproducible / machine-specific fields removed."""
    d = copy.deepcopy(report)
    meta = d.get("meta", {})
    meta.pop("timestamp_utc", None)
    dataset = meta.get("dataset", {})
    dataset.pop("images_dir", None)  # absolute path, machine-specific
    dataset.pop("annotations_path", None)
    return d


def diff_against_baseline(produced: dict, baseline: dict, path: str = "") -> list[str]:
    """Recursively compare. Floats within FLOAT_TOL; everything else exact."""
    diffs: list[str] = []
    if isinstance(baseline, dict):
        if not isinstance(produced, dict):
            return [f"{path}: type mismatch (dict vs {type(produced).__name__})"]
        for key in set(baseline) | set(produced):
            p = f"{path}.{key}" if path else key
            if key not in produced:
                diffs.append(f"{p}: missing in produced")
            elif key not in baseline:
                diffs.append(f"{p}: unexpected in produced")
            else:
                diffs += diff_against_baseline(produced[key], baseline[key], p)
    elif isinstance(baseline, list):
        if not isinstance(produced, list) or len(produced) != len(baseline):
            return [f"{path}: list mismatch (len {len(baseline)} vs "
                    f"{len(produced) if isinstance(produced, list) else 'n/a'})"]
        for i, (p_item, b_item) in enumerate(zip(produced, baseline, strict=True)):
            diffs += diff_against_baseline(p_item, b_item, f"{path}[{i}]")
    elif isinstance(baseline, float) or isinstance(produced, float):
        if not math.isclose(float(produced), float(baseline), abs_tol=FLOAT_TOL):
            diffs.append(f"{path}: {produced} != {baseline} (tol {FLOAT_TOL})")
    else:
        if produced != baseline:
            diffs.append(f"{path}: {produced!r} != {baseline!r}")
    return diffs
