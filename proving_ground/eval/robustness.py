"""Robustness deltas: how much an attack degraded the measured performance.

A positive delta means the metric dropped under attack (clean - attacked), which
is the intuitive "how much did we lose" direction.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobustnessReport:
    """Clean vs attacked metrics and their deltas."""

    clean_map: float
    attacked_map: float
    map_delta: float  # clean - attacked
    clean_per_class: dict[str, float]
    attacked_per_class: dict[str, float]
    per_class_delta: dict[str, float]


def robustness_delta(
    clean_map: float,
    attacked_map: float,
    clean_per_class: dict[str, float],
    attacked_per_class: dict[str, float],
) -> RobustnessReport:
    """Combine clean and attacked metrics into a robustness report.

    Per-class deltas are computed over the union of classes seen in either run;
    a class missing from one side is treated as 0.0 there.
    """
    classes = sorted(set(clean_per_class) | set(attacked_per_class))
    per_class_delta = {
        c: clean_per_class.get(c, 0.0) - attacked_per_class.get(c, 0.0) for c in classes
    }
    return RobustnessReport(
        clean_map=clean_map,
        attacked_map=attacked_map,
        map_delta=clean_map - attacked_map,
        clean_per_class=clean_per_class,
        attacked_per_class=attacked_per_class,
        per_class_delta=per_class_delta,
    )
