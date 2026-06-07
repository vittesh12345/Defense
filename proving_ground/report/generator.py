"""Build a ``Report`` from measured metrics and write it to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from proving_ground import __version__
from proving_ground.eval.robustness import RobustnessReport
from proving_ground.report.schema import (
    SCHEMA_VERSION,
    AttackResult,
    DatasetInfo,
    Report,
    ReportMeta,
)


def build_report(
    *,
    model: str,
    seed: int,
    iou_threshold: float,
    dataset: DatasetInfo,
    clean_map: float,
    clean_per_class: dict[str, float],
    attack_name: str,
    attack_params: dict[str, float],
    robustness: RobustnessReport,
    timestamp_utc: str | None = None,
) -> Report:
    """Assemble a single-attack report from measured metrics."""
    meta = ReportMeta(
        tool_version=__version__,
        schema_version=SCHEMA_VERSION,
        seed=seed,
        model=model,
        iou_threshold=iou_threshold,
        dataset=dataset,
        timestamp_utc=timestamp_utc,
    )
    attack = AttackResult(
        name=attack_name,
        params=attack_params,
        attacked_map=robustness.attacked_map,
        attacked_per_class=robustness.attacked_per_class,
        map_delta=robustness.map_delta,
        per_class_delta=robustness.per_class_delta,
    )
    return Report(
        meta=meta,
        clean_map=clean_map,
        clean_per_class=clean_per_class,
        attacks=[attack],
    )


def write_report(report: Report, path: str | Path) -> Path:
    """Write the report as pretty JSON. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return path
