"""Versioned assurance-report schema.

The report is the product, so its shape is explicit and versioned. Anything that
is *not* a reproducible measurement (wall-clock timestamp) lives under ``meta``
and is documented as excluded from value comparisons / golden tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

SCHEMA_VERSION = "0.1.0"


@dataclass(frozen=True)
class DatasetInfo:
    images_dir: str
    annotations_path: str
    num_images: int
    classes: list[str]


@dataclass(frozen=True)
class ReportMeta:
    tool_version: str
    schema_version: str
    seed: int
    model: str
    iou_threshold: float
    dataset: DatasetInfo
    # Non-reproducible; excluded from any value/golden comparison.
    timestamp_utc: str | None = None


@dataclass(frozen=True)
class AttackResult:
    name: str
    params: dict[str, float]
    attacked_map: float
    attacked_per_class: dict[str, float]
    map_delta: float  # clean - attacked
    per_class_delta: dict[str, float]


@dataclass(frozen=True)
class Report:
    meta: ReportMeta
    clean_map: float
    clean_per_class: dict[str, float]
    attacks: list[AttackResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
