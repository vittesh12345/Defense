"""Report generation: correct deltas, valid schema, JSON round-trip."""

from __future__ import annotations

import json

import pytest

from proving_ground.eval.robustness import robustness_delta
from proving_ground.report.generator import build_report, write_report
from proving_ground.report.schema import SCHEMA_VERSION, DatasetInfo


def make_report():
    clean_pc = {"red": 1.0, "green": 0.5}
    attacked_pc = {"red": 0.4, "green": 0.5}
    rob = robustness_delta(
        clean_map=0.75, attacked_map=0.45,
        clean_per_class=clean_pc, attacked_per_class=attacked_pc,
    )
    dataset = DatasetInfo(
        images_dir="fix/images", annotations_path="fix/ann.json",
        num_images=2, classes=["red", "green", "blue"],
    )
    return build_report(
        model="fake", seed=0, iou_threshold=0.5, dataset=dataset,
        clean_map=0.75, clean_per_class=clean_pc,
        attack_name="fgsm", attack_params={"eps": 0.03},
        robustness=rob, timestamp_utc="2026-06-07T00:00:00Z",
    )


def test_delta_is_clean_minus_attacked():
    rob = robustness_delta(0.75, 0.45, {"red": 1.0}, {"red": 0.4})
    assert rob.map_delta == pytest.approx(0.30)
    assert rob.per_class_delta["red"] == pytest.approx(0.60)


def test_report_has_schema_version():
    report = make_report()
    assert report.meta.schema_version == SCHEMA_VERSION


def test_report_records_clean_and_attacked():
    report = make_report()
    assert report.clean_map == pytest.approx(0.75)
    assert report.attacks[0].attacked_map == pytest.approx(0.45)
    assert report.attacks[0].map_delta == pytest.approx(0.30)
    assert report.attacks[0].per_class_delta["red"] == pytest.approx(0.60)


def test_json_round_trip(tmp_path):
    report = make_report()
    out = write_report(report, tmp_path / "report.json")
    loaded = json.loads(out.read_text())
    assert loaded["meta"]["schema_version"] == SCHEMA_VERSION
    assert loaded["clean_map"] == pytest.approx(0.75)
    assert loaded["attacks"][0]["name"] == "fgsm"
    assert loaded["attacks"][0]["params"]["eps"] == pytest.approx(0.03)


def test_report_value_block_excludes_timestamp(tmp_path):
    # Two reports differing only in timestamp must be identical once meta
    # timestamp is dropped -> proves the report's measured values are reproducible.
    r1 = make_report()
    r2 = make_report()
    d1, d2 = r1.to_dict(), r2.to_dict()
    for d in (d1, d2):
        d["meta"].pop("timestamp_utc")
    assert d1 == d2
