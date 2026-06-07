"""End-to-end CLI smoke test on the committed 2-image fixture.

Uses the weight-free FakeDetector (which also implements WhiteBox), so the real
clean-eval -> FGSM -> attacked-eval -> report path runs with no downloads.
"""

from __future__ import annotations

import json

from proving_ground.cli import main
from proving_ground.report.schema import SCHEMA_VERSION


def test_cli_run_produces_valid_report(fixtures_dir, tmp_path):
    out = tmp_path / "report.json"
    rc = main([
        "run",
        "--images", str(fixtures_dir / "images"),
        "--ann", str(fixtures_dir / "annotations.json"),
        "--model", "fake",
        "--attack", "fgsm",
        "--eps", "0.05",
        "--seed", "0",
        "--out", str(out),
    ])
    assert rc == 0
    assert out.exists()

    report = json.loads(out.read_text())
    assert report["meta"]["schema_version"] == SCHEMA_VERSION
    assert report["meta"]["model"] == "fake"
    assert report["meta"]["dataset"]["num_images"] == 2
    assert 0.0 <= report["clean_map"] <= 1.0

    attack = report["attacks"][0]
    assert attack["name"] == "fgsm"
    assert attack["params"]["eps"] == 0.05
    # delta == clean - attacked
    assert attack["map_delta"] == report["clean_map"] - attack["attacked_map"]


def test_cli_run_is_reproducible(fixtures_dir, tmp_path):
    def run_once(path):
        main([
            "run",
            "--images", str(fixtures_dir / "images"),
            "--ann", str(fixtures_dir / "annotations.json"),
            "--model", "fake", "--eps", "0.05", "--seed", "0",
            "--out", str(path),
        ])
        d = json.loads(path.read_text())
        d["meta"].pop("timestamp_utc")  # only non-reproducible field
        return d

    a = run_once(tmp_path / "a.json")
    b = run_once(tmp_path / "b.json")
    assert a == b
