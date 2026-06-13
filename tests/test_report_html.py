"""Fast-tier tests for the HTML report renderer (no weights)."""

from __future__ import annotations

from proving_ground.report.html import render_html

BENCH = {
    "clean_map": 0.29, "num_images": 6, "iou_threshold": 0.5,
    "attacks": [
        {"name": "fgsm", "attacked_map": 0.07, "map_delta": 0.22},
        {"name": "pgd-linf", "attacked_map": 0.01, "map_delta": 0.28},
    ],
}
SCORECARD = {
    "num_attacks": 13, "num_images": 6,
    "models": [{"model": "yolov8n.pt", "clean_map": 0.29, "mean_attacked_map": 0.12,
                "robustness_retained": 0.40, "weakest_attack": "pgd-linf"}],
}


def test_bench_report_renders_numbers():
    h = render_html(BENCH, "Test Report")
    assert "<!doctype html>" in h.lower()
    assert "Test Report" in h
    assert "fgsm" in h and "pgd-linf" in h
    assert "0.290" in h  # clean mAP


def test_scorecard_report_shows_caveat():
    h = render_html(SCORECARD)
    assert "yolov8n.pt" in h
    assert "Methodology note" in h  # the incomplete-GT caveat banner
    assert "COCO val" in h


def test_html_escapes_names():
    res = {"clean_map": 0.5, "num_images": 1, "iou_threshold": 0.5,
           "attacks": [{"name": "<script>", "attacked_map": 0.1, "map_delta": 0.4}]}
    h = render_html(res)
    assert "<script>" not in h
    assert "&lt;script&gt;" in h
