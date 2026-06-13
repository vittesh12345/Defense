"""Fast-tier tests for the TEVV assurance-case assessment + render (no weights)."""

from __future__ import annotations

from proving_ground.report.tevv import assess, render_tevv_html


def _res(clean, attacked):
    return {"clean_map": clean, "num_images": 50, "iou_threshold": 0.5,
            "attacks": [{"name": n, "attacked_map": m, "map_delta": clean - m}
                        for n, m in attacked]}


def test_verdict_pass():
    a = assess(_res(0.5, [("fgsm", 0.30), ("fog", 0.40)]), min_clean_map=0.3, min_retained=0.5)
    assert a["verdict"] == "PASS" and a["baseline_pass"] and not a["failing"]


def test_verdict_conditional():
    # pgd retained 0.1/0.5 = 0.2 < 0.5 -> fails; fgsm 0.6 -> passes
    a = assess(_res(0.5, [("fgsm", 0.30), ("pgd", 0.10)]), min_clean_map=0.3, min_retained=0.5)
    assert a["verdict"] == "CONDITIONAL"
    assert a["failing"] == ["pgd"] and a["n_pass"] == 1


def test_verdict_fail_baseline():
    a = assess(_res(0.20, [("fgsm", 0.10)]), min_clean_map=0.3, min_retained=0.5)
    assert a["verdict"] == "FAIL" and not a["baseline_pass"]


def test_render_has_verdict_claim_and_doctype():
    a = assess(_res(0.5, [("fgsm", 0.30)]), 0.3, 0.5)
    h = render_tevv_html(a, model="yolov8n", title="T")
    assert "<!doctype html>" in h.lower()
    assert "VERDICT: PASS" in h and "yolov8n" in h and "Claim" in h


def test_render_escapes_names():
    a = assess(_res(0.5, [("<x>", 0.30)]), 0.3, 0.5)
    h = render_tevv_html(a, model="<m>")
    assert "<x>" not in h and "&lt;x&gt;" in h
    assert "&lt;m&gt;" in h
