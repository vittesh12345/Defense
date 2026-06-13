"""Fast-tier tests for the continuous-monitoring regression diff (no weights)."""

from __future__ import annotations

from proving_ground.report.monitor import diff_results, render_monitor_html


def _res(clean, attacked):
    return {"clean_map": clean, "num_images": 50, "iou_threshold": 0.5,
            "attacks": [{"name": n, "attacked_map": m} for n, m in attacked]}


def test_no_regression_when_stable():
    base = _res(0.50, [("fgsm", 0.30)])
    cur = _res(0.495, [("fgsm", 0.299)])  # tiny drop, below both floors
    d = diff_results(base, cur, abs_floor=0.02, rel_floor=0.05)
    assert d["verdict"] == "OK" and d["n_regressed"] == 0


def test_regression_needs_both_floors():
    base = _res(0.50, [("fgsm", 0.30)])
    # clean drops 0.04 abs (>=0.02) AND 8% rel (>=0.05) -> regressed
    cur = _res(0.46, [("fgsm", 0.30)])
    d = diff_results(base, cur, abs_floor=0.02, rel_floor=0.05)
    assert d["verdict"] == "REGRESSED"
    assert d["regressions"] == ["clean (no attack)"]


def test_abs_only_does_not_gate():
    # huge relative drop but tiny absolute (below abs floor) -> not a regression
    base = _res(0.50, [("fgsm", 0.04)])
    cur = _res(0.50, [("fgsm", 0.025)])  # 0.015 abs (< 0.02), 37% rel
    d = diff_results(base, cur, abs_floor=0.02, rel_floor=0.05)
    assert d["verdict"] == "OK"


def test_rel_only_does_not_gate():
    # 0.03 abs drop (>= floor) but only 3% rel (< 0.05) -> not a regression
    base = _res(1.00, [("fgsm", 0.50)])
    cur = _res(0.97, [("fgsm", 0.50)])
    d = diff_results(base, cur, abs_floor=0.02, rel_floor=0.05)
    assert d["verdict"] == "OK"


def test_improvement_and_coverage_changes():
    base = _res(0.40, [("fgsm", 0.20), ("old_only", 0.10)])
    cur = _res(0.50, [("fgsm", 0.20), ("new_only", 0.30)])  # clean improved a lot
    d = diff_results(base, cur)
    assert d["verdict"] == "OK"
    clean_row = next(r for r in d["rows"] if r["name"] == "clean (no attack)")
    assert clean_row["status"] == "improved"
    assert d["new_conditions"] == ["new_only"]
    assert d["dropped_conditions"] == ["old_only"]


def test_render_banner_escaped():
    base = _res(0.50, [("<x>", 0.30)])
    cur = _res(0.40, [("<x>", 0.30)])  # clean regressed
    d = diff_results(base, cur)
    h = render_monitor_html(d, model="<m>", title="T")
    assert "<!doctype html>" in h.lower()
    assert "VERDICT: REGRESSED" in h
    assert "<x>" not in h and "&lt;x&gt;" in h and "&lt;m&gt;" in h
