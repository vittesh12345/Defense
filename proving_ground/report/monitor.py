"""Continuous-monitoring regression diff.

A model's robustness drifts: a retrain, a new checkpoint, a quantisation step, or
an upstream dependency bump can silently erode performance under attack. This
compares a *current* `bench` result against a locked *baseline* result and flags
**regressions** — conditions where mAP dropped materially — so the change can be
gated in CI.

A condition is a regression only when the drop clears BOTH floors:
  * absolute — mAP fell by at least ``abs_floor`` (default 0.02), and
  * relative — that fall is at least ``rel_floor`` of the baseline (default 5%).

Requiring both avoids two failure modes: a tiny-mAP condition tripping on noise
(absolute alone) and a 0.5→0.49 wobble tripping on percentage (relative alone).

Verdict: ``REGRESSED`` if any condition regresses, else ``OK``. Conditions that
exist only in one of the two runs are reported (coverage changed) but never gate.
"""

from __future__ import annotations

import html


def _metrics(results: dict) -> dict:
    """Flatten a bench result into {condition_name: mAP}, with clean as a condition."""
    m = {"clean (no attack)": results["clean_map"]}
    for a in results.get("attacks", []):
        m[a["name"]] = a["attacked_map"]
    return m


def diff_results(baseline: dict, current: dict,
                 abs_floor: float = 0.02, rel_floor: float = 0.05) -> dict:
    """Diff a current bench result against a baseline; flag mAP regressions."""
    base, cur = _metrics(baseline), _metrics(current)
    shared = [k for k in base if k in cur]
    new_conditions = sorted(k for k in cur if k not in base)
    dropped_conditions = sorted(k for k in base if k not in cur)

    rows = []
    for name in shared:
        old, new = base[name], cur[name]
        abs_delta = new - old  # negative = worse
        rel_delta = abs_delta / old if old > 0 else 0.0
        drop = -abs_delta
        regressed = drop >= abs_floor and (old > 0 and drop / old >= rel_floor)
        if regressed:
            status = "regressed"
        elif abs_delta >= abs_floor and (old > 0 and abs_delta / old >= rel_floor):
            status = "improved"
        else:
            status = "stable"
        rows.append({
            "name": name, "baseline": old, "current": new,
            "abs_delta": abs_delta, "rel_delta": rel_delta, "status": status,
        })
    rows.sort(key=lambda r: r["abs_delta"])  # worst (most negative) first

    regressions = [r["name"] for r in rows if r["status"] == "regressed"]
    verdict = "REGRESSED" if regressions else "OK"
    if regressions:
        rationale = (f"{len(regressions)} of {len(rows)} conditions regressed beyond the "
                     f"floors (Δ≥{abs_floor:.2f} abs and ≥{rel_floor:.0%} rel): "
                     f"{', '.join(regressions)}.")
    else:
        rationale = (f"No regression: all {len(rows)} shared conditions held within the "
                     f"{abs_floor:.2f} abs / {rel_floor:.0%} rel floors.")

    return {
        "abs_floor": abs_floor, "rel_floor": rel_floor,
        "iou_threshold": current.get("iou_threshold", baseline.get("iou_threshold", 0.5)),
        "baseline_images": baseline.get("num_images"),
        "current_images": current.get("num_images"),
        "rows": rows, "regressions": regressions, "n_regressed": len(regressions),
        "n_conditions": len(rows),
        "new_conditions": new_conditions, "dropped_conditions": dropped_conditions,
        "verdict": verdict, "rationale": rationale,
    }


_VERDICT_COLOR = {"OK": "#1a7f37", "REGRESSED": "#b42318"}

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:#1a2330;background:#f6f8fa;margin:0;padding:32px;}
.card{max-width:900px;margin:0 auto;background:#fff;border:1px solid #d8dee4;
border-radius:10px;padding:28px 34px;box-shadow:0 1px 3px rgba(0,0,0,.06);}
h1{font-size:21px;margin:0 0 2px;} h2{font-size:13px;text-transform:uppercase;
letter-spacing:.05em;color:#5d6b78;margin:22px 0 8px;border-bottom:1px solid #eaeef2;
padding-bottom:4px;}
.verdict{display:inline-block;color:#fff;font-weight:700;font-size:15px;
padding:6px 16px;border-radius:6px;margin:8px 0 4px;}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0;}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #eaeef2;}
th{color:#5d6b78;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;}
td.num{text-align:right;font-variant-numeric:tabular-nums;}
.regressed{color:#b42318;font-weight:700;} .improved{color:#1a7f37;font-weight:700;}
.stable{color:#5d6b78;}
.prov{font-size:12px;color:#5d6b78;margin-top:6px;}
.note{font-size:12.5px;color:#5d6b78;border-top:1px solid #eaeef2;padding-top:12px;margin-top:18px;}
"""

_STATUS_LABEL = {"regressed": "REGRESSED", "improved": "improved", "stable": "stable"}


def render_monitor_html(diff: dict, model: str = "model under test",
                        title: str = "Robustness Regression Monitor") -> str:
    d = diff
    color = _VERDICT_COLOR[d["verdict"]]
    rows = ""
    for r in d["rows"]:
        cls = r["status"]
        badge = f"<span class='{cls}'>{_STATUS_LABEL[cls]}</span>"
        sign = "+" if r["abs_delta"] >= 0 else "−"
        rows += (f"<tr><td>{html.escape(r['name'])}</td>"
                 f"<td class='num'>{r['baseline']:.3f}</td>"
                 f"<td class='num'>{r['current']:.3f}</td>"
                 f"<td class='num'>{sign}{abs(r['abs_delta']):.3f}</td>"
                 f"<td class='num'>{sign}{abs(r['rel_delta']) * 100:.0f}%</td>"
                 f"<td>{badge}</td></tr>")
    cov = ""
    if d["new_conditions"]:
        cov += (f"<p class='prov'>New conditions in current (not gated): "
                f"{html.escape(', '.join(d['new_conditions']))}.</p>")
    if d["dropped_conditions"]:
        cov += (f"<p class='prov'>Conditions dropped from current (coverage gap): "
                f"{html.escape(', '.join(d['dropped_conditions']))}.</p>")
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style></head><body><div class='card'>"
        f"<h1>{html.escape(title)}</h1>"
        f"<p class='prov'>{html.escape(model)} — current vs locked baseline, "
        f"mAP@{d['iou_threshold']}.</p>"
        f"<div class='verdict' style='background:{color}'>VERDICT: {d['verdict']}</div>"
        f"<p class='prov'>{html.escape(d['rationale'])}</p>"
        f"<h2>Per-condition diff</h2>"
        f"<p class='prov'>Regression floors: Δ &ge; {d['abs_floor']:.2f} absolute "
        f"<b>and</b> &ge; {d['rel_floor']:.0%} relative. "
        f"Baseline {d['baseline_images']} images, current {d['current_images']} images.</p>"
        "<table><tr><th>Condition</th><th>Baseline mAP</th><th>Current mAP</th>"
        "<th>Δ</th><th>Δ%</th><th>Status</th></tr>"
        f"{rows}</table>"
        f"{cov}"
        "<p class='note'>Floors are configurable; a condition gates only when it clears "
        "both. Conditions present in only one run are reported but never gate. "
        "Reproducible: fixed seed, deterministic torch, single-thread BLAS.</p>"
        "</div></body></html>"
    )
