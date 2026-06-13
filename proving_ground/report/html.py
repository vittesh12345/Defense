"""Render benchmark / scorecard JSON into a self-contained HTML report.

The numbers a client cares about shouldn't arrive as raw JSON. This turns a
`bench` results dict (single-model robustness) or a `compare` scorecard into a
one-page, inline-styled HTML report — color-coded by how much each attack
degrades detection, with the methodology caveat shown for cross-model scorecards.
"""

from __future__ import annotations

import html

_CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
color:#1a2330;background:#f6f8fa;margin:0;padding:32px;}
.card{max-width:860px;margin:0 auto;background:#fff;border:1px solid #d8dee4;
border-radius:10px;padding:28px 32px;box-shadow:0 1px 3px rgba(0,0,0,.06);}
h1{font-size:20px;margin:0 0 2px;} .sub{color:#5d6b78;font-size:13px;margin:0 0 18px;}
table{border-collapse:collapse;width:100%;font-size:14px;margin:6px 0 18px;}
th,td{text-align:left;padding:8px 10px;border-bottom:1px solid #eaeef2;}
th{color:#5d6b78;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.03em;}
td.num{font-variant-numeric:tabular-nums;text-align:right;}
.bar{height:10px;border-radius:5px;}
.note{font-size:12.5px;color:#5d6b78;border-top:1px solid #eaeef2;padding-top:14px;}
.warn{background:#fff7e6;border:1px solid #ffd591;color:#7a4f01;border-radius:8px;
padding:10px 14px;font-size:13px;margin:0 0 18px;}
.grade{font-weight:700;}
"""


def _color(retained: float) -> str:
    """Red (0% retained) -> amber -> green (100% retained)."""
    retained = max(0.0, min(1.0, retained))
    if retained < 0.5:
        r, g = 211, int(120 * (retained / 0.5))
    else:
        r, g = int(211 * (1 - (retained - 0.5) / 0.5)), 150
    return f"rgb({r},{g + 60},60)"


def _bar(frac: float, color: str) -> str:
    pct = max(2.0, min(100.0, frac * 100))
    return f'<div class="bar" style="width:{pct:.0f}%;background:{color}"></div>'


def render_html(results: dict, title: str = "Robustness Assurance Report") -> str:
    if results.get("kind") == "ensemble":
        body = _render_ensemble(results)
    elif "models" in results:
        body = _render_scorecard(results)
    else:
        body = _render_bench(results)
    return (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head>"
        f"<body><div class='card'><h1>{html.escape(title)}</h1>{body}</div></body></html>"
    )


def _render_bench(r: dict) -> str:
    clean = r["clean_map"]
    attacks = r.get("attacks", [])
    worst = min(attacks, key=lambda a: a["attacked_map"]) if attacks else None
    sub = (f"Pooled mAP@{r.get('iou_threshold', 0.5)} over {r.get('num_images', '?')} "
           f"images &middot; clean mAP {clean:.3f}")
    if worst:
        sub += f" &middot; weakest against <b>{html.escape(worst['name'])}</b>"
    rows = ""
    for a in sorted(attacks, key=lambda a: a["attacked_map"]):
        retained = a["attacked_map"] / clean if clean > 0 else 0.0
        col = _color(retained)
        rows += (
            f"<tr><td>{html.escape(a['name'])}</td>"
            f"<td class='num'>{clean:.3f}</td>"
            f"<td class='num'>{a['attacked_map']:.3f}</td>"
            f"<td class='num'>&minus;{a['map_delta']:.3f}</td>"
            f"<td class='num'>{retained * 100:.0f}%</td>"
            f"<td style='width:160px'>{_bar(retained, col)}</td></tr>"
        )
    return (
        f"<p class='sub'>{sub}</p>"
        "<table><tr><th>Attack</th><th>Clean</th><th>Attacked</th><th>Drop</th>"
        "<th>Retained</th><th>Robustness</th></tr>"
        f"{rows}</table>"
        "<p class='note'>Lower bars = more detection destroyed. Numbers are pooled "
        "mAP@0.5, reproducible against locked baselines.</p>"
    )


def _render_ensemble(r: dict) -> str:
    clean = r["clean_map"]
    ens = r["ensemble_map"]
    retained = ens / clean if clean > 0 else 0.0
    sub = (f"Worst-case-per-image over {len(r['attacks'])} attacks &middot; "
           f"{r.get('num_images', '?')} images &middot; mAP@{r.get('iou_threshold', 0.5)}")
    headline = (
        f"<p class='sub'>{sub}</p>"
        f"<table><tr><th>Metric</th><th>mAP</th></tr>"
        f"<tr><td>Clean</td><td class='num'>{clean:.3f}</td></tr>"
        f"<tr><td><b>Worst-case ensemble</b> (lower bound)</td>"
        f"<td class='num'><b>{ens:.3f}</b></td></tr>"
        f"<tr><td>Strongest single attack "
        f"({html.escape(r['strongest_single_attack'])})</td>"
        f"<td class='num'>{r['strongest_single_map']:.3f}</td></tr></table>"
        f"<p class='note'>The ensemble is {r['ensemble_gain_over_single']:.3f} mAP "
        f"tighter than the strongest single attack &middot; {retained * 100:.0f}% of "
        f"clean performance retained against a method-choosing adversary.</p>"
    )
    rows = ""
    for name in sorted(r["attacks"], key=lambda n: r["single_attack_maps"][n]):
        m = r["single_attack_maps"][name]
        col = _color(m / clean if clean > 0 else 0.0)
        rows += (
            f"<tr><td>{html.escape(name)}</td>"
            f"<td class='num'>{m:.3f}</td>"
            f"<td class='num'>{r['win_counts'][name]}/{r['num_images']}</td>"
            f"<td style='width:160px'>{_bar(m / clean if clean > 0 else 0.0, col)}</td></tr>"
        )
    return (
        headline
        + "<table><tr><th>Attack</th><th>Pooled mAP</th><th>Images won</th>"
        "<th>Robustness</th></tr>"
        + rows
        + "</table><p class='note'>\"Images won\" = how often each attack was the "
        "worst for an image. A spread across attacks means no single method dominates "
        "&mdash; the ensemble is doing real work.</p>"
    )


def _render_scorecard(r: dict) -> str:
    rows = ""
    for i, m in enumerate(r["models"], 1):
        retained = m["robustness_retained"]
        rows += (
            f"<tr><td>{i}</td><td>{html.escape(m['model'])}</td>"
            f"<td class='num'>{m['clean_map']:.3f}</td>"
            f"<td class='num'>{m['mean_attacked_map']:.3f}</td>"
            f"<td class='num grade'>{retained * 100:.0f}%</td>"
            f"<td>{html.escape(str(m['weakest_attack']))}</td>"
            f"<td style='width:140px'>{_bar(retained, _color(retained))}</td></tr>"
        )
    return (
        f"<p class='sub'>{r.get('num_attacks', '?')} attacks over "
        f"{r.get('num_images', '?')} images &middot; ranked by performance retained</p>"
        "<div class='warn'><b>Methodology note:</b> a credible cross-model ranking "
        "needs <i>complete</i> ground truth. On salient-annotated fixtures, higher-recall "
        "models are penalised for detecting unlabelled objects — run on an exhaustively "
        "labelled benchmark (e.g. COCO val) for a trustworthy ranking.</div>"
        "<table><tr><th>Rank</th><th>Model</th><th>Clean</th><th>Mean attacked</th>"
        "<th>Retained</th><th>Weakest vs</th><th></th></tr>"
        f"{rows}</table>"
    )
