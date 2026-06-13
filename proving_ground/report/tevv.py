"""TEVV assurance-case report.

The HTML report (`report/html.py`) shows *numbers*; this turns them into an
*assurance judgement* against stated acceptance criteria — the artifact a
certifying authority can act on. It maps a single-model `bench` result onto the
TEVV frame (Test conditions / Evaluation / Verification & Validation) and emits a
PASS / CONDITIONAL / FAIL verdict.

Acceptance criteria (overridable):
  * Baseline competence  — clean mAP >= ``min_clean_map``.
  * Per-condition robustness — under each attack/DVE condition the model must
    retain >= ``min_retained`` of its clean mAP.

Verdict:
  * FAIL        — baseline competence not met (the model is unfit to assess).
  * PASS        — baseline met and every condition meets the robustness floor.
  * CONDITIONAL — baseline met but one or more conditions fall below the floor
                  (fielding allowed only with mitigations for those conditions).
"""

from __future__ import annotations

import html


def assess(results: dict, min_clean_map: float = 0.30, min_retained: float = 0.50) -> dict:
    """Judge a single-model bench result against acceptance criteria."""
    clean = results["clean_map"]
    baseline_pass = clean >= min_clean_map

    conditions = []
    for a in results.get("attacks", []):
        retained = a["attacked_map"] / clean if clean > 0 else 0.0
        conditions.append({
            "name": a["name"],
            "attacked_map": a["attacked_map"],
            "retained": retained,
            "passed": retained >= min_retained,
        })
    conditions.sort(key=lambda c: c["retained"])  # weakest first
    failing = [c["name"] for c in conditions if not c["passed"]]

    if not baseline_pass:
        verdict = "FAIL"
        rationale = (f"Baseline competence not met: clean mAP {clean:.3f} "
                     f"< required {min_clean_map:.2f}.")
    elif not failing:
        verdict = "PASS"
        rationale = (f"Baseline met (clean mAP {clean:.3f}) and all "
                     f"{len(conditions)} conditions retain >= {min_retained:.0%}.")
    else:
        verdict = "CONDITIONAL"
        rationale = (f"Baseline met, but {len(failing)} of {len(conditions)} conditions "
                     f"fall below the {min_retained:.0%} retention floor: "
                     f"{', '.join(failing)}.")

    return {
        "min_clean_map": min_clean_map, "min_retained": min_retained,
        "clean_map": clean, "num_images": results.get("num_images"),
        "iou_threshold": results.get("iou_threshold", 0.5),
        "baseline_pass": baseline_pass, "conditions": conditions,
        "n_pass": sum(c["passed"] for c in conditions), "failing": failing,
        "verdict": verdict, "rationale": rationale,
    }


_VERDICT_COLOR = {"PASS": "#1a7f37", "CONDITIONAL": "#9a6700", "FAIL": "#b42318"}

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
.claim{font-size:15px;background:#f0f4f8;border-left:4px solid #5d6b78;
padding:10px 14px;border-radius:4px;margin:6px 0;}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:6px 0;}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #eaeef2;}
th{color:#5d6b78;font-size:11.5px;text-transform:uppercase;letter-spacing:.03em;}
td.num{text-align:right;font-variant-numeric:tabular-nums;}
.pass{color:#1a7f37;font-weight:700;} .fail{color:#b42318;font-weight:700;}
.prov{font-size:12px;color:#5d6b78;margin-top:6px;}
.note{font-size:12.5px;color:#5d6b78;border-top:1px solid #eaeef2;padding-top:12px;margin-top:18px;}
"""


def render_tevv_html(assessment: dict, model: str = "model under test",
                     title: str = "TEVV Robustness Assurance Case") -> str:
    a = assessment
    color = _VERDICT_COLOR[a["verdict"]]
    rows = ""
    for c in a["conditions"]:
        badge = ("<span class='pass'>PASS</span>" if c["passed"]
                 else "<span class='fail'>FAIL</span>")
        rows += (f"<tr><td>{html.escape(c['name'])}</td>"
                 f"<td class='num'>{c['attacked_map']:.3f}</td>"
                 f"<td class='num'>{c['retained'] * 100:.0f}%</td><td>{badge}</td></tr>")
    claim = (f"<b>Claim.</b> Under the tested threat-representative and degraded-visual-"
             f"environment conditions, <b>{html.escape(model)}</b> retains at least "
             f"{a['min_retained']:.0%} of its clean detection performance "
             f"(clean mAP@{a['iou_threshold']} &ge; {a['min_clean_map']:.2f}).")
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{_CSS}</style></head><body><div class='card'>"
        f"<h1>{html.escape(title)}</h1>"
        f"<div class='verdict' style='background:{color}'>VERDICT: {a['verdict']}</div>"
        f"<p class='prov'>{html.escape(a['rationale'])}</p>"
        f"<div class='claim'>{claim}</div>"
        f"<h2>Test conditions (T)</h2><p class='prov'>{len(a['conditions'])} adversarial "
        f"and degraded-visual-environment conditions applied to {a['num_images']} images.</p>"
        f"<h2>Evaluation &amp; Verification (E/V)</h2>"
        f"<p class='prov'>Clean baseline mAP@{a['iou_threshold']} = "
        f"<b>{a['clean_map']:.3f}</b> (required &ge; {a['min_clean_map']:.2f} — "
        f"{'met' if a['baseline_pass'] else 'NOT met'}). Per-condition retention vs the "
        f"{a['min_retained']:.0%} floor:</p>"
        "<table><tr><th>Condition</th><th>Attacked mAP</th><th>Retained</th><th>Verify</th></tr>"
        f"{rows}</table>"
        "<h2>Validation (V)</h2><p class='prov'>Evidence is reproducible: fixed seed, "
        "deterministic torch + single-thread BLAS, and results locked against committed "
        "golden baselines. Re-runnable end-to-end via the CLI.</p>"
        "<p class='note'>Acceptance criteria are configurable; this artifact records the "
        "judgement, not policy. CONDITIONAL = fieldable only with mitigations for the failing "
        "conditions.</p>"
        "</div></body></html>"
    )
