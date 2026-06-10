# Proving Ground AI — Progress

_Snapshot: 2026-06-09. Reviewed against `project-spec.docx`._

## 1. Product Vision

Independent assurance layer for physical military AI: stress-tests perception /
autonomy models, measures where they break, emits a TEVV-aligned report a
certifying authority can act on. Four core modules: benchmark engine, attack
suite, evaluation layer, assurance reporting. Entry wedge: drone / satellite
object detection under adversarial attack and degraded visual conditions.
Positioning: **independent**, **physical-AI**, **threat-representative**,
**continuous**, **cross-vendor**.

## 2. What's Done

`proving-ground` v0.0.1 is a clean prototype covering the camera-only slice of
the entry wedge.

**Adapters** — `Detector` protocol and optional `WhiteBox` sub-protocol with
fixed xyxy / [0,1]-tensor conventions; concrete `FakeDetector` (weight-free,
also white-box) and `UltralyticsYOLOAdapter` (real YOLOv8 with differentiable
`compute_loss`).

**Attacks**
- FGSM — single-step sign-gradient with `L_inf <= eps` bound.
- PGD-Linf — multi-step FGSM with L_inf projection after each step; strictly
  stronger than FGSM at the same `eps` budget. Optional uniform `random_init`
  inside the eps-ball (seeded for reproducibility).
- PGD-L2 — L2-ball mirror of PGD-Linf: unit-norm gradient step followed by L2
  projection of the cumulative perturbation. `eps` bounds the L2 norm of the
  whole perturbation tensor (literature default 3.0). Optional `random_init`
  samples uniformly inside the L2 ball.
- Optimized patch — PGD-style ascent with byte-exact containment outside the
  patch rectangle.
- EOT patch — gradient averaged over sampled scale/rotation/translation/lighting
  transforms; survives held-out renderings.
- DVE degradation — six black-box modes (gaussian blur, motion blur, gaussian
  noise, fog, low light, JPEG).

**Evaluation & reporting**
- IoU, per-class AP (Pascal-VOC all-points), pooled mAP, clean-vs-attacked
  deltas — hand-validated in `test_metrics.py`.
- Versioned report schema (`SCHEMA_VERSION = "0.1.0"`) with deterministic JSON
  writer; `meta.timestamp_utc` is the only non-reproducible field.

**CLI & benchmark** — `proving-ground run` orchestrates one attack end-to-end;
`proving-ground bench` runs the canonical suite pooled across an image set —
five white-box attacks (FGSM, PGD-Linf, PGD-L2, patch, EOT-patch) plus all six
DVE modes at severity 0.8, so one invocation produces the full README headline;
`seeding.set_seed()` pins all RNGs and torch deterministic algorithms.

**Fixtures & locked baselines**
- 2 synthetic PNGs for smoke tests; 1-image CC0 anchor (`coco_sample`); 3-image
  realistic CC0 set (`coco_scenes`).
- Seven golden JSONs in `tests/baselines/`: FGSM, PGD-Linf, PGD-L2, patch,
  EOT-patch single-image anchors plus aggregate benchmark and DVE severity grid.

**Tests** — fast tier (no weights) covers adapter contract, every attack,
metrics, report, viz, CLI smoke. Integration tier (`-m integration`) loads real
YOLO and asserts byte-identical reproducibility, drift < 1e-4 vs baselines,
attacks degrade mAP, and EOT patches survive held-out transforms.

**Demo & packaging** — `tools/make_figures.py` renders six clean-vs-attacked
PNGs; `pyproject.toml` (Python 3.11+, Apache-2.0, ruff/mypy/pytest), README
result tables, `CONTRIBUTING.md`, `NOTICE` with AGPL warning for ultralytics.

**Headline locked numbers** — clean mAP@0.5 = 0.39 on `coco_scenes`; FGSM →
0.07, PGD-Linf → 0.00, PGD-L2 → 0.00, patch → 0.09, EOT patch → 0.17; full DVE
severity grid locked.

## 3. In Progress

`main` is clean. "In progress" only in the sense of being a first slice of a
larger spec capability:

- **Threat-representative library (cap. #1)** — generic FGSM, PGD-Linf, PGD-L2,
  patch, and EOT exist; C&W, physical camo, decoys, EO/IR/SAR/GPS spoofing not
  started.
- **DVE simulator (cap. #2)** — six modes shipped; spec also wants smoke, dust,
  realistic night/IR.
- **Assurance reporting (module #4)** — JSON exists but is not TEVV-mapped.

## 4. Next Steps

**Near-term (close entry-wedge gaps)**
1. Add a CC0/CC-BY drone / satellite fixture set with hand annotations so the
   wedge is visible in the result table _(module #1)_.
2. Add **C&W** (binary-search-over-trade-off) to round out the textbook
   white-box trio _(cap. #1)_.
3. Add the missing DVE modes — smoke, dust, sensor-realistic night/IR
   _(cap. #2)_.
4. Replace single-seed point estimates with a seed sweep / bootstrap CI in the
   report _(module #3)_.
5. Promote the DVE severity sweep into a first-class `bench` mode (currently
   hard-coded to severity 0.8) so the full severity grid is reproducible
   without per-mode `run` invocations _(module #1)_.
6. Regenerate the patch / EOT-patch locked baselines on the current PyTorch
   version — drift is now manifest (`test_patch_snapshot` and
   `test_benchmark_snapshot` fail at the 1e-4 tolerance on the current
   PyTorch 2.12 venv); a deliberate re-lock is needed to restore the green
   integration tier _(engineering risk)_.

**Mid-term (differentiating capabilities)**
7. Physical patch / camouflage over multi-view scenes and printability
   constraints _(cap. #1)_.
8. Adapter protocols and degradations for IR, SAR, LiDAR, plus a sensor-fusion
   adapter _(cap. #2)_.
9. Domain-shift / sim-to-real distance metrics with a threshold report
   _(cap. #3)_.
10. Continuous-monitoring agent that re-runs the suite on model updates and
    diffs against the prior report — the recurring-revenue hook _(cap. #4)_.
11. Cross-vendor stack composition (detector → tracker → planner) to surface
    seam failures _(cap. #5)_.
12. TEVV assurance-case generator: render the report as a TEVV-mapped PDF/HTML
    artifact _(cap. #6)_.
13. OOD detection and confidence-calibration checks in `eval/` _(module #3)_.
14. Failure-mode taxonomy (occlusion / small-object / domain-shift /
    adversarial) tagged per missed detection _(module #3)_.

**Later / harder**
15. Air-gapped packaging — pin weights and dependencies, no network calls
    _(cap. #7)_.
16. Test-range / HIL / digital-twin frame-stream adapters _(cap. #8)_.
17. Saliency / attribution overlays and calibrated-confidence reporting
    _(cap. #9)_.
18. Ship a permissively-licensed default detector to remove the AGPL taint
    _(procurement blocker)_.

## 5. Gaps and Risks

**Scope gaps**
- RGB-only modality — no IR / SAR / LiDAR / fusion (cap. #2 unstarted).
- No defense-relevant data — every fixture is a street scene, not overhead
  imagery.
- Attack library is "generic textbook + EOT" (FGSM, PGD-Linf, patch, EOT-patch),
  not the threat-representative depth the spec promises as its #1 differentiator.
- One-shot CLI only — no continuous-monitoring agent, no model-version diff.
- Report is JSON, not a TEVV-mapped assurance artifact.
- OOD detection, confidence calibration, and failure-mode taxonomy are absent
  from `eval/`.
- Single detector per run — cross-vendor stack composition unsupported.
- `UltralyticsYOLOAdapter` downloads weights on first use, so air-gapped
  deployment is impossible today.

**Engineering risks**
- AGPL exposure: the default backend taints any combined deployment, blocking
  paid pilots without a swap.
- Single-seed point estimates with no confidence intervals — easy to challenge
  externally.
- Fixture sets (1 + 3 images) are too small to call a "benchmark" publicly.
- No auth, sandboxing, or report signing on the CLI — blockers for the
  certifying-authority use case.
- No CI committed; two-tier test discipline lives only in `CONTRIBUTING.md`.
- Snapshot byte-identity holds on CPU but will be fragile across GPU/driver
  upgrades.

**Strategic positioning**
- Of the spec's five positioning properties, only **independent** and
  **physical-AI** are delivered; threat-representative, continuous, and
  cross-vendor are still aspirational.
- The engine works for any object detector but does not yet demonstrate the
  stated entry wedge (drone / satellite); a defense-relevant fixture plus one
  overhead-imagery-specific attack is the cheapest way to make it visible.
