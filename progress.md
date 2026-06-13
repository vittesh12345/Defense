# Proving Ground AI — Progress

_Snapshot: 2026-06-09. Reviewed against `project-spec.docx`._

## Changelog

Reverse-chronological log of changes we make; trim oldest entries to keep this
file under 250 lines.

- **2026-06-13** — Added the TEVV assurance report (`proving_ground/report/tevv.py`,
  `cli tevv --in results.json --out assurance.html`): turns a `bench` result into a
  PASS/CONDITIONAL/FAIL **verdict** against acceptance criteria — baseline competence
  (clean mAP ≥ `--min-clean`) + per-condition robustness (retained ≥ `--min-retained`)
  — rendered as a certifying-authority assurance case (Claim → Test conditions →
  Evaluation/Verification table → Validation/provenance), conditions sorted
  weakest-first. CONDITIONAL = fieldable only with mitigations for the failing
  conditions. Criteria are configurable; the artifact records the judgement, not the
  policy. Fast-tested (verdict logic on synthetic dicts + HTML escaping, no weights).
- **2026-06-13** — Solidified the COCO demo: added `tools/fetch_coco_val.py`
  (downloads a deterministic COCO val subset — first N by id — so the scorecard is
  reproducible; images stay external). Ran a 50-image credible scorecard:
  yolov8m 64% retained > s 60% > n 52%, all weakest vs pgd-linf — sane ranking on
  complete labels (bigger = more accurate AND more robust). Re-locked the two
  cross-machine-drifted iterative baselines on this machine (patch 0.125→0.167 by
  one detection; pgd-linf map unchanged, only incidental FP classes) to restore a
  green integration tier. NOTE: iterative-attack snapshot baselines are
  machine-specific — the BLAS pin gives cross-*session* determinism, not
  cross-*machine*; re-lock when the canonical machine changes.
- **2026-06-13** — Added a COCO instances-JSON loader (`proving_ground/data/coco.py`)
  + `--coco`/`--limit` on `bench` and `compare` — the credible-scorecard path:
  COCO val2017 is exhaustively annotated, so cross-model comparison is no longer
  biased against higher-recall models. Maps COCO categories to the detector's
  classes by name (robust to gappy ids), `xywh→xyxy`, skips crowd regions.
  Fast-tested on a synthetic COCO file; COCO images aren't committed (mixed
  licenses) — point it at a local download. Resolves the comparison caveat.
- **2026-06-13** — Added video support (`proving_ground/video.py`, `cli video
  --video clip.mp4 --frames N --mode <dve>`): samples frames, applies a black-box
  degradation per frame, reports a GT-free detection-stability metric
  (attacked/clean detections) for unlabelled footage. On a CC-BY city-drone clip,
  fog cut yolov8n's detections to 23%, motion-blur to 38%. Fast-tested (synthetic
  frames + written-clip round-trip, no weights). Viewpoint caveat documented.
- **2026-06-13** — Added a client-readable HTML report (`proving_ground/report/html.py`,
  `cli report --in results.json --out report.html`): renders a `bench` (single-model
  robustness) or `compare` (scorecard, with the incomplete-GT caveat banner) JSON into
  a self-contained, color-coded one-pager. Fast-tested (no weights).
- **2026-06-13** — Added multi-model comparison (`proving_ground/compare.py`,
  `cli compare --models a.pt,b.pt`): runs the attack suite per model, ranks by
  performance retained under attack. **Key finding:** cross-model mAP comparison
  is confounded by incomplete (salient) GT — higher-recall models (yolov8s/m)
  detect more unlabelled objects → counted as FPs → deflated mAP, so yolov8n
  spuriously tops clean mAP. The machinery is shipped + fast-tested; a *credible*
  scorecard needs an exhaustively-annotated benchmark (COCO val subset). Within a
  single model, clean-vs-attacked degradation is unaffected and stays valid.
- **2026-06-13** — Expanded `coco_scenes` from 3 to 6 images (added `cyclist`,
  `cafe_window`, `bus_stop`; +bicycle/+bus classes), hand-annotated; merged on top
  of the smoke/dust + BLAS-pin work and re-locked all three coco_scenes baselines
  over the combined 6-image × 8-mode state (clean mAP 0.39 → 0.29). Honest note:
  bootstrap CIs stayed wide (clean [0.091, 0.480]) — the new scenes are
  heterogeneous (cyclist 1.0 vs cafe/bus ~0.12), so the CI reflects real
  scene-to-scene variance, not just small-n.
- **2026-06-12** — Added two DVE modes (`smoke`, `dust`) — `_smoke` blends image
  against a dark-grey veil with a billowing per-pixel opacity field (battlefield
  obscurants); `_dust` blends against a warm-tan veil with fine grain (brownout
  / sandstorm). Both seeded-stochastic, both auto-extend the canonical bench
  and fast-tier snapshot grid via `MODES`. Also **pinned BLAS to a single
  thread in `seeding.set_seed()`** (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
  `torch.set_num_threads(1)`) — multi-threaded reductions are non-associative
  in float and their scheduling drifts across Python processes, which is what
  was actually causing the recurring "PyTorch X.YY drifted PGD baselines" pain.
  Re-locked `coco_scenes_benchmark.json` and `coco_scenes_benchmark_ci.json`;
  verified inter-session byte-identity from two fresh Python processes. DVE +
  FGSM rows byte-identical to the prior lock; iterative WB rows shifted to the
  new pinned values (one-time re-lock to gain inter-session determinism).
  Updated README tables.
- **2026-06-12** — Added GitHub Actions CI (`.github/workflows/ci.yml`): ruff +
  fast tier (weight-free) on push/PR; integration tier stays local (not portable
  across CI architectures). README CI badge. Addresses the "no CI committed" risk.
- **2026-06-12** — Added confidence intervals (`proving_ground/stats.py`,
  `bench --seeds K`): a **seed sweep** (attack-randomness CI; deterministic
  attacks show zero width, EOT/noise/low-light get real intervals) and an
  **image bootstrap** (1000 resamples; error bars for every attack, surfacing
  the wide small-fixture uncertainty — clean 0.39, bootstrap 95% CI
  [0.125, 0.460]). Locked golden + integration/fast tests + README table. The
  single-seed `bench` and its baseline are unchanged.
- **2026-06-12** — Re-locked the two drifted baselines on PyTorch 2.12
  (`coco_sample_pgd_linf_report.json`, `coco_scenes_benchmark.json` — benign PGD
  float drift: near-zero `attacked_map` and which incidental FP classes appear).
  `pytest -m integration` is **green again** (31 passed). Updated Claude Code CLI
  2.1.169 -> 2.1.174.
- **2026-06-11** — Added `UltralyticsOBBAdapter` (black-box; predict only) so the
  engine can run aerial-trained DOTA-OBB models (oriented boxes -> axis-aligned
  `xyxy`); CLI routes `--model *obb*` to it; unit + integration tests. **Aerial
  benchmark shelved** — see the aerial finding below.
- **2026-06-11** — Added DVE demo figures (low-light, fog, and a combined
  fog-vs-low-light composite) and referenced them in the README; added the
  README DVE drop table (severity 0.8) and the full severity-sweep grid;
  installed the `karpathy-guidelines` project skill; pulled the PGD-Linf /
  PGD-L2 white-box suite and the full-canonical `bench` into `main`.

### Finding: aerial / overhead wedge is detector- and metric-blocked

Attempted the drone/satellite wedge and hit hard limits, evidenced:
- COCO-pretrained `yolov8n` is out-of-distribution for overhead views — it
  localizes aerial cars (IoU up to 0.86) but **mislabels them as `boat`**, so
  car-mAP@0.5 = 0 (nothing for attacks to degrade).
- DOTA-trained `yolov8n-obb` detects aerial vehicles but **only on true nadir**
  imagery (71 `small vehicle` on a nadir lot; 0 on oblique drone shots).
- Freely-sourceable CC0/CC-BY nadir imagery is **dense small-vehicle lots**;
  large IoU-stable DOTA classes (ship/plane/large-vehicle) only appear in
  *oblique* Commons aerials, where DOTA detects nothing.
- On a nadir lot, mAP@0.5 on ~30-45 px cars is **too fragile** (box error drops
  IoU < 0.5) — clean mAP ~0.06 and degradations bounce *above* it. Not credible.

Shipped the working OBB adapter; **deferred**: white-box OBB `compute_loss` (for
FGSM/PGD/patch/EOT on aerial), a credible nadir large-object fixture (needs a
licensed DOTA/xView sample or lower IoU threshold), and the aerial benchmark.

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
- DVE degradation — eight black-box modes (gaussian blur, motion blur, gaussian
  noise, fog, low light, JPEG, smoke, dust).

**Evaluation & reporting**
- IoU, per-class AP (Pascal-VOC all-points), pooled mAP, clean-vs-attacked
  deltas — hand-validated in `test_metrics.py`.
- Versioned report schema (`SCHEMA_VERSION = "0.1.0"`) with deterministic JSON
  writer; `meta.timestamp_utc` is the only non-reproducible field.

**CLI & benchmark** — `proving-ground run` orchestrates one attack end-to-end;
`proving-ground bench` runs the canonical suite pooled across an image set —
five white-box attacks (FGSM, PGD-Linf, PGD-L2, patch, EOT-patch) plus all eight
DVE modes at severity 0.8, so one invocation produces the full README headline;
`seeding.set_seed()` pins all RNGs, enables torch deterministic algorithms, and
pins BLAS to one thread for inter-session byte-identity on iterative attacks.

**Fixtures & locked baselines**
- 2 synthetic PNGs for smoke tests; 1-image CC0 anchor (`coco_sample`); 6-image
  realistic CC0 set (`coco_scenes`).
- Golden JSONs in `tests/baselines/`: FGSM, PGD-Linf, PGD-L2, patch, EOT-patch
  single-image anchors plus the aggregate benchmark, DVE severity grid, and the
  seed-sweep / image-bootstrap confidence intervals.

**Tests** — fast tier (no weights) covers adapter contract, every attack,
metrics, report, viz, CLI smoke. Integration tier (`-m integration`) loads real
YOLO and asserts byte-identical reproducibility, drift < 1e-4 vs baselines,
attacks degrade mAP, and EOT patches survive held-out transforms.

**Demo & packaging** — `tools/make_figures.py` renders six clean-vs-attacked
PNGs; `pyproject.toml` (Python 3.11+, Apache-2.0, ruff/mypy/pytest), README
result tables, `CONTRIBUTING.md`, `NOTICE` with AGPL warning for ultralytics.

**Headline locked numbers** — clean mAP@0.5 = 0.29 on `coco_scenes` (6 images);
FGSM → 0.07, PGD-Linf → 0.01, PGD-L2 → 0.03, patch → 0.07, EOT patch → 0.10; full
DVE severity grid locked (eight modes × three severities) + seed/bootstrap CIs.

## 3. In Progress

`main` is clean. "In progress" only in the sense of being a first slice of a
larger spec capability:

- **Threat-representative library (cap. #1)** — generic FGSM, PGD-Linf, PGD-L2,
  patch, and EOT exist; C&W, physical camo, decoys, EO/IR/SAR/GPS spoofing not
  started.
- **DVE simulator (cap. #2)** — eight modes shipped (incl. smoke + dust); spec
  also wants realistic night/IR.
- **Assurance reporting (module #4)** — JSON exists but is not TEVV-mapped.

## 4. Next Steps

**Near-term (close entry-wedge gaps)**
1. Add a CC0/CC-BY drone / satellite fixture set with hand annotations so the
   wedge is visible in the result table _(module #1)_.
2. Add **C&W** (binary-search-over-trade-off) to round out the textbook
   white-box trio _(cap. #1)_.
3. Add sensor-realistic night/IR DVE mode _(cap. #2; smoke + dust shipped
   2026-06-12)_.
4. Replace single-seed point estimates with a seed sweep / bootstrap CI in the
   report _(module #3)_.
5. Promote the DVE severity sweep into a first-class `bench` mode (currently
   hard-coded to severity 0.8) so the full severity grid is reproducible
   without per-mode `run` invocations _(module #1)_.
6. ~~Re-lock baselines drifting on PyTorch 2.12.~~ **Done (2026-06-12):**
   re-locked `coco_sample_pgd_linf_report.json` and `coco_scenes_benchmark.json`;
   `pytest -m integration` is green (31 passed). Snapshot byte-identity is still
   CPU-only and will need re-locking across future torch/driver upgrades
   _(engineering risk)_.

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
- ~~Single-seed point estimates with no confidence intervals.~~ Addressed for
  the benchmark via `bench --seeds K` (seed sweep + image bootstrap); the wide
  bootstrap CIs now make the small-fixture uncertainty explicit rather than hidden.
- Fixture sets (1 + 3 images) are too small to call a "benchmark" publicly.
- No auth, sandboxing, or report signing on the CLI — blockers for the
  certifying-authority use case.
- ~~No CI committed.~~ GitHub Actions runs ruff + the fast tier on push/PR; the
  integration tier still runs locally only (baselines aren't CI-portable).
- Snapshot byte-identity holds on CPU (single-threaded BLAS pinned in
  `set_seed()` to keep iterative attacks deterministic across Python processes)
  but will be fragile across GPU/driver upgrades, and re-locking has runtime
  cost on cores > 1 since torch is held to one thread.

**Strategic positioning**
- Of the spec's five positioning properties, only **independent** and
  **physical-AI** are delivered; threat-representative, continuous, and
  cross-vendor are still aspirational.
- The engine works for any object detector but does not yet demonstrate the
  stated entry wedge (drone / satellite); a defense-relevant fixture plus one
  overhead-imagery-specific attack is the cheapest way to make it visible.
