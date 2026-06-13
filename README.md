# Proving Ground AI — Robustness Testing Engine

[![CI](https://github.com/vittesh12345/Defense/actions/workflows/ci.yml/badge.svg)](https://github.com/vittesh12345/Defense/actions/workflows/ci.yml)

Stress-tests object-detection models (YOLO, etc.) with adversarial attacks and
degradations, measures the failure, and emits a reproducible JSON assurance
report. The output is **measurement** — correctness and reproducibility beat
speed.

It ships several attacks — white-box **FGSM** (single-step generic gradient),
**PGD-Linf** (the standard iterative FGSM at the same L-inf budget — strictly
stronger), **PGD-L2** (the L2-ball mirror of PGD-Linf — eps budgets the L2 norm
of the whole perturbation, often a more realistic similarity metric than L-inf),
an **optimized patch**, and an **EOT patch** (Expectation Over Transformation:
a localized patch trained to survive being printed and re-photographed at
different scales, angles, and lighting), plus a black-box **degradation**
family (DVE: gaussian/motion blur, gaussian noise, fog, low light, JPEG
compression, smoke, dust) that simulates real sensor and weather conditions —
including battlefield obscurants and brownout.

## Results

Pooled mAP@0.5 of `yolov8n` over a small set of cluttered, hand-annotated CC0
street scenes (`proving_ground/data/fixtures/coco_scenes/`, clean = **0.39** — a
believable baseline, not a toy 1.0). Every attack measurably degrades detection.

**White-box attacks:**

| Attack | Clean mAP | Attacked mAP | Δ (drop) |
|---|---|---|---|
| fgsm | 0.39 | 0.07 | 0.32 |
| pgd-linf | 0.39 | 0.00 | 0.39 |
| pgd-l2 | 0.39 | 0.00 | 0.39 |
| patch | 0.39 | 0.06 | 0.33 |
| eot-patch | 0.39 | 0.13 | 0.26 |

**Black-box degradation (DVE), at severity 0.8** — simulated sensor/weather
conditions:

| Degradation | Clean mAP | Attacked mAP | Δ (drop) |
|---|---|---|---|
| motion_blur | 0.39 | 0.00 | 0.39 |
| gaussian_noise | 0.39 | 0.02 | 0.37 |
| low_light | 0.39 | 0.12 | 0.27 |
| dust | 0.39 | 0.17 | 0.22 |
| jpeg_compression | 0.39 | 0.19 | 0.20 |
| smoke | 0.39 | 0.22 | 0.17 |
| gaussian_blur | 0.39 | 0.27 | 0.12 |
| fog | 0.39 | 0.28 | 0.11 |

Full severity sweep (attacked mAP@0.5 by severity; higher severity ⇒ more
degradation, monotonic-ish — mild blur can even raise mAP slightly by suppressing
spurious detections). Locked in `tests/baselines/coco_scenes_degradation.json`:

| Degradation mode | sev 0.25 | sev 0.50 | sev 0.80 |
|---|---|---|---|
| motion_blur | 0.10 | 0.00 | 0.00 |
| gaussian_noise | 0.33 | 0.07 | 0.02 |
| low_light | 0.39 | 0.24 | 0.12 |
| dust | 0.37 | 0.17 | 0.17 |
| jpeg_compression | 0.39 | 0.31 | 0.19 |
| smoke | 0.25 | 0.21 | 0.22 |
| gaussian_blur | 0.42 | 0.42 | 0.27 |
| fog | 0.37 | 0.36 | 0.28 |

Reproduce a single mode at a chosen severity (used for the severity sweep
above), e.g.:

```bash
.venv/bin/python -m proving_ground.cli run \
  --images proving_ground/data/fixtures/coco_scenes/images \
  --ann   proving_ground/data/fixtures/coco_scenes/annotations.json \
  --model yolov8n.pt --attack degradation --mode fog --severity 0.8 --out fog.json
```

The EOT patch drops less *at its clean placement* because it trades peak damage
for **robustness under transformation** — it keeps degrading detection when the
patch is re-rendered at scales/rotations held out from training (see
`tests/test_eot_patch_snapshot.py`). The headline white-box + DVE drop tables
(both at their locked severity) are produced together by `bench` and locked in
`tests/baselines/coco_scenes_benchmark.json`; reproduce them with:

```bash
.venv/bin/python -m proving_ground.cli bench \
  --images proving_ground/data/fixtures/coco_scenes/images \
  --ann   proving_ground/data/fixtures/coco_scenes/annotations.json \
  --model yolov8n.pt --out results.json
```

### Confidence intervals

Single-seed point estimates are easy to challenge, so `bench --seeds K` (K > 1)
adds error bars two independent ways: a **seed sweep** (sensitivity to each
attack's randomness — deterministic attacks honestly show a zero-width interval)
and an **image bootstrap** (1000 resamples — "would different images change the
number", which also surfaces the small-fixture uncertainty). Locked in
`tests/baselines/coco_scenes_benchmark_ci.json`:

| Attack | Attacked mAP (mean) | Seed 95% CI | Image-bootstrap 95% CI |
|---|---|---|---|
| fgsm | 0.073 | [0.073, 0.073] | [0.000, 0.131] |
| pgd-linf | 0.000 | [0.000, 0.000] | [0.000, 0.001] |
| pgd-l2 | 0.001 | [0.001, 0.001] | [0.000, 0.007] |
| patch | 0.065 | [0.065, 0.065] | [0.000, 0.098] |
| eot-patch | 0.131 | [0.108, 0.154] | [0.000, 0.165] |
| gaussian_blur | 0.273 | [0.273, 0.273] | [0.092, 0.652] |
| motion_blur | 0.000 | [0.000, 0.000] | [0.000, 0.000] |
| gaussian_noise | 0.009 | [0.003, 0.015] | [0.000, 0.125] |
| fog | 0.284 | [0.284, 0.284] | [0.235, 0.361] |
| low_light | 0.164 | [0.105, 0.224] | [0.000, 0.355] |
| jpeg_compression | 0.189 | [0.189, 0.189] | [0.000, 0.427] |
| smoke | 0.235 | [0.128, 0.342] | [0.125, 0.562] |
| dust | 0.285 | [0.224, 0.345] | [0.079, 0.552] |

Clean mAP = 0.39, image-bootstrap 95% CI [0.125, 0.460] — the wide bootstrap
intervals honestly reflect that a 3-image set is too small to pin the number
down. Reproduce with `bench --seeds 5 --bootstrap 1000`.

### Clean vs attacked detections

Generated by `tools/make_figures.py` (green = clean detections, red = under attack):

![FGSM](figures/fgsm_nyc_crossing.png)
![Patch](figures/patch_hanoi_market.png)
![EOT patch](figures/eot-patch_hanoi_market.png)
![Degradation: low-light](figures/low_light_nyc_crossing.png)
![Degradation: fog](figures/fog_nyc_crossing.png)
![Degradation: fog vs low-light](figures/dve_fog_lowlight_nyc.png)

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run

Weight-free smoke run (built-in `FakeDetector`, no downloads):

```bash
.venv/bin/python -m proving_ground.cli run \
  --images proving_ground/data/fixtures/images \
  --ann   proving_ground/data/fixtures/annotations.json \
  --model fake --attack fgsm --eps 0.03 --seed 0 --out report.json
```

Real YOLO (downloads `yolov8n.pt` on first use) — one command per attack:

```bash
SCENE="--images proving_ground/data/fixtures/coco_sample/images \
       --ann proving_ground/data/fixtures/coco_sample/annotations.json \
       --model yolov8n.pt --seed 0"

# FGSM (single-step generic gradient perturbation)
.venv/bin/python -m proving_ground.cli run $SCENE --attack fgsm --eps 0.03 --out fgsm.json

# PGD-Linf (iterative FGSM at the same eps budget — strictly stronger)
.venv/bin/python -m proving_ground.cli run $SCENE --attack pgd-linf \
  --eps 0.03 --pgd-steps 10 --pgd-step-size 0.0075 --out pgd_linf.json

# PGD-L2 (L2-ball mirror of PGD-Linf; eps bounds ||delta||_2)
.venv/bin/python -m proving_ground.cli run $SCENE --attack pgd-l2 \
  --pgd-l2-eps 3.0 --pgd-l2-steps 10 --pgd-l2-step-size 0.75 --out pgd_l2.json

# Optimized patch (localized sticker)
.venv/bin/python -m proving_ground.cli run $SCENE --attack patch \
  --patch-size 0.4 --steps 20 --step-size 0.1 --out patch.json

# EOT patch (robust to scale/rotation/lighting)
.venv/bin/python -m proving_ground.cli run $SCENE --attack eot-patch \
  --patch-size 0.4 --steps 15 --step-size 0.1 --out eot_patch.json

# Degradation / DVE (black-box; modes: gaussian_blur, motion_blur,
# gaussian_noise, fog, low_light, jpeg_compression, smoke, dust)
.venv/bin/python -m proving_ground.cli run $SCENE --attack degradation \
  --mode fog --severity 0.7 --out fog.json
```

## Test

```bash
.venv/bin/python -m pytest -q          # fast suite, no weight downloads
.venv/bin/python -m pytest -m integration   # opt-in: loads real YOLO weights
```

The default run excludes the `integration` marker, so it never downloads
weights and stays fast and deterministic.

## Layout

| Module       | Responsibility                                              |
|--------------|-------------------------------------------------------------|
| `adapters/`  | One interface every detector plugs in behind (`Detector`, optional `WhiteBox`) |
| `data/`      | Loaders + tiny committed fixtures                           |
| `attacks/`   | FGSM, PGD-Linf, PGD-L2, optimized patch, EOT patch (white-box); DVE degradation (black-box) |
| `eval/`      | IoU, per-class AP, mAP, robustness deltas                   |
| `report/`    | Versioned schema + JSON generator                           |
| `cli.py`     | Orchestrates one run end-to-end                             |
| `seeding.py` | Single entry point for reproducibility                      |

## Reproducibility

`seeding.set_seed()` pins Python/NumPy/Torch RNGs, enables deterministic
torch algorithms, and pins BLAS to a single thread (multi-threaded reductions
are non-associative in float and their scheduling varies across Python
processes, which would otherwise drift iterative attacks across sessions). It
is called by both the CLI and the test suite. The only non-reproducible field
in a report is `meta.timestamp_utc`, which is excluded from value comparisons.

## Adapter contract

A detector implements `Detector` — `predict(rgb_uint8_hwc) -> list[Detection]`
plus `class_names`. Boxes are canonical `xyxy`, absolute pixels. Gradient-based
attacks additionally require the optional `WhiteBox` protocol
(`to_input_tensor` + differentiable `compute_loss`), so black-box detectors
still work for everything except white-box attacks.

## Credits & licensing

This project's own code is licensed under **Apache-2.0** (see `LICENSE`).

All committed image fixtures are **CC0 / public domain**; per-image authors and
sources are listed in `NOTICE` (and each fixture's `SOURCE.md`). Note that the
default detector backend, **Ultralytics YOLO, is AGPL-3.0** — deploying this tool
together with ultralytics subjects that combination to the AGPL; the adapter
interface lets you swap in a differently-licensed detector. See `NOTICE` for the
full third-party and attribution details, and `CONTRIBUTING.md` for the
test tiers and baseline-regeneration workflow.
