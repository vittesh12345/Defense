# Proving Ground AI — Robustness Testing Engine

Stress-tests object-detection models (YOLO, etc.) with adversarial attacks and
degradations, measures the failure, and emits a reproducible JSON assurance
report. The output is **measurement** — correctness and reproducibility beat
speed.

This is the **Phase 0 MVP**: one detector adapter, a clean mAP evaluation, one
attack (FGSM), and a JSON report, end-to-end.

## Install

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Run

Weight-free (uses the built-in `FakeDetector`, no downloads):

```bash
.venv/bin/python -m proving_ground.cli run \
  --images proving_ground/data/fixtures/images \
  --ann   proving_ground/data/fixtures/annotations.json \
  --model fake --attack fgsm --eps 0.03 --seed 0 \
  --out report.json
```

Real YOLO (downloads `yolov8n.pt` on first use):

```bash
.venv/bin/python -m proving_ground.cli run \
  --images proving_ground/data/fixtures/images \
  --ann   proving_ground/data/fixtures/annotations.json \
  --model yolov8n.pt --attack fgsm --eps 0.03 --seed 0 \
  --out report.json
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
| `attacks/`   | Perturbations (FGSM today)                                  |
| `eval/`      | IoU, per-class AP, mAP, robustness deltas                   |
| `report/`    | Versioned schema + JSON generator                           |
| `cli.py`     | Orchestrates one run end-to-end                             |
| `seeding.py` | Single entry point for reproducibility                      |

## Reproducibility

`seeding.set_seed()` pins Python/NumPy/Torch RNGs and enables deterministic
torch algorithms; it is called by both the CLI and the test suite. The only
non-reproducible field in a report is `meta.timestamp_utc`, which is excluded
from value comparisons.

## Adapter contract

A detector implements `Detector` — `predict(rgb_uint8_hwc) -> list[Detection]`
plus `class_names`. Boxes are canonical `xyxy`, absolute pixels. Gradient-based
attacks additionally require the optional `WhiteBox` protocol
(`to_input_tensor` + differentiable `compute_loss`), so black-box detectors
still work for everything except white-box attacks.
