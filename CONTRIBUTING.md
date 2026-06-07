# Contributing / reproducing

## Setup

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

## Tests — two tiers

```bash
.venv/bin/python -m pytest -q          # fast: weight-free, deterministic, no downloads
.venv/bin/python -m pytest -m integration   # opt-in: loads real YOLO weights (downloads once)
```

The default `pytest -q` excludes the `integration` marker, so it never downloads
model weights and stays fast. Always run **both** tiers before committing.

## Lint

```bash
.venv/bin/ruff check proving_ground tests tools
```

## Hard rules (this is a measurement tool)

- Every numeric result must be reproducible: seeds + torch deterministic flags
  are set centrally in `proving_ground/seeding.py`.
- Float comparisons use tolerances (`pytest.approx` / `np.allclose`), never `==`.
- Small committed fixtures only; never download datasets inside tests.
- **Never change a locked metric value without a deliberate, reviewed baseline
  update.** The golden snapshots live in `tests/baselines/`:
  - `coco_sample_report.json` — FGSM, single-image anchor
  - `coco_sample_patch_report.json` — patch, single-image anchor
  - `coco_sample_eot_patch_report.json` — EOT patch, single-image anchor
  - `coco_scenes_benchmark.json` — aggregate benchmark over the realistic set

## Regenerating a baseline (deliberate step)

Single-attack snapshots — run the pipeline, then normalize + write the baseline:

```bash
.venv/bin/python -m proving_ground.cli run \
  --images proving_ground/data/fixtures/coco_sample/images \
  --ann   proving_ground/data/fixtures/coco_sample/annotations.json \
  --model yolov8n.pt --attack fgsm --eps 0.03 --seed 0 --out /tmp/new.json
.venv/bin/python tests/baselines/_regen.py /tmp/new.json coco_sample_report.json
```

(See `proving_ground/data/fixtures/coco_sample/SOURCE.md` for the patch / EOT
commands.) The aggregate benchmark baseline regenerates directly:

```bash
.venv/bin/python -m proving_ground.cli bench \
  --images proving_ground/data/fixtures/coco_scenes/images \
  --ann   proving_ground/data/fixtures/coco_scenes/annotations.json \
  --model yolov8n.pt --out tests/baselines/coco_scenes_benchmark.json
```

Review the diff and the reason for any change before committing it.

## Adding image fixtures

Use only small **CC0 / public-domain** images. Record source + license in a
`SOURCE.md` next to them, add the credit to the top-level `NOTICE`, and
hand-annotate by visual inspection — never copy a model's predictions.
