# Fixture source & license

## images/sf_cable_car.jpg

- **Title:** Cable Cars on California Street close to Drumm
- **Author:** SaarPro (Wikimedia Commons user)
- **Source:** https://commons.wikimedia.org/wiki/File:Cable_Cars_on_California_Street_close_to_Drumm.jpg
- **License:** CC0 1.0 Universal (Public Domain Dedication) — https://creativecommons.org/publicdomain/zero/1.0/deed.en
- **Retrieved:** 2026-06-07
- **Modifications:** downscaled to 640×480 and re-encoded as JPEG (quality 80) to keep
  the committed fixture small (~87 KB). No content was altered.

CC0 places the work in the public domain, so no attribution is legally required;
it is recorded here for provenance.

## annotations.json

Ground-truth boxes were hand-drawn by visual inspection of the committed
640×480 image (not copied from any model's predictions). Class ids index into
the 80 COCO class names (so they line up with what `UltralyticsYOLOAdapter`
returns): `person` = 0, `train` = 6 (the string of cable cars is annotated as a
single `train`).

## Regenerating the golden baseline

`tests/baselines/coco_sample_report.json` is the locked snapshot of the real
clean→attacked measurement on this fixture. It must only change as a deliberate,
reviewed baseline update. To regenerate after an intentional change:

```bash
.venv/bin/python -m proving_ground.cli run \
  --images proving_ground/data/fixtures/coco_sample/images \
  --ann   proving_ground/data/fixtures/coco_sample/annotations.json \
  --model yolov8n.pt --eps 0.03 --seed 0 --out /tmp/new.json
# then normalise (drop timestamp + filesystem paths) and overwrite the baseline:
.venv/bin/python tests/baselines/_regen.py /tmp/new.json
```
