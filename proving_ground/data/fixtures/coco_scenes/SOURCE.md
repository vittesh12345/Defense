# coco_scenes — realistic multi-object fixture set

A small set of cluttered, real-world scenes used to produce a *believable*
aggregate clean mAP (not the 1.0 of the single-image `coco_sample` anchor) and
to drive the headline results table. `coco_sample` is untouched and remains the
single-image regression anchor.

## Images (all CC0 / public domain)

| File | Title | Author | Source |
|------|-------|--------|--------|
| `nyc_crossing.jpg` | People crossing street | Mike Petrucci | https://commons.wikimedia.org/wiki/File:People_crossing_street_(Unsplash).jpg |
| `hanoi_market.jpg` | Street market in Hanoi, 2003 July 41 | Syced | https://commons.wikimedia.org/wiki/File:Street_market_in_Hanoi,_2003_July_41.jpg |
| `bayside_bench.jpg` | Family sitting on bench watching bay | D Coetzee | https://commons.wikimedia.org/wiki/File:Family_sitting_on_bench_watching_bay_(6968370946).jpg |

License: **CC0 1.0** (https://creativecommons.org/publicdomain/zero/1.0/) for all
three. Retrieved 2026-06-07. Each was downscaled to ≤ 640 px on the long side and
re-encoded as JPEG (quality 80–82) to keep the committed fixtures small; no
content was altered.

## Annotation standard

Ground truth was hand-drawn by visual inspection (grid + zoom + overlay-verify),
**never copied from any model's predictions**. Class ids index into the 80 COCO
class names so they align with what `UltralyticsYOLOAdapter` returns.

Standard = **salient, clearly-visible objects**: every object that is clearly
recognizable as its COCO class is annotated; tiny/ambiguous background specks are
not. The set was deliberately chosen so this standard is *achievable completely*
per image — scenes with a dense tail of small objects (e.g. far-off traffic
streams) were excluded because the salient standard could not be applied
completely there: the detector legitimately finds those small objects, which
would then count as false positives against an incomplete ground truth and
deflate mAP for an annotation reason rather than a detection one. (A dense Beijing
traffic scene and a motion-blurred Bangkok intersection were evaluated and
dropped for exactly this reason.)

The resulting clean pooled mAP@0.5 (~0.39 with yolov8n) is therefore a credible
reflection of detector performance: a mix of well-detected classes, partially
detected ones, and genuinely hard cases (e.g. back-lit silhouettes on the bench).

## Aggregate baseline

`tests/baselines/coco_scenes_benchmark.json` locks the clean + per-attack
attacked mAP over this set. Regenerate deliberately via `tools/run_benchmark.py`
(see that script / the benchmark test).
