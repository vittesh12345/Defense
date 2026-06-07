"""Regenerate a locked baseline from a fresh report.

Usage:
    .venv/bin/python tests/baselines/_regen.py <new_report.json> [baseline_name.json]

The optional second arg names the baseline file under tests/baselines/
(default: coco_sample_report.json — the FGSM snapshot).

Deliberate, reviewed step only — see fixtures/coco_sample/SOURCE.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import tests/ helpers
from _snapshot_util import normalize_report  # noqa: E402

BASELINE_DIR = Path(__file__).resolve().parent


def main(src: str, baseline_name: str = "coco_sample_report.json") -> None:
    report = json.loads(Path(src).read_text())
    normalized = normalize_report(report)
    out = BASELINE_DIR / baseline_name
    out.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:])
