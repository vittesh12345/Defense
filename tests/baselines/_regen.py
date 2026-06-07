"""Regenerate the locked baseline from a fresh report.

Usage:
    .venv/bin/python tests/baselines/_regen.py /tmp/new.json

Deliberate, reviewed step only — see fixtures/coco_sample/SOURCE.md.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # import tests/ helpers
from _snapshot_util import normalize_report  # noqa: E402

BASELINE = Path(__file__).resolve().parent / "coco_sample_report.json"


def main(src: str) -> None:
    report = json.loads(Path(src).read_text())
    normalized = normalize_report(report)
    BASELINE.write_text(json.dumps(normalized, indent=2, sort_keys=True) + "\n")
    print(f"wrote {BASELINE}")


if __name__ == "__main__":
    main(sys.argv[1])
