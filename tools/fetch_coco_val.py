"""Fetch a reproducible COCO val2017 subset for the robustness scorecard.

COCO images carry mixed third-party licenses, so we don't commit them. This
downloads the official annotations and the first N val images (sorted by id, so
the subset is deterministic), into a directory you then point `compare --coco`
at:

    python tools/fetch_coco_val.py --out-dir coco_val --limit 100
    proving-ground compare --coco --limit 100 \\
        --images coco_val/val2017 \\
        --ann   coco_val/annotations/instances_val2017.json \\
        --models yolov8n.pt,yolov8s.pt,yolov8m.pt --out scorecard.json
    proving-ground report --in scorecard.json --out scorecard.html

The annotations archive is ~240 MB (one-time); images are ~150 KB each.
"""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
import zipfile
from pathlib import Path

ANN_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/val2017/{file_name}"
INST_MEMBER = "annotations/instances_val2017.json"
_HDR = {"User-Agent": "ProvingGroundAI/0.1 (coco subset fetcher)"}


def _download(url: str, dest: Path) -> None:
    ctx = ssl.create_default_context()
    req = urllib.request.Request(url, headers=_HDR)
    with urllib.request.urlopen(req, timeout=120, context=ctx) as r:
        dest.write_bytes(r.read())


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch a COCO val2017 subset.")
    ap.add_argument("--out-dir", default="coco_val", help="destination directory")
    ap.add_argument("--limit", type=int, default=100, help="number of images (first N by id)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "annotations").mkdir(parents=True, exist_ok=True)
    (out / "val2017").mkdir(parents=True, exist_ok=True)
    inst = out / "annotations" / "instances_val2017.json"

    if not inst.exists():
        zip_path = out / "annotations_trainval2017.zip"
        if not zip_path.exists():
            print(f"downloading annotations (~240 MB) -> {zip_path} ...")
            _download(ANN_ZIP_URL, zip_path)
        print("extracting instances_val2017.json ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract(INST_MEMBER, out)

    coco = json.loads(inst.read_text())
    subset = sorted(coco["images"], key=lambda im: im["id"])[: args.limit]
    for i, im in enumerate(subset, 1):
        dest = out / "val2017" / im["file_name"]
        if not dest.exists():
            _download(IMG_URL.format(file_name=im["file_name"]), dest)
        if i % 25 == 0 or i == len(subset):
            print(f"  images {i}/{len(subset)}")

    print(f"\nready: {len(subset)} images in {out / 'val2017'}")
    print(f"annotations: {inst}")
    print("now run:  proving-ground compare --coco --limit "
          f"{args.limit} --images {out / 'val2017'} --ann {inst} "
          "--models yolov8n.pt,yolov8s.pt,yolov8m.pt --out scorecard.json")


if __name__ == "__main__":
    main()
