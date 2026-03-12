"""
convert_ecp_to_yolo.py — Convert ECP dataset annotations to YOLO format.

ECP annotation format (JSON):
  {
    "imagewidth": 1920,
    "imageheight": 1024,
    "children": [
      { "identity": "pedestrian", "x0": 100, "y0": 200, "x1": 150, "y1": 350 },
      ...
    ]
  }

YOLO format (one .txt per image):
  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
  class_id is always 0 (pedestrian only)

Usage:
  python training/convert_ecp_to_yolo.py ^
      --ecp_root  C:\\ECP ^
      --out_root  dataset
"""

import os
import json
import shutil
import argparse
from pathlib import Path


IDENTITIES_TO_KEEP = {"pedestrian"}   # ignore "rider", "vehicle", etc.
CLASS_ID           = 0                # only one class


def convert_split(ecp_img_dir: Path, ecp_lbl_dir: Path,
                  out_img_dir: Path,  out_lbl_dir: Path,
                  split: str) -> tuple[int, int]:
    """
    Convert one split (train / val).
    Returns (n_images_processed, n_boxes_total).
    """
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    n_boxes  = 0

    city_dirs = sorted(ecp_lbl_dir.iterdir())
    total_cities = len(city_dirs)

    for city_idx, city_dir in enumerate(city_dirs, 1):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        print(f"  [{city_idx}/{total_cities}] {split}/{city} ...", end=" ", flush=True)

        city_boxes = 0
        json_files = sorted(city_dir.glob("*.json"))

        for json_path in json_files:
            stem = json_path.stem   # e.g. "amsterdam_000001_000019_leftImg8bit"

            # ── Parse annotation ───────────────────────────────── #
            with open(json_path, "r") as f:
                ann = json.load(f)

            img_w = ann.get("imagewidth",  1920)
            img_h = ann.get("imageheight", 1024)

            lines = []
            for child in ann.get("children", []):
                if child.get("identity") not in IDENTITIES_TO_KEEP:
                    continue

                x0 = float(child["x0"])
                y0 = float(child["y0"])
                x1 = float(child["x1"])
                y1 = float(child["y1"])

                # Skip degenerate boxes
                if x1 <= x0 or y1 <= y0:
                    continue

                # Convert to YOLO normalised cx/cy/w/h
                cx = ((x0 + x1) / 2) / img_w
                cy = ((y0 + y1) / 2) / img_h
                bw = (x1 - x0) / img_w
                bh = (y1 - y0) / img_h

                # Clamp to [0, 1]
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.0, min(1.0, bw))
                bh = max(0.0, min(1.0, bh))

                lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            # ── Write label file ───────────────────────────────── #
            lbl_out = out_lbl_dir / f"{stem}.txt"
            with open(lbl_out, "w") as f:
                f.write("\n".join(lines))

            # ── Copy image ─────────────────────────────────────── #
            # Try common ECP image extensions
            copied = False
            for ext in (".png", ".jpg", ".jpeg"):
                src_img = ecp_img_dir / city / f"{stem}{ext}"
                if src_img.exists():
                    shutil.copy2(src_img, out_img_dir / f"{stem}{ext}")
                    copied = True
                    break

            if not copied:
                # Write empty label and skip (model ignores images with no boxes)
                pass

            n_images  += 1
            city_boxes += len(lines)
            n_boxes    += len(lines)

        print(f"{len(json_files)} images, {city_boxes} boxes")

    return n_images, n_boxes


def main():
    parser = argparse.ArgumentParser(
        description="Convert ECP annotations to YOLO format"
    )
    parser.add_argument(
        "--ecp_root", required=True,
        help="Root of ECP dataset, e.g. C:\\ECP"
    )
    parser.add_argument(
        "--out_root", default="dataset",
        help="Output directory (default: dataset/)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="Splits to convert (default: train val)"
    )
    args = parser.parse_args()

    ecp_root = Path(args.ecp_root)
    out_root = Path(args.out_root)

    print(f"\nECP root : {ecp_root}")
    print(f"Output   : {out_root.resolve()}\n")

    total_images = 0
    total_boxes  = 0

    for split in args.splits:
        print(f"Converting {split}...")

        ecp_img_dir = ecp_root / "ECP" / split / "img"
        ecp_lbl_dir = ecp_root / "ECP" / split / "labels"

        if not ecp_img_dir.exists():
            # Try alternative ECP directory layout
            ecp_img_dir = ecp_root / split / "img"
            ecp_lbl_dir = ecp_root / split / "labels"

        if not ecp_img_dir.exists():
            print(f"  WARNING: could not find images at {ecp_img_dir}, skipping.")
            continue

        out_img_dir = out_root / "images" / split
        out_lbl_dir = out_root / "labels" / split

        n_img, n_box = convert_split(
            ecp_img_dir, ecp_lbl_dir,
            out_img_dir, out_lbl_dir,
            split,
        )
        total_images += n_img
        total_boxes  += n_box
        print(f"  -> {n_img} images, {n_box} boxes\n")

    print("=" * 50)
    print(f"Total: {total_images} images, {total_boxes} boxes")
    print(f"Output saved to: {out_root.resolve()}")
    print("\nNext step: python training/train.py")


if __name__ == "__main__":
    main()
