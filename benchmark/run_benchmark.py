"""
benchmark/run_benchmark.py — CLI benchmark runner.

Runs all 9 models on ECP val images and saves a CSV + prints a table.
Use this for the README results table.

Usage:
  python benchmark/run_benchmark.py --val_root C:\\ECP\\ECP\\val\\img --n 200

Output:
  benchmark/results/benchmark_YYYYMMDD_HHMMSS.csv
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow imports from viewer/
sys.path.insert(0, str(Path(__file__).parent.parent / "viewer"))

import numpy as np


MODELS = {
    # display name          weights key / path
    "YOLOv8n":              "yolov8n.pt",
    "YOLOv8s":              "yolov8s.pt",
    "YOLOv8m":              "yolov8m.pt",
    "YOLOv8l":              "yolov8l.pt",
    "RT-DETR-l":            "rtdetr-l.pt",
    "SSDLite320-MV3":       "ssdlite320",
    "FasterRCNN-MV3-320":   "fasterrcnn-mv3-320",
    "FasterRCNN-MV3-FPN":   "fasterrcnn-mv3-fpn",
    "YOLOv8s-ECP":          "runs/train/yolov8s-ecp/weights/best.pt",
}

CONF = 0.35
IOU  = 0.50


def collect_images(val_root: Path, max_n: int) -> list:
    exts = {".jpg", ".jpeg", ".png"}
    imgs = []
    for city in sorted(val_root.iterdir()):
        if not city.is_dir():
            continue
        for f in sorted(city.iterdir()):
            if f.suffix.lower() in exts:
                imgs.append(f)
    return imgs[:max_n]


def benchmark_model(name: str, weights: str, images: list) -> dict | None:
    try:
        from adapters import get_adapter
        from PIL import Image
    except ImportError as e:
        print(f"  Import error: {e}")
        return None

    adapter = get_adapter(weights)
    try:
        device = adapter.load(weights)
    except Exception as e:
        print(f"  Load failed: {e}")
        return None

    times, dets = [], []

    for img_path in images:
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
        except Exception:
            continue

        t0 = time.perf_counter()
        try:
            boxes, scores, _ = adapter.predict(img, CONF, IOU)
        except Exception:
            continue
        times.append((time.perf_counter() - t0) * 1000)
        dets.append(len(boxes))

    if not times:
        return None

    return {
        "model":    name,
        "family":   adapter.family,
        "size_mb":  adapter.model_size_mb,
        "device":   device,
        "n":        len(times),
        "avg_ms":   round(float(np.mean(times)),   1),
        "p50_ms":   round(float(np.percentile(times, 50)), 1),
        "p95_ms":   round(float(np.percentile(times, 95)), 1),
        "fps":      round(1000 / float(np.mean(times)), 1),
        "avg_dets": round(float(np.mean(dets)), 1),
    }


def print_table(results: list):
    header = f"{'Model':<26} {'Family':<12} {'MB':>6} {'Device':<8} {'Avg ms':>8} {'P95 ms':>8} {'FPS':>7} {'Avg dets':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    best_fps = max(r["fps"] for r in results)
    for r in sorted(results, key=lambda x: -x["fps"]):
        marker = " ◀ fastest" if r["fps"] == best_fps else ""
        print(
            f"{r['model']:<26} {r['family']:<12} {r['size_mb']:>6} {r['device']:<8} "
            f"{r['avg_ms']:>8} {r['p95_ms']:>8} {r['fps']:>7} {r['avg_dets']:>9}{marker}"
        )
    print("=" * len(header))


def save_csv(results: list, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "family", "size_mb", "device", "n", "avg_ms", "p50_ms", "p95_ms", "fps", "avg_dets"]
    with open(out_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in results:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nCSV saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="ECP model benchmark")
    parser.add_argument("--val_root", required=True,
                        help="Path to ECP val img folder (contains city subdirs)")
    parser.add_argument("--n", type=int, default=200,
                        help="Max images per model (default: 200)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        help="Which models to run (default: all)")
    args = parser.parse_args()

    val_root = Path(args.val_root)
    if not val_root.exists():
        print(f"ERROR: val_root not found: {val_root}")
        sys.exit(1)

    images = collect_images(val_root, args.n)
    print(f"Found {len(images)} images in {val_root}")

    results = []
    for name in args.models:
        if name not in MODELS:
            print(f"Unknown model: {name}, skipping.")
            continue

        weights = MODELS[name]

        # Skip fine-tuned model if weights don't exist yet
        if weights.endswith(".pt") and not os.path.exists(weights) and "/" in weights:
            print(f"\n[{name}] Skipping — fine-tuned weights not found at {weights}")
            continue

        print(f"\n[{name}] Loading {weights} ...")
        result = benchmark_model(name, weights, images)
        if result:
            results.append(result)
            print(f"  -> {result['avg_ms']} ms avg  |  {result['fps']} FPS  |  {result['avg_dets']} dets avg")
        else:
            print(f"  -> FAILED")

    if not results:
        print("No results to show.")
        return

    print_table(results)

    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).parent / "results" / f"benchmark_{ts}.csv"
    save_csv(results, out_path)


if __name__ == "__main__":
    main()
