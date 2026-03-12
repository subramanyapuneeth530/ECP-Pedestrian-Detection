"""
train.py — Fine-tune YOLOv8s on the ECP pedestrian dataset.

Prerequisites:
  1. Run convert_ecp_to_yolo.py first to build dataset/
  2. GPU with CUDA recommended (CPU training will be very slow)

Usage:
  python training/train.py [options]

Results saved to:
  runs/train/yolov8s-ecp/weights/best.pt   <- use this in the viewer
"""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 on ECP")
    parser.add_argument("--model",   default="yolov8s.pt",
                        help="Base weights (default: yolov8s.pt)")
    parser.add_argument("--epochs",  type=int, default=50,
                        help="Training epochs (default: 50)")
    parser.add_argument("--batch",   type=int, default=16,
                        help="Batch size (default: 16, reduce if OOM)")
    parser.add_argument("--imgsz",   type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--device",  default="0",
                        help="Device: 0 for GPU, cpu for CPU (default: 0)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--name",    default="yolov8s-ecp",
                        help="Run name under runs/train/ (default: yolov8s-ecp)")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from last checkpoint")
    args = parser.parse_args()

    # ── Sanity checks ─────────────────────────────────────────────── #
    yaml_path    = Path(__file__).parent / "ecp.yaml"
    dataset_path = Path(__file__).parent.parent / "dataset"

    if not yaml_path.exists():
        raise FileNotFoundError(f"ecp.yaml not found at {yaml_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"dataset/ folder not found at {dataset_path}\n"
            f"Run convert_ecp_to_yolo.py first."
        )

    train_imgs = dataset_path / "images" / "train"
    val_imgs   = dataset_path / "images" / "val"

    if not train_imgs.exists() or not any(train_imgs.iterdir()):
        raise FileNotFoundError(
            f"No training images found at {train_imgs}\n"
            f"Run convert_ecp_to_yolo.py first."
        )

    n_train = len(list(train_imgs.rglob("*.png"))) + len(list(train_imgs.rglob("*.jpg")))
    n_val   = len(list(val_imgs.rglob("*.png")))   + len(list(val_imgs.rglob("*.jpg")))

    print(f"\nECP Fine-tuning")
    print(f"{'='*50}")
    print(f"  Base model : {args.model}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch size : {args.batch}")
    print(f"  Image size : {args.imgsz}")
    print(f"  Device     : {args.device}")
    print(f"  Train imgs : {n_train:,}")
    print(f"  Val imgs   : {n_val:,}")
    print(f"  Output     : runs/train/{args.name}/")
    print(f"{'='*50}\n")

    # ── Training ──────────────────────────────────────────────────── #
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics not installed. Run: pip install ultralytics")

    model = YOLO(args.model)

    results = model.train(
        data      = str(yaml_path),
        epochs    = args.epochs,
        imgsz     = args.imgsz,
        batch     = args.batch,
        device    = args.device,
        workers   = args.workers,
        project   = "runs/train",
        name      = args.name,
        resume    = args.resume,
        # Augmentation — good defaults for pedestrian detection
        hsv_h     = 0.015,
        hsv_s     = 0.7,
        hsv_v     = 0.4,
        degrees   = 0.0,     # no rotation (pedestrians are always upright)
        translate = 0.1,
        scale     = 0.5,
        flipud    = 0.0,     # no vertical flip
        fliplr    = 0.5,
        mosaic    = 1.0,
        mixup     = 0.0,
        # Logging
        plots     = True,
        save      = True,
        exist_ok  = True,
    )

    best_weights = Path("runs") / "train" / args.name / "weights" / "best.pt"
    print(f"\nTraining complete!")
    print(f"Best weights: {best_weights.resolve()}")
    print(f"\nTo use in the viewer, select:")
    print(f"  Model dropdown -> 'YOLOv8s · ECP fine-tuned'")
    print(f"  or paste the path above into the custom path field.")

    # Print val metrics
    print(f"\nFinal validation metrics:")
    print(f"  mAP@0.5     : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"  mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
    print(f"  Precision   : {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
    print(f"  Recall      : {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
