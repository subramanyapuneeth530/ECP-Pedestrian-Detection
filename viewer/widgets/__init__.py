"""
adapters/__init__.py — Model registry and router.

MODEL_MENU  : ordered dict shown in the GUI dropdown
get_adapter : factory — routes a weights_path to the right adapter
"""

from .ultralytics_adapter import UltralyticsAdapter
from .mobilenet_adapter   import MobileNetAdapter

# ── Model menu (display name → weights key / path) ──────────────────── #
# None entries are visual separators in the dropdown.
MODEL_MENU: dict = {
    # ── YOLO family ──────────────────────────────────────────────────
    "── YOLO ────────────────────":         None,
    "YOLOv8n  · nano   ~3 MB":              "yolov8n.pt",
    "YOLOv8s  · small  ~11 MB":             "yolov8s.pt",
    "YOLOv8m  · medium ~25 MB":             "yolov8m.pt",
    "YOLOv8l  · large  ~43 MB":             "yolov8l.pt",
    "RT-DETR-l · transformer":              "rtdetr-l.pt",

    # ── MobileNet family ─────────────────────────────────────────────
    "── MobileNet ───────────────":         None,
    "SSDLite320-MobileNetV3     · fastest": "ssdlite320",
    "FasterRCNN-MobileNetV3-320 · mobile":  "fasterrcnn-mv3-320",
    "FasterRCNN-MobileNetV3-FPN · accurate":"fasterrcnn-mv3-fpn",

    # ── Fine-tuned (local) ────────────────────────────────────────────
    "── Fine-tuned ──────────────":         None,
    "YOLOv8s  · ECP fine-tuned":            "runs/train/yolov8s-ecp/weights/best.pt",
}


def get_adapter(weights_path: str) -> "BaseAdapter":
    """
    Route weights_path → correct adapter.
      .pt files           → UltralyticsAdapter
      MobileNet keys      → MobileNetAdapter
    """
    key = weights_path.strip().lower()

    if key in MobileNetAdapter.AVAILABLE:
        return MobileNetAdapter()

    # default: ultralytics handles all .pt models
    return UltralyticsAdapter()
