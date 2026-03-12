"""
ultralytics_adapter.py — Adapter for all Ultralytics models.

Supported (all auto-download on first use):
  YOLOv8n / s / m / l / x
  YOLOv9c / e
  YOLOv10s / m / l
  RT-DETR-l / x
  Custom fine-tuned .pt files
"""

import os
import numpy as np
from .base import BaseAdapter


class UltralyticsAdapter(BaseAdapter):
    family      = "YOLO"
    _PERSON_CLS = 0       # COCO person = class 0 in ultralytics

    def __init__(self):
        self._model        = None
        self.name          = "ultralytics"
        self.model_size_mb = 0.0

    # ---------------------------------------------------------------- #
    def load(self, weights_path: str) -> str:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed.\n"
                "Fix: pip install ultralytics"
            )

        self._model        = YOLO(weights_path)
        self.name          = os.path.basename(weights_path)
        self.model_size_mb = self._get_size_mb(weights_path)
        return str(self._model.device)

    def is_loaded(self) -> bool:
        return self._model is not None

    # ---------------------------------------------------------------- #
    def predict(self, img_rgb: np.ndarray, conf: float, iou: float):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        res = self._model.predict(
            source=img_rgb,
            conf=conf,
            iou=iou,
            classes=[self._PERSON_CLS],
            verbose=False,
        )[0]

        if res.boxes is not None and res.boxes.xyxy is not None:
            boxes   = res.boxes.xyxy.cpu().numpy().tolist()
            scores  = res.boxes.conf.cpu().numpy().tolist()
            classes = [0] * len(boxes)
        else:
            boxes, scores, classes = [], [], []

        return boxes, scores, classes

    # ---------------------------------------------------------------- #
    @staticmethod
    def _get_size_mb(weights_path: str) -> float:
        """Try to find the cached file and return its size in MB."""
        candidates = [
            weights_path,
            os.path.join(os.environ.get("APPDATA", ""), "Ultralytics", weights_path),
        ]
        for p in candidates:
            if os.path.exists(p):
                return round(os.path.getsize(p) / 1e6, 1)
        return 0.0
