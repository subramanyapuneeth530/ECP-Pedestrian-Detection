"""
mobilenet_adapter.py — Adapter for MobileNet-based detection models (torchvision).

Supported (all auto-download on first use via torchvision):
  ssdlite320              SSDLite320 + MobileNetV3        fastest / lightest
  fasterrcnn-mv3-320      FasterRCNN + MobileNetV3-320    mobile-tuned
  fasterrcnn-mv3-fpn      FasterRCNN + MobileNetV3-FPN    most accurate

NOTE: torchvision uses COCO person = class 1 (not 0).
      We normalise all output classes to 0 for the viewer.
"""

import numpy as np
import torch
from .base import BaseAdapter


class MobileNetAdapter(BaseAdapter):
    family      = "MobileNet"
    _PERSON_CLS = 1       # torchvision COCO: person = class 1

    # Registry of supported model keys -------------------------------- #
    AVAILABLE: dict = {}   # populated below after class definition

    def __init__(self):
        self._model        = None
        self._device       = None
        self.name          = "mobilenet"
        self.model_size_mb = 0.0

    # ---------------------------------------------------------------- #
    def load(self, weights_path: str) -> str:
        """
        weights_path is a key from AVAILABLE,
        e.g. "ssdlite320" or "fasterrcnn-mv3-fpn"
        """
        key = weights_path.strip().lower()
        if key not in self.AVAILABLE:
            raise ValueError(
                f"Unknown MobileNet key: '{key}'\n"
                f"Choose from: {list(self.AVAILABLE.keys())}"
            )

        try:
            import torchvision
        except ImportError:
            raise ImportError(
                "torchvision not installed.\n"
                "Fix: pip install torchvision"
            )

        cfg = self.AVAILABLE[key]
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model  = cfg["loader"]().eval().to(self._device)
        self.name    = cfg["display"]

        # estimate size from parameter bytes
        param_bytes        = sum(
            p.numel() * p.element_size() for p in self._model.parameters()
        )
        self.model_size_mb = round(param_bytes / 1e6, 1)
        return self._device

    def is_loaded(self) -> bool:
        return self._model is not None

    # ---------------------------------------------------------------- #
    def predict(self, img_rgb: np.ndarray, conf: float, iou: float):
        if not self.is_loaded():
            raise RuntimeError("Model not loaded. Call load() first.")

        import torchvision.transforms.functional as F

        tensor = F.to_tensor(img_rgb).unsqueeze(0).to(self._device)

        with torch.no_grad():
            preds = self._model(tensor)[0]

        boxes_out, scores_out, classes_out = [], [], []
        for box, score, label in zip(
            preds["boxes"].cpu().numpy(),
            preds["scores"].cpu().numpy(),
            preds["labels"].cpu().numpy(),
        ):
            if float(score) < conf:
                continue
            if int(label) != self._PERSON_CLS:
                continue
            boxes_out.append(box.tolist())
            scores_out.append(float(score))
            classes_out.append(0)          # normalise to 0 for viewer

        return boxes_out, scores_out, classes_out


# ── Populate AVAILABLE after import so torchvision is only imported lazily ── #
def _build_available():
    try:
        import torchvision.models.detection as D

        return {
            "ssdlite320": {
                "display": "SSDLite320-MobileNetV3",
                "loader":  lambda: D.ssdlite320_mobilenet_v3_large(
                    weights=D.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
                ),
            },
            "fasterrcnn-mv3-320": {
                "display": "FasterRCNN-MobileNetV3-320",
                "loader":  lambda: D.fasterrcnn_mobilenet_v3_large_320_fpn(
                    weights=D.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
                ),
            },
            "fasterrcnn-mv3-fpn": {
                "display": "FasterRCNN-MobileNetV3-FPN",
                "loader":  lambda: D.fasterrcnn_mobilenet_v3_large_fpn(
                    weights=D.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
                ),
            },
        }
    except ImportError:
        return {}


MobileNetAdapter.AVAILABLE = _build_available()
