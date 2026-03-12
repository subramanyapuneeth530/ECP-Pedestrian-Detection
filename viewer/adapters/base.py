"""
base.py — Abstract base class for all model adapters.

Every adapter must implement:
  load(weights_path)  -> str            (device string)
  predict(img_rgb, conf, iou) -> (boxes, scores, classes)

This keeps the viewer completely decoupled from model internals.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseAdapter(ABC):
    name:           str   = "base"
    family:         str   = "unknown"
    model_size_mb:  float = 0.0

    @abstractmethod
    def load(self, weights_path: str) -> str:
        """Load the model. Return device string e.g. 'cuda:0' or 'cpu'."""
        ...

    @abstractmethod
    def predict(
        self,
        img_rgb: np.ndarray,
        conf: float,
        iou: float,
    ):
        """
        Run inference on one RGB image (H x W x 3, uint8).

        Returns:
            boxes   : list of [x1, y1, x2, y2]   pixel coords, original size
            scores  : list of float
            classes : list of int  (always 0 = person, normalised)
        """
        ...

    def is_loaded(self) -> bool:
        return False

    def get_info(self) -> dict:
        return {
            "name":    self.name,
            "family":  self.family,
            "size_mb": self.model_size_mb,
        }
