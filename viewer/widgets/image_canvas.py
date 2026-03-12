"""
widgets/image_canvas.py

QLabel-based image display that:
  - scales images to fit the available space (keeps aspect ratio)
  - draws bounding boxes + confidence labels + centroids via QPainter
  - exposes set_image() and set_detections() for the main window
"""

from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui     import (
    QPixmap, QPainter, QPen, QColor, QFont, QImage, QBrush
)
from PySide6.QtCore    import Qt, QRect, QPoint
import numpy as np


# Box drawing constants
BOX_COLOR     = QColor("#4ec94e")   # green
BOX_THICKNESS = 2
LABEL_FONT_SZ = 11
DOT_RADIUS    = 4


class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background-color: #1e1e1e;")
        self.setMinimumSize(640, 480)

        # State
        self._orig_pixmap   = None    # original image as QPixmap
        self._boxes         = []      # [[x1,y1,x2,y2], ...]  original coords
        self._scores        = []
        self._show_centroid = True
        self._img_meta      = ""      # filename + resolution string

    # ── Public API ──────────────────────────────────────────────────── #

    def set_image(self, img_rgb: np.ndarray, meta: str = ""):
        """Display a new image. img_rgb is H×W×3 uint8."""
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(
            img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        self._orig_pixmap = QPixmap.fromImage(qimg)
        self._boxes       = []
        self._scores      = []
        self._img_meta    = meta
        self._refresh()

    def set_detections(self, boxes, scores, show_centroid: bool = True):
        """Overlay detection boxes on the current image."""
        self._boxes         = boxes
        self._scores        = scores
        self._show_centroid = show_centroid
        self._refresh()

    def clear(self):
        self._orig_pixmap = None
        self._boxes       = []
        self._scores      = []
        self._img_meta    = ""
        self.clear()

    def set_show_centroid(self, value: bool):
        self._show_centroid = value
        self._refresh()

    # ── Qt overrides ────────────────────────────────────────────────── #

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh()

    # ── Internal ────────────────────────────────────────────────────── #

    def _refresh(self):
        if self._orig_pixmap is None:
            self._draw_placeholder()
            return

        # Scale image to fit canvas while keeping aspect ratio
        canvas_w = self.width()
        canvas_h = self.height()
        scaled   = self._orig_pixmap.scaled(
            canvas_w, canvas_h,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        # Compute offset to centre the scaled image
        off_x = (canvas_w - scaled.width())  // 2
        off_y = (canvas_h - scaled.height()) // 2

        # Scale factors (original → display)
        sx = scaled.width()  / self._orig_pixmap.width()
        sy = scaled.height() / self._orig_pixmap.height()

        # Create composite pixmap (canvas-sized, dark bg)
        composite = QPixmap(canvas_w, canvas_h)
        composite.fill(QColor("#1e1e1e"))

        painter = QPainter(composite)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw image
        painter.drawPixmap(off_x, off_y, scaled)

        # Draw metadata text (top-left, inside image)
        if self._img_meta:
            painter.setPen(QPen(QColor("#f0f0f0")))
            painter.setFont(QFont("Segoe UI", 9))
            painter.drawText(off_x + 8, off_y + 18, self._img_meta)

        # Draw boxes
        pen = QPen(BOX_COLOR, BOX_THICKNESS)
        painter.setPen(pen)
        font = QFont("Segoe UI", LABEL_FONT_SZ)
        painter.setFont(font)

        for box, score in zip(self._boxes, self._scores):
            x1 = int(box[0] * sx) + off_x
            y1 = int(box[1] * sy) + off_y
            x2 = int(box[2] * sx) + off_x
            y2 = int(box[3] * sy) + off_y

            # Rectangle
            painter.setPen(QPen(BOX_COLOR, BOX_THICKNESS))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            # Label background + text
            label   = f"person {score:.2f}"
            lbl_y   = max(off_y + 2, y1 - 18)
            bg_rect = QRect(x1, lbl_y, len(label) * 7 + 6, 18)
            painter.fillRect(bg_rect, QColor(0, 120, 212, 200))
            painter.setPen(QPen(Qt.white))
            painter.drawText(bg_rect, Qt.AlignCenter, label)

            # Centroid dot
            if self._show_centroid:
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(BOX_COLOR))
                painter.drawEllipse(
                    QPoint(cx, cy), DOT_RADIUS, DOT_RADIUS
                )

        # Detection count (bottom-left of image)
        if self._boxes:
            count_text = f"  {len(self._boxes)} detection{'s' if len(self._boxes) != 1 else ''}  "
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
            count_rect = QRect(off_x, off_y + scaled.height() - 24,
                               len(count_text) * 8, 22)
            painter.fillRect(count_rect, QColor(0, 0, 0, 160))
            painter.setPen(QPen(QColor("#4ec94e")))
            painter.setFont(QFont("Segoe UI", 10, QFont.Bold))
            painter.drawText(count_rect, Qt.AlignVCenter | Qt.AlignLeft, count_text)

        painter.end()
        self.setPixmap(composite)

    def _draw_placeholder(self):
        canvas_w = max(self.width(),  640)
        canvas_h = max(self.height(), 480)
        pm = QPixmap(canvas_w, canvas_h)
        pm.fill(QColor("#1e1e1e"))
        painter = QPainter(pm)
        painter.setPen(QPen(QColor("#3f3f46")))
        painter.setFont(QFont("Segoe UI", 14))
        painter.drawText(
            pm.rect(), Qt.AlignCenter,
            "Select a dataset folder and city to begin"
        )
        painter.end()
        self.setPixmap(pm)
