"""
widgets/stats_panel.py

Right sidebar showing live inference statistics:
  - Model name + family
  - FPS
  - Inference time (ms)
  - Detection count
  - Confidence threshold
  - Device (CPU / CUDA)
"""

from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QLabel, QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt


class StatsPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self.setFixedWidth(200)
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(4)

        layout.addWidget(self._section("STATS"))

        # Each stat: (attribute_name, label_text)
        stats = [
            ("_fps",        "FPS"),
            ("_inf_ms",     "Inference (ms)"),
            ("_det_count",  "Detections"),
            ("_conf",       "Conf threshold"),
        ]

        for attr, label in stats:
            layout.addWidget(self._small_label(label))
            val_lbl = QLabel("—")
            val_lbl.setObjectName("StatValue")
            setattr(self, attr, val_lbl)
            layout.addWidget(val_lbl)
            layout.addSpacing(6)

        layout.addWidget(self._separator())
        layout.addWidget(self._section("MODEL"))

        self._model_name = QLabel("None loaded")
        self._model_name.setWordWrap(True)
        self._model_name.setStyleSheet("color: #d4d4d4; font-size: 12px;")
        layout.addWidget(self._model_name)

        self._model_family = QLabel("")
        self._model_family.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        layout.addWidget(self._model_family)

        self._model_size = QLabel("")
        self._model_size.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        layout.addWidget(self._model_size)

        layout.addWidget(self._separator())
        layout.addWidget(self._section("DEVICE"))

        self._device_lbl = QLabel("—")
        self._device_lbl.setStyleSheet("color: #4ec94e; font-size: 13px; font-weight: bold;")
        layout.addWidget(self._device_lbl)

        layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    # ── Public update methods ────────────────────────────────────────── #
    def update_inference(self, inf_ms: float, det_count: int, conf: float):
        fps = 1000 / inf_ms if inf_ms > 0 else 0
        self._fps.setText(f"{fps:.1f}")
        self._inf_ms.setText(f"{inf_ms:.0f}")
        self._det_count.setText(str(det_count))
        self._conf.setText(f"{conf:.2f}")

    def update_model(self, name: str, family: str, size_mb: float, device: str):
        self._model_name.setText(name)
        self._model_family.setText(family)
        self._model_size.setText(f"{size_mb} MB" if size_mb else "")
        self._device_lbl.setText(device.upper())

        # colour device label
        color = "#4ec94e" if "cuda" in device.lower() else "#ce9178"
        self._device_lbl.setStyleSheet(
            f"color: {color}; font-size: 13px; font-weight: bold;"
        )

    def reset(self):
        for attr in ("_fps", "_inf_ms", "_det_count", "_conf"):
            getattr(self, attr).setText("—")

    # ── Widget factories ─────────────────────────────────────────────── #
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    @staticmethod
    def _small_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        return lbl

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        return line
