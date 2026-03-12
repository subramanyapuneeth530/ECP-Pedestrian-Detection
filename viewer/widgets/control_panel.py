"""
widgets/control_panel.py

Left sidebar containing:
  - Dataset folder picker
  - City dropdown
  - Model family + model dropdowns (grouped)
  - Load model button
  - Conf / IoU sliders
  - Save drawn / Show centroid checkboxes
  - Navigation buttons (Prev / Next / Infer)
"""

import os
from PySide6.QtWidgets import (
    QFrame, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSlider, QCheckBox, QFileDialog,
    QSizePolicy, QSpacerItem
)
from PySide6.QtCore import Qt, Signal


class ControlPanel(QFrame):
    # Signals emitted to the main window
    sig_root_changed    = Signal(str)          # new dataset root
    sig_city_changed    = Signal(str)          # city name selected
    sig_load_model      = Signal(str)          # weights key / path
    sig_prev            = Signal()
    sig_next            = Signal()
    sig_infer           = Signal()
    sig_conf_changed    = Signal(float)
    sig_iou_changed     = Signal(float)
    sig_save_changed    = Signal(bool)
    sig_centroid_changed= Signal(bool)

    def __init__(self, model_menu: dict, parent=None):
        super().__init__(parent)
        self.setObjectName("SidePanel")
        self.setFixedWidth(280)
        self._model_menu = model_menu
        self._build()

    # ── Build UI ────────────────────────────────────────────────────── #
    def _build(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        # ── Dataset ──────────────────────────────────────────────── #
        root_layout.addWidget(self._section("DATASET"))

        folder_row = QHBoxLayout()
        self.root_edit = QLineEdit()
        self.root_edit.setPlaceholderText("ECP Images folder…")
        self.root_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(64)
        browse_btn.clicked.connect(self._browse_root)
        folder_row.addWidget(self.root_edit)
        folder_row.addWidget(browse_btn)
        root_layout.addLayout(folder_row)

        root_layout.addWidget(self._label("City"))
        self.city_combo = QComboBox()
        self.city_combo.currentTextChanged.connect(
            lambda t: self.sig_city_changed.emit(t) if t else None
        )
        root_layout.addWidget(self.city_combo)

        root_layout.addWidget(self._separator())

        # ── Model ────────────────────────────────────────────────── #
        root_layout.addWidget(self._section("MODEL"))

        root_layout.addWidget(self._label("Select model"))
        self.model_combo = QComboBox()
        self._populate_model_combo()
        root_layout.addWidget(self.model_combo)

        root_layout.addWidget(self._label("Custom .pt path (optional)"))
        self.custom_path = QLineEdit()
        self.custom_path.setPlaceholderText("path\\to\\best.pt")
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_path)
        browse_model_btn = QPushButton("…")
        browse_model_btn.setFixedWidth(32)
        browse_model_btn.clicked.connect(self._browse_model)
        custom_row.addWidget(browse_model_btn)
        root_layout.addLayout(custom_row)

        self.load_btn = QPushButton("Load Model")
        self.load_btn.setObjectName("PrimaryBtn")
        self.load_btn.clicked.connect(self._emit_load)
        root_layout.addWidget(self.load_btn)

        root_layout.addWidget(self._separator())

        # ── Inference settings ────────────────────────────────────── #
        root_layout.addWidget(self._section("INFERENCE"))

        # Conf slider
        conf_row = QHBoxLayout()
        conf_row.addWidget(self._label("Conf"))
        self.conf_val = QLabel("0.35")
        self.conf_val.setAlignment(Qt.AlignRight)
        conf_row.addWidget(self.conf_val)
        root_layout.addLayout(conf_row)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(5, 90)
        self.conf_slider.setValue(35)
        self.conf_slider.valueChanged.connect(self._conf_changed)
        root_layout.addWidget(self.conf_slider)

        # IoU slider
        iou_row = QHBoxLayout()
        iou_row.addWidget(self._label("IoU"))
        self.iou_val = QLabel("0.50")
        self.iou_val.setAlignment(Qt.AlignRight)
        iou_row.addWidget(self.iou_val)
        root_layout.addLayout(iou_row)

        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 90)
        self.iou_slider.setValue(50)
        self.iou_slider.valueChanged.connect(self._iou_changed)
        root_layout.addWidget(self.iou_slider)

        root_layout.addWidget(self._separator())

        # ── Options ───────────────────────────────────────────────── #
        root_layout.addWidget(self._section("OPTIONS"))

        self.save_cb = QCheckBox("Save annotated images")
        self.save_cb.stateChanged.connect(
            lambda s: self.sig_save_changed.emit(bool(s))
        )
        root_layout.addWidget(self.save_cb)

        self.centroid_cb = QCheckBox("Show centroid dot")
        self.centroid_cb.setChecked(True)
        self.centroid_cb.stateChanged.connect(
            lambda s: self.sig_centroid_changed.emit(bool(s))
        )
        root_layout.addWidget(self.centroid_cb)

        root_layout.addWidget(self._separator())

        # ── Navigation ────────────────────────────────────────────── #
        root_layout.addWidget(self._section("NAVIGATION"))

        nav_row = QHBoxLayout()
        self.prev_btn = QPushButton("← Prev")
        self.next_btn = QPushButton("Next →")
        self.prev_btn.clicked.connect(self.sig_prev.emit)
        self.next_btn.clicked.connect(self.sig_next.emit)
        nav_row.addWidget(self.prev_btn)
        nav_row.addWidget(self.next_btn)
        root_layout.addLayout(nav_row)

        self.infer_btn = QPushButton("Run Inference  [Space]")
        self.infer_btn.setObjectName("PrimaryBtn")
        self.infer_btn.clicked.connect(self.sig_infer.emit)
        root_layout.addWidget(self.infer_btn)

        # ── Spacer ─────────────────────────────────────────────────── #
        root_layout.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        )

    # ── Public helpers ───────────────────────────────────────────────── #
    def set_cities(self, cities: list):
        self.city_combo.blockSignals(True)
        self.city_combo.clear()
        self.city_combo.addItems(cities)
        # prefer amsterdam
        if "amsterdam" in cities:
            self.city_combo.setCurrentText("amsterdam")
        self.city_combo.blockSignals(False)
        if cities:
            self.sig_city_changed.emit(self.city_combo.currentText())

    def set_root_text(self, path: str):
        self.root_edit.setText(path)

    def get_conf(self) -> float:
        return self.conf_slider.value() / 100

    def get_iou(self) -> float:
        return self.iou_slider.value() / 100

    def get_save(self) -> bool:
        return self.save_cb.isChecked()

    def get_centroid(self) -> bool:
        return self.centroid_cb.isChecked()

    def set_load_enabled(self, enabled: bool):
        self.load_btn.setEnabled(enabled)

    # ── Internal ─────────────────────────────────────────────────────── #
    def _populate_model_combo(self):
        for display, key in self._model_menu.items():
            if key is None:
                # separator item
                self.model_combo.addItem(display)
                idx = self.model_combo.count() - 1
                self.model_combo.model().item(idx).setEnabled(False)
            else:
                self.model_combo.addItem(display)

        # default to YOLOv8s
        for i in range(self.model_combo.count()):
            if "yolov8s" in self.model_combo.itemText(i).lower():
                self.model_combo.setCurrentIndex(i)
                break

    def _get_selected_key(self) -> str:
        """Return the weights key for the currently selected model."""
        custom = self.custom_path.text().strip()
        if custom:
            return custom
        display = self.model_combo.currentText()
        return self._model_menu.get(display, display)

    def _emit_load(self):
        key = self._get_selected_key()
        if key:
            self.sig_load_model.emit(key)

    def _browse_root(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select ECP Images folder"
        )
        if path:
            self.root_edit.setText(path)
            self.sig_root_changed.emit(path)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select model weights", "", "Model files (*.pt *.onnx)"
        )
        if path:
            self.custom_path.setText(path)

    def _conf_changed(self, value: int):
        f = value / 100
        self.conf_val.setText(f"{f:.2f}")
        self.sig_conf_changed.emit(f)

    def _iou_changed(self, value: int):
        f = value / 100
        self.iou_val.setText(f"{f:.2f}")
        self.sig_iou_changed.emit(f)

    # ── Widget factories ─────────────────────────────────────────────── #
    @staticmethod
    def _section(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("SectionTitle")
        return lbl

    @staticmethod
    def _label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #9d9d9d; font-size: 12px;")
        return lbl

    @staticmethod
    def _separator() -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        return line
