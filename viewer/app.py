"""
app.py — ECP Pedestrian Detection Viewer  (PySide6 · Python 3.11 · Windows)

Run:
    python viewer\\app.py

Keyboard shortcuts:
    Space        Run inference on current image
    Left / Right Navigate images
    Ctrl+L       Load selected model
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QTabWidget, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui  import QKeySequence, QShortcut

from theme    import DARK_THEME
from adapters import MODEL_MENU, get_adapter
from widgets  import ImageCanvas, ControlPanel, StatsPanel, BenchmarkTab


# ── Background worker: model loading ─────────────────────────────────── #
class ModelLoaderWorker(QObject):
    sig_loaded = Signal(str, float)   # device, elapsed_ms
    sig_error  = Signal(str)

    def __init__(self, adapter, weights_path: str):
        super().__init__()
        self._adapter      = adapter
        self._weights_path = weights_path

    def run(self):
        try:
            t0     = time.perf_counter()
            device = self._adapter.load(self._weights_path)
            ms     = (time.perf_counter() - t0) * 1000
            self.sig_loaded.emit(device, ms)
        except Exception as exc:
            self.sig_error.emit(str(exc))


# ── Main window ───────────────────────────────────────────────────────── #
class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ECP Pedestrian Detection Viewer")
        self.resize(1440, 880)
        self.setMinimumSize(1000, 680)

        # ── App state ─────────────────────────────────────────── #
        self._root:           str   = ""
        self._images:         list  = []
        self._idx:            int   = 0
        self._adapter               = None
        self._pending_adapter       = None
        self._last_result:    dict  = {}   # {path, boxes, scores}
        self._show_centroid:  bool  = True
        self._save_drawn:     bool  = False
        self._conf:           float = 0.35
        self._iou:            float = 0.50
        self._loader_thread         = None
        self._loader_worker         = None

        self._build_ui()
        self._connect_signals()
        self._setup_shortcuts()

    # ── UI construction ──────────────────────────────────────────────── #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(8, 8, 8, 4)
        root_layout.setSpacing(6)

        # Tabs
        self.tabs = QTabWidget()
        root_layout.addWidget(self.tabs)

        viewer_widget = QWidget()
        self.tabs.addTab(viewer_widget, "  \U0001f50d  Viewer  ")
        self._build_viewer_tab(viewer_widget)

        self.bench_tab = BenchmarkTab(MODEL_MENU)
        self.tabs.addTab(self.bench_tab, "  \U0001f4ca  Benchmark  ")

        # Status bar
        self._status_lbl = QLabel("Ready — select a dataset folder to begin.")
        self._status_lbl.setObjectName("StatusBar")
        self.statusBar().addWidget(self._status_lbl, 1)
        self.statusBar().setStyleSheet(
            "QStatusBar { background:#252526; border-top:1px solid #3f3f46; }"
        )

    def _build_viewer_tab(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(3)

        self.ctrl   = ControlPanel(MODEL_MENU)
        self.canvas = ImageCanvas()
        self.stats  = StatsPanel()

        splitter.addWidget(self.ctrl)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.stats)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)

    # ── Signal wiring ────────────────────────────────────────────────── #
    def _connect_signals(self):
        c = self.ctrl
        c.sig_root_changed.connect(self._on_root_changed)
        c.sig_city_changed.connect(self._on_city_changed)
        c.sig_load_model.connect(self._on_load_model)
        c.sig_prev.connect(self.prev_image)
        c.sig_next.connect(self.next_image)
        c.sig_infer.connect(self.run_inference)
        c.sig_conf_changed.connect(self._on_conf_changed)
        c.sig_iou_changed.connect(self._on_iou_changed)
        c.sig_save_changed.connect(self._on_save_changed)
        c.sig_centroid_changed.connect(self._on_centroid_changed)

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self).activated.connect(self.run_inference)
        QShortcut(QKeySequence(Qt.Key_Left),  self).activated.connect(self.prev_image)
        QShortcut(QKeySequence(Qt.Key_Right), self).activated.connect(self.next_image)
        QShortcut(QKeySequence("Ctrl+L"),     self).activated.connect(self.ctrl._emit_load)

    # ── Dataset ──────────────────────────────────────────────────────── #
    def _on_root_changed(self, path: str):
        if not os.path.isdir(path):
            self._set_status(f"ERROR: folder not found — {path}")
            return
        cities = sorted(
            d for d in os.listdir(path)
            if os.path.isdir(os.path.join(path, d))
        )
        self._root = path
        self.ctrl.set_cities(cities)
        self.ctrl.set_root_text(path)
        self._set_status(f"Dataset: {path}  ·  {len(cities)} cities found")

    def _on_city_changed(self, city: str):
        if not city or not self._root:
            return
        city_dir = os.path.join(self._root, city)
        exts     = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self._images = sorted(
            os.path.join(city_dir, f)
            for f in os.listdir(city_dir)
            if os.path.splitext(f)[1].lower() in exts
        )
        self._idx         = 0
        self._last_result = {}
        self._show_current_image()
        self._set_status(f"City: {city}  ·  {len(self._images)} images")

    # ── Model loading ────────────────────────────────────────────────── #
    def _on_load_model(self, weights_path: str):
        if self._loader_thread and self._loader_thread.isRunning():
            self._set_status("A model is already loading — please wait.")
            return
        if not weights_path:
            return

        self.ctrl.set_load_enabled(False)
        self._set_status(f"Loading {weights_path} …")

        adapter               = get_adapter(weights_path)
        self._pending_adapter = adapter

        self._loader_thread = QThread(self)
        self._loader_worker = ModelLoaderWorker(adapter, weights_path)
        self._loader_worker.moveToThread(self._loader_thread)

        self._loader_thread.started.connect(self._loader_worker.run)
        self._loader_worker.sig_loaded.connect(
            lambda dev, ms: self._on_model_loaded(adapter, dev, ms)
        )
        self._loader_worker.sig_error.connect(self._on_model_error)
        self._loader_worker.sig_loaded.connect(self._loader_thread.quit)
        self._loader_worker.sig_error.connect(self._loader_thread.quit)

        self._loader_thread.start()

    def _on_model_loaded(self, adapter, device: str, load_ms: float):
        self._adapter = adapter
        info = adapter.get_info()
        self.stats.update_model(info["name"], info["family"], info["size_mb"], device)
        self.ctrl.set_load_enabled(True)
        self._set_status(
            f"\u2714  {info['name']}  ·  {device}  ·  "
            f"{info['size_mb']} MB  ·  loaded in {load_ms:.0f} ms"
        )

    def _on_model_error(self, error: str):
        self.ctrl.set_load_enabled(True)
        self._set_status(f"\u2716  Model load failed: {error}")
        QMessageBox.critical(self, "Model Load Error", error)

    # ── Inference ────────────────────────────────────────────────────── #
    def run_inference(self):
        if self._adapter is None or not self._adapter.is_loaded():
            self._set_status("No model loaded — select a model and click Load Model.")
            return
        if not self._images:
            self._set_status("No images — select a dataset folder and city first.")
            return

        path = self._images[self._idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:
            self._set_status(f"Cannot open image: {exc}")
            return

        t0 = time.perf_counter()
        try:
            boxes, scores, _ = self._adapter.predict(img, self._conf, self._iou)
        except Exception as exc:
            self._set_status(f"Inference error: {exc}")
            return
        inf_ms = (time.perf_counter() - t0) * 1000

        self._last_result = {"path": path, "boxes": boxes, "scores": scores}
        self.canvas.set_detections(boxes, scores, self._show_centroid)
        self.stats.update_inference(inf_ms, len(boxes), self._conf)
        self._set_status(
            f"{os.path.basename(path)}  ·  "
            f"{len(boxes)} detection{'s' if len(boxes) != 1 else ''}  ·  "
            f"{inf_ms:.0f} ms  ·  {1000/inf_ms:.1f} FPS  ·  "
            f"conf\u2265{self._conf:.2f}  IoU {self._iou:.2f}"
        )

        if self._save_drawn:
            self._save_annotated(path, boxes, scores, img)

    # ── Navigation ───────────────────────────────────────────────────── #
    def prev_image(self):
        if not self._images:
            return
        self._idx         = (self._idx - 1) % len(self._images)
        self._last_result = {}
        self._show_current_image()

    def next_image(self):
        if not self._images:
            return
        self._idx         = (self._idx + 1) % len(self._images)
        self._last_result = {}
        self._show_current_image()

    def _show_current_image(self):
        if not self._images:
            return
        path = self._images[self._idx]
        try:
            img = np.array(Image.open(path).convert("RGB"))
        except Exception as exc:
            self._set_status(f"Cannot open image: {exc}")
            return

        h, w = img.shape[:2]
        meta = (
            f"{os.path.basename(path)}   "
            f"[{w} x {h}]   "
            f"({self._idx + 1} / {len(self._images)})"
        )
        self.canvas.set_image(img, meta)

        if self._last_result.get("path") == path:
            self.canvas.set_detections(
                self._last_result["boxes"],
                self._last_result["scores"],
                self._show_centroid,
            )

        self._set_status(meta)

    # ── Control panel handlers ────────────────────────────────────────── #
    def _on_conf_changed(self, val: float):
        self._conf = val

    def _on_iou_changed(self, val: float):
        self._iou = val

    def _on_save_changed(self, val: bool):
        self._save_drawn = val

    def _on_centroid_changed(self, val: bool):
        self._show_centroid = val
        if self._last_result:
            self.canvas.set_detections(
                self._last_result["boxes"],
                self._last_result["scores"],
                self._show_centroid,
            )

    # ── Save annotated ───────────────────────────────────────────────── #
    def _save_annotated(self, path: str, boxes, scores, img_rgb):
        try:
            from PIL import ImageDraw
            pil  = Image.fromarray(img_rgb)
            draw = ImageDraw.Draw(pil)
            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
                draw.text((x1 + 2, max(0, y1 - 14)), f"person {score:.2f}", fill=(0, 255, 0))
                if self._show_centroid:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    draw.ellipse([cx-3, cy-3, cx+3, cy+3], fill=(0, 255, 0))
            out_dir = os.path.join(os.path.dirname(path), "_detections")
            os.makedirs(out_dir, exist_ok=True)
            pil.save(os.path.join(out_dir, os.path.basename(path)), quality=95)
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")

    # ── Status bar ────────────────────────────────────────────────────── #
    def _set_status(self, msg: str):
        self._status_lbl.setText(msg)


# ── Entry point ──────────────────────────────────────────────────────── #
def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("ECP Pedestrian Viewer")
    app.setStyle("Fusion")      # consistent baseline on Windows 11
    app.setStyleSheet(DARK_THEME)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
