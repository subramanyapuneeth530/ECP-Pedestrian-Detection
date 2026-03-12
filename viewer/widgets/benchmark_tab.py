"""
widgets/benchmark_tab.py

Benchmark tab — runs all models on a sample of ECP val images
and shows results in a sortable table inside the viewer.

Runs inference in a background QThread so the UI stays responsive.
"""

import os, time
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QFileDialog, QFrame
)
from PySide6.QtCore    import Qt, QThread, Signal, QObject
from PySide6.QtGui     import QColor


# ── Worker (runs in background thread) ──────────────────────────────── #
class BenchmarkWorker(QObject):
    sig_progress = Signal(int, int, str)      # current, total, model_name
    sig_row_done = Signal(dict)               # one result row
    sig_finished = Signal()
    sig_error    = Signal(str)

    def __init__(self, model_menu, images, conf, iou):
        super().__init__()
        self._model_menu = model_menu
        self._images     = images
        self._conf       = conf
        self._iou        = iou
        self._abort      = False

    def abort(self):
        self._abort = True

    def run(self):
        from adapters import get_adapter
        from PIL import Image

        runnable = [
            (name, key)
            for name, key in self._model_menu.items()
            if key is not None
        ]
        total = len(runnable)

        for i, (display, key) in enumerate(runnable):
            if self._abort:
                break

            self.sig_progress.emit(i + 1, total, display)

            adapter = get_adapter(key)
            try:
                device = adapter.load(key)
            except Exception as e:
                self.sig_error.emit(f"Failed to load {display}: {e}")
                continue

            times, dets = [], []
            for img_path in self._images:
                if self._abort:
                    break
                try:
                    img = np.array(Image.open(img_path).convert("RGB"))
                except Exception:
                    continue
                t0 = time.perf_counter()
                boxes, scores, _ = adapter.predict(img, self._conf, self._iou)
                times.append((time.perf_counter() - t0) * 1000)
                dets.append(len(boxes))

            if not times:
                continue

            avg_ms = float(np.mean(times))
            self.sig_row_done.emit({
                "model":    display,
                "family":   adapter.family,
                "size_mb":  adapter.model_size_mb,
                "avg_ms":   round(avg_ms, 1),
                "fps":      round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
                "avg_dets": round(float(np.mean(dets)), 1),
                "device":   device,
                "n_images": len(times),
            })

        self.sig_finished.emit()


# ── Tab widget ───────────────────────────────────────────────────────── #
class BenchmarkTab(QWidget):
    COLUMNS = ["Model", "Family", "Size (MB)", "Avg ms", "FPS",
               "Avg dets", "Device", "Images"]

    def __init__(self, model_menu: dict, parent=None):
        super().__init__(parent)
        self._model_menu = model_menu
        self._thread     = None
        self._worker     = None
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ── Config row ───────────────────────────────────────────── #
        cfg = QHBoxLayout()

        cfg.addWidget(QLabel("Val images folder:"))
        self.val_edit = QLineEdit()
        self.val_edit.setPlaceholderText("ECP\\Val\\Images")
        cfg.addWidget(self.val_edit)
        browse = QPushButton("Browse")
        browse.setFixedWidth(70)
        browse.clicked.connect(self._browse_val)
        cfg.addWidget(browse)

        cfg.addWidget(QLabel("Max images:"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(10, 5000)
        self.max_spin.setValue(100)
        self.max_spin.setFixedWidth(70)
        cfg.addWidget(self.max_spin)

        layout.addLayout(cfg)

        # ── Action row ───────────────────────────────────────────── #
        action = QHBoxLayout()
        self.run_btn = QPushButton("Run Benchmark")
        self.run_btn.setObjectName("PrimaryBtn")
        self.run_btn.setFixedWidth(140)
        self.run_btn.clicked.connect(self._run)
        action.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedWidth(70)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        action.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setFixedWidth(90)
        self.export_btn.clicked.connect(self._export)
        action.addWidget(self.export_btn)

        action.addStretch()
        layout.addLayout(action)

        # ── Progress ─────────────────────────────────────────────── #
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(18)
        layout.addWidget(self.progress)

        self.progress_lbl = QLabel("")
        self.progress_lbl.setStyleSheet("color: #9d9d9d; font-size: 12px;")
        layout.addWidget(self.progress_lbl)

        # ── Results table ────────────────────────────────────────── #
        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

    # ── Slots ────────────────────────────────────────────────────────── #
    def _browse_val(self):
        path = QFileDialog.getExistingDirectory(self, "Select Val Images folder")
        if path:
            self.val_edit.setText(path)

    def _run(self):
        val_root = self.val_edit.text().strip()
        if not os.path.isdir(val_root):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Select a valid Val Images folder first.")
            return

        images = self._collect_images(val_root, self.max_spin.value())
        if not images:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "No images found in that folder.")
            return

        # Reset table
        self.table.setRowCount(0)

        # Thread setup
        self._thread = QThread()
        self._worker = BenchmarkWorker(
            self._model_menu, images, conf=0.35, iou=0.50
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_row_done.connect(self._on_row)
        self._worker.sig_error.connect(self._on_error)
        self._worker.sig_finished.connect(self._on_finished)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setMaximum(
            len([k for k in self._model_menu.values() if k is not None])
        )
        self.progress.setValue(0)

        self._thread.start()

    def _stop(self):
        if self._worker:
            self._worker.abort()

    def _on_progress(self, current: int, total: int, name: str):
        self.progress.setValue(current)
        self.progress_lbl.setText(f"Running {current}/{total}: {name}")

    def _on_row(self, row: dict):
        r = self.table.rowCount()
        self.table.insertRow(r)

        values = [
            row["model"], row["family"], str(row["size_mb"]),
            str(row["avg_ms"]), str(row["fps"]),
            str(row["avg_dets"]), row["device"], str(row["n_images"]),
        ]
        for c, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, c, item)

        # Highlight fastest row green
        if row["fps"] > 0:
            self._highlight_best()

    def _on_error(self, msg: str):
        print(f"[BENCHMARK ERROR] {msg}")

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_lbl.setText("Benchmark complete.")
        self._thread.quit()

    def _highlight_best(self):
        """Highlight the row with the highest FPS in green."""
        best_fps = -1
        best_row = -1
        fps_col  = self.COLUMNS.index("FPS")

        for r in range(self.table.rowCount()):
            item = self.table.item(r, fps_col)
            if item:
                try:
                    fps = float(item.text())
                    if fps > best_fps:
                        best_fps = fps
                        best_row = r
                except ValueError:
                    pass

        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if item:
                    if r == best_row:
                        item.setForeground(QColor("#4ec94e"))
                    else:
                        item.setForeground(QColor("#d4d4d4"))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "benchmark_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        rows = []
        rows.append(",".join(self.COLUMNS))
        for r in range(self.table.rowCount()):
            row_data = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                row_data.append(item.text() if item else "")
            rows.append(",".join(row_data))
        with open(path, "w") as f:
            f.write("\n".join(rows))

    @staticmethod
    def _collect_images(root: str, max_n: int) -> list:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = []
        for city in sorted(os.scandir(root), key=lambda e: e.name):
            if not city.is_dir():
                continue
            for f in sorted(os.scandir(city.path), key=lambda e: e.name):
                if os.path.splitext(f.name)[1].lower() in exts:
                    imgs.append(f.path)
        return imgs[:max_n]
