"""
widgets/benchmark_tab.py

Benchmark tab — runs all models on ECP val images and measures:
  - Speed  : avg ms, FPS
  - Accuracy: mAP@0.5, mAP@0.5:0.95, Precision, Recall

Ground truth is loaded from ECP JSON annotation files.
ECP identity classes used for evaluation: "pedestrian" only.
"""

import os, time, json
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QSpinBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QFileDialog, QFrame, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui  import QColor


# ── IoU and mAP helpers ──────────────────────────────────────────────── #

def compute_iou(box_a, box_b):
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls, precisions):
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.arange(0.0, 1.1, 0.1):
        p = precisions[recalls >= t]
        ap += (np.max(p) if len(p) > 0 else 0.0)
    return ap / 11.0


def evaluate_detections(all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh=0.5):
    """
    Compute AP at a given IoU threshold across all images.

    all_pred_boxes  : list of lists [[x1,y1,x2,y2], ...]  per image
    all_pred_scores : list of lists [score, ...]           per image
    all_gt_boxes    : list of lists [[x1,y1,x2,y2], ...]  per image
    """
    # Flatten all predictions with image index
    preds = []
    for img_idx, (boxes, scores) in enumerate(zip(all_pred_boxes, all_pred_scores)):
        for box, score in zip(boxes, scores):
            preds.append((score, img_idx, box))

    # Sort by descending confidence
    preds.sort(key=lambda x: -x[0])

    n_gt = sum(len(g) for g in all_gt_boxes)
    if n_gt == 0:
        return 0.0, 0.0, 0.0

    matched = [set() for _ in all_gt_boxes]   # track matched GTs per image
    tp_list = []
    fp_list = []

    for score, img_idx, pred_box in preds:
        gt_boxes = all_gt_boxes[img_idx]
        best_iou = 0.0
        best_j   = -1
        for j, gt_box in enumerate(gt_boxes):
            iou = compute_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_j   = j

        if best_iou >= iou_thresh and best_j not in matched[img_idx]:
            tp_list.append(1)
            fp_list.append(0)
            matched[img_idx].add(best_j)
        else:
            tp_list.append(0)
            fp_list.append(1)

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)

    recalls    = tp_cum / n_gt
    precisions = tp_cum / (tp_cum + fp_cum + 1e-9)

    ap        = compute_ap(recalls, precisions)
    precision = float(precisions[-1]) if len(precisions) else 0.0
    recall    = float(recalls[-1])    if len(recalls)    else 0.0

    return ap, precision, recall


def compute_map(all_pred_boxes, all_pred_scores, all_gt_boxes):
    """Compute mAP@0.5 and mAP@0.5:0.95."""
    ap50 = evaluate_detections(
        all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh=0.5
    )[0]

    aps = []
    for thresh in np.arange(0.5, 1.0, 0.05):
        ap, _, _ = evaluate_detections(
            all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh=thresh
        )
        aps.append(ap)
    map5095 = float(np.mean(aps))

    _, precision, recall = evaluate_detections(
        all_pred_boxes, all_pred_scores, all_gt_boxes, iou_thresh=0.5
    )
    return ap50, map5095, precision, recall


# ── Ground truth loader ──────────────────────────────────────────────── #

EVAL_IDENTITIES = {"pedestrian"}   # only full pedestrians for eval


def load_gt_boxes(label_path: str) -> list:
    """
    Load ground truth boxes from an ECP JSON file.
    Returns list of [x1, y1, x2, y2] for pedestrian identities only.
    """
    try:
        with open(label_path, "r") as f:
            ann = json.load(f)
        boxes = []
        for child in ann.get("children", []):
            if child.get("identity") in EVAL_IDENTITIES:
                x0 = float(child["x0"])
                y0 = float(child["y0"])
                x1 = float(child["x1"])
                y1 = float(child["y1"])
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1, y1])
        return boxes
    except Exception:
        return []


def find_label_path(img_path: str, labels_root: str) -> str | None:
    """
    Given an image path and labels root, find the matching JSON file.
    ECP structure: labels_root/city/imagename.json
    """
    img_name  = os.path.splitext(os.path.basename(img_path))[0]
    city_name = os.path.basename(os.path.dirname(img_path))
    label_path = os.path.join(labels_root, city_name, img_name + ".json")
    return label_path if os.path.exists(label_path) else None


# ── Worker ───────────────────────────────────────────────────────────── #

class BenchmarkWorker(QObject):
    sig_progress = Signal(int, int, str)
    sig_row_done = Signal(dict)
    sig_finished = Signal()
    sig_error    = Signal(str)

    def __init__(self, model_menu, images, labels_root, conf, iou, eval_mode):
        super().__init__()
        self._model_menu  = model_menu
        self._images      = images
        self._labels_root = labels_root   # None if speed-only
        self._conf        = conf
        self._iou         = iou
        self._eval_mode   = eval_mode     # True = compute mAP
        self._abort       = False

    def abort(self):
        self._abort = True

    def run(self):
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            print(f"[BENCHMARK] {i+1}/{total}: {display}", flush=True)

            # Skip fine-tuned if weights don't exist
            if key.endswith(".pt") and "/" in key and not os.path.exists(key):
                self.sig_error.emit(f"Skipping {display} — weights not found")
                continue

            try:
                adapter = get_adapter(key)
                print(f"  [DEBUG] adapter={type(adapter).__name__} family={adapter.family} key={key}", flush=True)
                device  = adapter.load(key)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.sig_error.emit(f"Load failed [{display}]: {type(e).__name__}: {e}")
                continue

            times          = []
            all_pred_boxes = []
            all_pred_scores= []
            all_gt_boxes   = []

            for img_path in self._images:
                if self._abort:
                    break
                try:
                    img = np.array(Image.open(img_path).convert("RGB"))
                except Exception:
                    continue

                t0 = time.perf_counter()
                try:
                    boxes, scores, _ = adapter.predict(img, self._conf, self._iou)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.sig_error.emit(f"Predict failed [{display}]: {type(e).__name__}: {e}")
                    break
                times.append((time.perf_counter() - t0) * 1000)

                all_pred_boxes.append(boxes)
                all_pred_scores.append(scores)

                # Load GT if eval mode
                if self._eval_mode and self._labels_root:
                    lbl = find_label_path(img_path, self._labels_root)
                    if lbl is None and len(all_gt_boxes) == 0:
                        print(f"  [DEBUG GT] img={img_path}", flush=True)
                        print(f"  [DEBUG GT] labels_root={self._labels_root}", flush=True)
                        city = os.path.basename(os.path.dirname(img_path))
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        expected = os.path.join(self._labels_root, city, stem + ".json")
                        print(f"  [DEBUG GT] expected label={expected}", flush=True)
                        print(f"  [DEBUG GT] exists={os.path.exists(expected)}", flush=True)
                    all_gt_boxes.append(load_gt_boxes(lbl) if lbl else [])

            if not times:
                self.sig_error.emit(f"No results for {display} — skipped")
                continue

            avg_ms = float(np.mean(times))

            result = {
                "model":    display,
                "family":   adapter.family,
                "size_mb":  adapter.model_size_mb,
                "avg_ms":   round(avg_ms, 1),
                "fps":      round(1000 / avg_ms, 1) if avg_ms > 0 else 0,
                "avg_dets": round(float(np.mean([len(b) for b in all_pred_boxes])), 1),
                "device":   device,
                "n_images": len(times),
                "map50":    None,
                "map5095":  None,
                "precision":None,
                "recall":   None,
            }

            # Compute mAP if eval mode and we have GT
            if self._eval_mode and all_gt_boxes:
                n_with_gt = sum(1 for g in all_gt_boxes if g is not None)
                if n_with_gt > 0:
                    map50, map5095, prec, rec = compute_map(
                        all_pred_boxes, all_pred_scores, all_gt_boxes
                    )
                    result["map50"]     = round(map50    * 100, 1)
                    result["map5095"]   = round(map5095  * 100, 1)
                    result["precision"] = round(prec     * 100, 1)
                    result["recall"]    = round(rec      * 100, 1)

            self.sig_row_done.emit(result)
            print(f"  -> {result['avg_ms']} ms | {result['fps']} FPS | "
                  f"mAP@50={result['map50']}%", flush=True)

        self.sig_finished.emit()


# ── Tab widget ───────────────────────────────────────────────────────── #

class BenchmarkTab(QWidget):

    COLUMNS_SPEED = ["Model", "Family", "Size (MB)", "Avg ms", "FPS", "Avg dets", "Device", "N"]
    COLUMNS_EVAL  = ["Model", "Family", "Size (MB)", "Avg ms", "FPS",
                     "mAP@50 (%)", "mAP@50:95 (%)", "Precision (%)", "Recall (%)", "Device", "N"]

    def __init__(self, model_menu: dict, parent=None):
        super().__init__(parent)
        self._model_menu = model_menu
        self._thread     = None
        self._worker     = None
        self._eval_mode  = False
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        # ── Images folder row ────────────────────────────────── #
        img_row = QHBoxLayout()
        img_row.addWidget(QLabel("Images folder:"))
        self.val_edit = QLineEdit()
        self.val_edit.setPlaceholderText("ECP\\val\\img  (contains city subfolders)")
        img_row.addWidget(self.val_edit)
        browse_img = QPushButton("Browse")
        browse_img.setFixedWidth(70)
        browse_img.clicked.connect(self._browse_val)
        img_row.addWidget(browse_img)
        layout.addLayout(img_row)

        # ── Labels folder row ─────────────────────────────────── #
        lbl_row = QHBoxLayout()
        self._eval_cb = QCheckBox("Evaluate accuracy (mAP)")
        self._eval_cb.stateChanged.connect(self._toggle_eval)
        lbl_row.addWidget(self._eval_cb)
        lbl_row.addSpacing(16)
        self._lbl_edit = QLineEdit()
        self._lbl_edit.setPlaceholderText("ECP\\val\\labels  (required for mAP)")
        self._lbl_edit.setEnabled(False)
        lbl_row.addWidget(self._lbl_edit)
        browse_lbl = QPushButton("Browse")
        browse_lbl.setFixedWidth(70)
        browse_lbl.setObjectName("browse_lbl")
        browse_lbl.setEnabled(False)
        browse_lbl.clicked.connect(self._browse_labels)
        lbl_row.addWidget(browse_lbl)
        layout.addLayout(lbl_row)
        self._browse_lbl_btn = browse_lbl

        # ── Config row ────────────────────────────────────────── #
        cfg_row = QHBoxLayout()
        cfg_row.addWidget(QLabel("Max images per model:"))
        self.max_spin = QSpinBox()
        self.max_spin.setRange(10, 5000)
        self.max_spin.setValue(100)
        self.max_spin.setFixedWidth(80)
        cfg_row.addWidget(self.max_spin)
        cfg_row.addStretch()
        layout.addLayout(cfg_row)

        # ── Action row ────────────────────────────────────────── #
        action_row = QHBoxLayout()
        self.run_btn = QPushButton("Run Benchmark")
        self.run_btn.setObjectName("PrimaryBtn")
        self.run_btn.setFixedWidth(150)
        self.run_btn.clicked.connect(self._run)
        action_row.addWidget(self.run_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFixedWidth(70)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        action_row.addWidget(self.stop_btn)

        self.export_btn = QPushButton("Export CSV")
        self.export_btn.setFixedWidth(100)
        self.export_btn.clicked.connect(self._export)
        action_row.addWidget(self.export_btn)

        self.export_xlsx_btn = QPushButton("Export Excel")
        self.export_xlsx_btn.setFixedWidth(110)
        self.export_xlsx_btn.setObjectName("PrimaryBtn")
        self.export_xlsx_btn.clicked.connect(self._export_excel)
        action_row.addWidget(self.export_xlsx_btn)

        action_row.addStretch()
        layout.addLayout(action_row)

        # ── Progress ──────────────────────────────────────────── #
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setFixedHeight(18)
        layout.addWidget(self.progress)

        self.progress_lbl = QLabel("")
        self.progress_lbl.setStyleSheet("color: #9d9d9d; font-size: 12px;")
        layout.addWidget(self.progress_lbl)

        # ── Results table ─────────────────────────────────────── #
        self.table = QTableWidget(0, len(self.COLUMNS_SPEED))
        self.table.setHorizontalHeaderLabels(self.COLUMNS_SPEED)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSortingEnabled(True)
        layout.addWidget(self.table)

        # Note about evaluation
        note = QLabel(
            "ℹ  Speed-only mode uses any image folder.  "
            "Accuracy mode (mAP) requires ECP val images + labels folders."
        )
        note.setStyleSheet("color: #9d9d9d; font-size: 11px;")
        layout.addWidget(note)

    # ── Slots ─────────────────────────────────────────────────────────── #

    def _toggle_eval(self, state):
        self._eval_mode = bool(state)
        self._lbl_edit.setEnabled(self._eval_mode)
        self._browse_lbl_btn.setEnabled(self._eval_mode)
        cols = self.COLUMNS_EVAL if self._eval_mode else self.COLUMNS_SPEED
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setRowCount(0)

    def _browse_val(self):
        path = QFileDialog.getExistingDirectory(self, "Select Images folder (contains city subfolders)")
        if path:
            self.val_edit.setText(path)

    def _browse_labels(self):
        path = QFileDialog.getExistingDirectory(self, "Select Labels folder (contains city subfolders)")
        if path:
            self._lbl_edit.setText(path)

    def _run(self):
        val_root = self.val_edit.text().strip()
        if not os.path.isdir(val_root):
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", "Select a valid Images folder first.")
            return

        images = self._collect_images(val_root, self.max_spin.value())
        if not images:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(
                self, "No images found",
                "No images found. Make sure the folder contains city subfolders with images inside."
            )
            return

        labels_root = None
        if self._eval_mode:
            labels_root = self._lbl_edit.text().strip()
            if not os.path.isdir(labels_root):
                from PySide6.QtWidgets import QMessageBox
                QMessageBox.warning(self, "Error",
                    "Select a valid Labels folder for accuracy evaluation,\n"
                    "or uncheck 'Evaluate accuracy' for speed-only mode.")
                return

        self.table.setRowCount(0)

        n_models = len([k for k in self._model_menu.values() if k is not None])
        self._thread = QThread(self)
        self._worker = BenchmarkWorker(
            self._model_menu, images, labels_root,
            conf=0.35, iou=0.50, eval_mode=self._eval_mode
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_row_done.connect(self._on_row)
        self._worker.sig_error.connect(self._on_error)
        self._worker.sig_finished.connect(self._on_finished)
        self._worker.sig_finished.connect(self._thread.quit)

        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress.setVisible(True)
        self.progress.setMaximum(n_models)
        self.progress.setValue(0)
        self.progress_lbl.setText(f"Starting benchmark on {len(images)} images…")

        self._thread.start()

    def _stop(self):
        if self._worker:
            self._worker.abort()

    def _on_progress(self, current: int, total: int, name: str):
        self.progress.setValue(current)
        self.progress_lbl.setText(f"Running {current}/{total}: {name}")

    def _on_row(self, row: dict):
        cols = self.COLUMNS_EVAL if self._eval_mode else self.COLUMNS_SPEED
        r = self.table.rowCount()
        self.table.insertRow(r)

        if self._eval_mode:
            values = [
                row["model"], row["family"], str(row["size_mb"]),
                str(row["avg_ms"]), str(row["fps"]),
                str(row["map50"])     if row["map50"]     is not None else "—",
                str(row["map5095"])   if row["map5095"]   is not None else "—",
                str(row["precision"]) if row["precision"] is not None else "—",
                str(row["recall"])    if row["recall"]    is not None else "—",
                row["device"], str(row["n_images"]),
            ]
        else:
            values = [
                row["model"], row["family"], str(row["size_mb"]),
                str(row["avg_ms"]), str(row["fps"]),
                str(row["avg_dets"]), row["device"], str(row["n_images"]),
            ]

        self.table.setSortingEnabled(False)
        for c, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(r, c, item)
        self.table.setSortingEnabled(True)

        self._highlight_best()

    def _on_error(self, msg: str):
        import sys
        print(f"[BENCHMARK ERROR] {msg}", flush=True, file=sys.stderr)
        self.progress_lbl.setText(f"⚠  {msg}")

    def _on_finished(self):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_lbl.setText("✔  Benchmark complete.")

    def _highlight_best(self):
        """Highlight best FPS row in green, best mAP row in blue."""
        cols = self.COLUMNS_EVAL if self._eval_mode else self.COLUMNS_SPEED
        fps_col  = cols.index("FPS")
        map_col  = cols.index("mAP@50 (%)") if self._eval_mode else None

        best_fps = -1
        best_map = -1
        best_fps_row = -1
        best_map_row = -1

        for r in range(self.table.rowCount()):
            try:
                fps = float(self.table.item(r, fps_col).text())
                if fps > best_fps:
                    best_fps = fps
                    best_fps_row = r
            except (ValueError, AttributeError):
                pass
            if map_col is not None:
                try:
                    m = float(self.table.item(r, map_col).text())
                    if m > best_map:
                        best_map = m
                        best_map_row = r
                except (ValueError, AttributeError):
                    pass

        for r in range(self.table.rowCount()):
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                if not item:
                    continue
                if r == best_map_row and map_col is not None:
                    item.setForeground(QColor("#9cdcfe"))   # blue = best accuracy
                elif r == best_fps_row:
                    item.setForeground(QColor("#4ec94e"))   # green = best speed
                else:
                    item.setForeground(QColor("#d4d4d4"))

    def _export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "benchmark_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        cols = self.COLUMNS_EVAL if self._eval_mode else self.COLUMNS_SPEED
        rows = [",".join(cols)]
        for r in range(self.table.rowCount()):
            row_data = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                row_data.append(item.text() if item else "")
            rows.append(",".join(row_data))
        with open(path, "w") as f:
            f.write("\n".join(rows))
        print(f"[BENCHMARK] CSV saved: {path}", flush=True)

    def _export_excel(self):
        import sys, tempfile
        from datetime import datetime

        # First save a temp CSV
        ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_csv  = os.path.join(tempfile.gettempdir(), f"benchmark_{ts}.csv")

        cols = self.COLUMNS_EVAL if self._eval_mode else self.COLUMNS_SPEED
        rows = [",".join(cols)]
        for r in range(self.table.rowCount()):
            row_data = []
            for c in range(self.table.columnCount()):
                item = self.table.item(r, c)
                row_data.append(item.text() if item else "")
            rows.append(",".join(row_data))

        if len(rows) <= 1:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "No data", "Run a benchmark first before exporting.")
            return

        with open(tmp_csv, "w", encoding="utf-8") as f:
            f.write("\n".join(rows))

        # Ask where to save the Excel file
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Excel",
            f"benchmark_{ts}.xlsx",
            "Excel files (*.xlsx)"
        )
        if not path:
            os.remove(tmp_csv)
            return

        # Run the exporter
        try:
            exporter_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "benchmark", "export_excel.py"
            )
            if not os.path.exists(exporter_path):
                raise FileNotFoundError(f"export_excel.py not found at {exporter_path}")

            import importlib.util
            spec   = importlib.util.spec_from_file_location("export_excel", exporter_path)
            mod    = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            mod.build_workbook(tmp_csv, path)

            print(f"[BENCHMARK] Excel saved: {path}", flush=True)
            self.progress_lbl.setText(f"✔  Excel saved: {os.path.basename(path)}")

            # Open the file
            import subprocess
            subprocess.Popen(["start", "", path], shell=True)

        except Exception as e:
            import traceback
            traceback.print_exc()
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export Error", str(e))
        finally:
            try:
                os.remove(tmp_csv)
            except Exception:
                pass

    @staticmethod
    def _collect_images(root: str, max_n: int) -> list:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        imgs = []
        for entry in sorted(os.scandir(root), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            for f in sorted(os.scandir(entry.path), key=lambda e: e.name):
                if os.path.splitext(f.name)[1].lower() in exts:
                    imgs.append(f.path)
        return imgs[:max_n]
