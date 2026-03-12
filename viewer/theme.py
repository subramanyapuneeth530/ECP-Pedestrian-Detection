"""
theme.py — VS Code-inspired dark theme for the ECP Viewer.
Apply with: app.setStyleSheet(DARK_THEME)
"""

# Colour palette
BG_DARK    = "#1e1e1e"   # main background
BG_PANEL   = "#252526"   # sidebar / panel background
BG_WIDGET  = "#2d2d30"   # input / combobox background
BG_HOVER   = "#3e3e42"   # hover state
ACCENT     = "#0078d4"   # blue accent (buttons, selections)
ACCENT_HOV = "#1084d8"   # accent hover
TEXT_PRI   = "#d4d4d4"   # primary text
TEXT_SEC   = "#9d9d9d"   # secondary / label text
TEXT_DIS   = "#6d6d6d"   # disabled text
BORDER     = "#3f3f46"   # subtle border
GREEN      = "#4ec94e"   # detection boxes / positive
ORANGE     = "#ce9178"   # warnings
RED        = "#f44747"   # errors
SEP        = "#3f3f46"   # separator lines

DARK_THEME = f"""
/* ── Global ─────────────────────────────────────────────────────── */
QMainWindow, QDialog, QWidget {{
    background-color: {BG_DARK};
    color: {TEXT_PRI};
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}}

/* ── Panels / Frames ─────────────────────────────────────────────── */
QFrame#SidePanel, QFrame#BottomPanel {{
    background-color: {BG_PANEL};
    border: 1px solid {BORDER};
    border-radius: 6px;
}}

/* ── Labels ──────────────────────────────────────────────────────── */
QLabel {{
    color: {TEXT_PRI};
    background: transparent;
}}
QLabel#SectionTitle {{
    color: {TEXT_SEC};
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 4px 0px 2px 0px;
}}
QLabel#StatValue {{
    color: {TEXT_PRI};
    font-size: 22px;
    font-weight: bold;
}}
QLabel#StatLabel {{
    color: {TEXT_SEC};
    font-size: 11px;
}}
QLabel#StatusBar {{
    color: {TEXT_SEC};
    font-size: 12px;
    padding: 4px 8px;
    background-color: {BG_PANEL};
    border-top: 1px solid {BORDER};
}}

/* ── Buttons ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 13px;
}}
QPushButton:hover {{
    background-color: {BG_HOVER};
    border-color: {ACCENT};
}}
QPushButton:pressed {{
    background-color: {ACCENT};
    color: white;
}}
QPushButton:disabled {{
    color: {TEXT_DIS};
    border-color: {BG_WIDGET};
}}
QPushButton#PrimaryBtn {{
    background-color: {ACCENT};
    color: white;
    border: none;
    font-weight: bold;
}}
QPushButton#PrimaryBtn:hover {{
    background-color: {ACCENT_HOV};
}}
QPushButton#PrimaryBtn:disabled {{
    background-color: {BG_WIDGET};
    color: {TEXT_DIS};
}}

/* ── ComboBox ────────────────────────────────────────────────────── */
QComboBox {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 13px;
}}
QComboBox:hover {{
    border-color: {ACCENT};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {TEXT_SEC};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    selection-background-color: {ACCENT};
    selection-color: white;
    outline: none;
}}

/* ── LineEdit ────────────────────────────────────────────────────── */
QLineEdit {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 8px;
    font-size: 13px;
}}
QLineEdit:focus {{
    border-color: {ACCENT};
}}

/* ── Sliders ─────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    background-color: {BG_WIDGET};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background-color: {ACCENT};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}
QSlider::handle:horizontal:hover {{
    background-color: {ACCENT_HOV};
}}
QSlider::sub-page:horizontal {{
    background-color: {ACCENT};
    border-radius: 2px;
}}

/* ── CheckBox ────────────────────────────────────────────────────── */
QCheckBox {{
    color: {TEXT_PRI};
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER};
    border-radius: 3px;
    background-color: {BG_WIDGET};
}}
QCheckBox::indicator:checked {{
    background-color: {ACCENT};
    border-color: {ACCENT};
}}

/* ── TabWidget ───────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background-color: {BG_DARK};
}}
QTabBar::tab {{
    background-color: {BG_PANEL};
    color: {TEXT_SEC};
    padding: 8px 20px;
    border: 1px solid {BORDER};
    border-bottom: none;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {BG_DARK};
    color: {TEXT_PRI};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{
    color: {TEXT_PRI};
}}

/* ── ScrollBar ───────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background-color: {BG_DARK};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background-color: {BG_HOVER};
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::handle:vertical:hover {{
    background-color: {TEXT_DIS};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}

/* ── TextEdit / Log ──────────────────────────────────────────────── */
QTextEdit {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 4px;
    font-family: "Consolas", monospace;
    font-size: 12px;
    padding: 4px;
}}

/* ── Table (benchmark results) ───────────────────────────────────── */
QTableWidget {{
    background-color: {BG_WIDGET};
    color: {TEXT_PRI};
    border: 1px solid {BORDER};
    border-radius: 4px;
    gridline-color: {BORDER};
    outline: none;
}}
QTableWidget::item {{
    padding: 6px 10px;
}}
QTableWidget::item:selected {{
    background-color: {ACCENT};
    color: white;
}}
QHeaderView::section {{
    background-color: {BG_PANEL};
    color: {TEXT_SEC};
    border: none;
    border-bottom: 1px solid {BORDER};
    border-right: 1px solid {BORDER};
    padding: 6px 10px;
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
}}

/* ── ProgressBar ─────────────────────────────────────────────────── */
QProgressBar {{
    background-color: {BG_WIDGET};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_PRI};
    font-size: 12px;
    height: 18px;
}}
QProgressBar::chunk {{
    background-color: {ACCENT};
    border-radius: 3px;
}}

/* ── Splitter ────────────────────────────────────────────────────── */
QSplitter::handle {{
    background-color: {BORDER};
}}
QSplitter::handle:horizontal {{
    width: 2px;
}}
QSplitter::handle:vertical {{
    height: 2px;
}}

/* ── Separator ───────────────────────────────────────────────────── */
QFrame[frameShape="4"],
QFrame[frameShape="5"] {{
    color: {SEP};
    background-color: {SEP};
    border: none;
    max-height: 1px;
}}
"""
