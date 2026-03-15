@echo off
REM ── Launch ECP Pedestrian Viewer ─────────────────────────────────────────
if not exist ".venv\Scripts\activate.bat" (
    echo  Virtual environment not found. Run setup.bat first.
    pause
    exit /b 1
)
call .venv\Scripts\activate.bat
python viewer\app.py
