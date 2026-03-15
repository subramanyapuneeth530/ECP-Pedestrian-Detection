@echo off
REM ── ECP Pedestrian Viewer — Windows Setup ────────────────────────────────
echo.
echo  ECP Pedestrian Detection Viewer — Setup
echo  =========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python not found. Install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo  [1/4] Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate.bat

echo.
echo  [2/4] Upgrading pip...
python -m pip install --upgrade pip --quiet

echo.
echo  [3/4] Installing PyTorch (CUDA 12.1)...
echo        If you need a different CUDA version, edit this file.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet

echo.
echo  [4/4] Installing remaining dependencies...
pip install -r requirements.txt --quiet

echo.
echo  =========================================
echo  Setup complete!
echo.
echo  To run the viewer:
echo    .venv\Scripts\activate
echo    python viewer\app.py
echo.
pause
