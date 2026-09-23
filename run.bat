@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === Local Studio ===
if not exist outputs mkdir outputs
if not exist models mkdir models
REM Fast path: skip pip install if deps already work (reinstalling numpy would break torch!)
python -c "import gradio, cv2, PIL, numpy" 2>nul
if errorlevel 1 (
  echo Installing dependencies first time...
  python -m pip install -r requirements-min.txt
  if errorlevel 1 (
    echo FAILED to install dependencies. Check internet and Python.
    pause
    exit /b 1
  )
) else (
  echo Dependencies OK, skipping install.
)
echo.
echo Starting... wait 30-60 sec, then open: http://127.0.0.1:7860
echo Do NOT close this black window while using the studio.
python -u app.py
pause
