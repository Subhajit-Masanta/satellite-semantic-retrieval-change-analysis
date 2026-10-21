@echo off
rem Starts the demo app. Double-click this file, or run it from a terminal.
cd /d "%~dp0"
set PYTHONPATH=%~dp0env\libs_gpu;%~dp0env\libs
set HF_HUB_OFFLINE=1
if not exist demo_index\vectors.npy python demo\build_index.py
python -m streamlit run demo\app.py --global.developmentMode false --server.address localhost --browser.gatherUsageStats false
pause
