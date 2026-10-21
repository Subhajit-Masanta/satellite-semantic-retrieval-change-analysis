@echo off
rem Runs any project script with the project's own libraries, for example:
rem     run.bat experiments\run_oscd.py
rem     run.bat demo\fetch_india.py
cd /d "%~dp0"
set PYTHONPATH=%~dp0env\libs_gpu;%~dp0env\libs
set HF_HUB_OFFLINE=1
python %*
