@echo off
rem Starts the NOAA-18 / NOAA-19 SST gap-filling web app at http://localhost:8052
rem and opens it in the browser once it is ready (the first start takes ~30 s).
rem Uses .venv in this folder (made by setup_windows.bat), else ..\.venv, else the python on PATH.
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set NOAA_APP_OPEN_BROWSER=1
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=%~dp0..\.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
echo Loading the models and archives; the browser opens when the app is ready ...
"%PY%" noaa_app.py
pause
