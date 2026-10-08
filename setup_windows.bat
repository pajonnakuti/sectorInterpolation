@echo off
rem One-time setup on Windows: creates .venv in this folder and installs what the app needs.
rem   set FORCE_CPU=1 before running to skip the ~3 GB CUDA download even with an NVIDIA GPU.
setlocal
cd /d "%~dp0"
python -c "import sys; assert sys.version_info >= (3, 11)" >nul 2>nul
if errorlevel 1 (
    echo Python 3.11 or newer is needed. Install it from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH" in the installer, then run this again.
    pause
    exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment in .venv ...
    python -m venv .venv || goto :fail
)
set "PY=%~dp0.venv\Scripts\python.exe"
"%PY%" -m pip install --upgrade pip || goto :fail

set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
if not defined FORCE_CPU (
    where nvidia-smi >nul 2>nul && set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
)
echo.
echo Installing PyTorch from %TORCH_INDEX% ...
"%PY%" -m pip install torch --index-url %TORCH_INDEX% || goto :fail

echo.
echo Installing the other packages (the versions tested with this app first) ...
"%PY%" -m pip install -r requirements_known_good.txt
if errorlevel 1 (
    echo Those exact versions are not available for this Python; installing the latest ones.
    "%PY%" -m pip install -r requirements_noaa.txt || goto :fail
)

echo.
echo Checking that everything loads ...
"%PY%" -c "import torch, numpy, pandas, xarray, netCDF4, scipy, rasterio, geopandas, dash, dash_bootstrap_components, waitress; print('OK - PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, the app runs on the CPU')" || goto :importfail
echo.
echo Setup finished. Start the app with run_noaa_app.bat
pause
exit /b 0

:importfail
echo.
echo A package failed to load. If the message says "An Application Control policy has
echo blocked this file", Windows Smart App Control is blocking a newly released package.
echo Install the previous release of that package, e.g.:
echo     .venv\Scripts\python -m pip install "rasterio<1.5.1"
echo and run this check again. See START_HERE.md, "Troubleshooting".
pause
exit /b 1

:fail
echo.
echo Setup failed; see the messages above.
pause
exit /b 1
