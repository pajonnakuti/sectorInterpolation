#!/usr/bin/env bash
# One-time setup on macOS / Linux: creates .venv in this folder and installs what the app needs.
#   FORCE_CPU=1 bash setup_mac_linux.sh   skips the CUDA build even with an NVIDIA GPU.
set -e
cd "$(dirname "$0")"
PYTHON=${PYTHON:-python3}
"$PYTHON" -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null || {
    echo "Python 3.11 or newer is needed (python.org, or your package manager)."; exit 1; }
[ -x .venv/bin/python ] || "$PYTHON" -m venv .venv
PY=.venv/bin/python
"$PY" -m pip install --upgrade pip
if [ "$(uname)" = "Darwin" ]; then
    "$PY" -m pip install torch                                   # CPU / Apple silicon build
elif command -v nvidia-smi >/dev/null && [ -z "$FORCE_CPU" ]; then
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cu128
else
    "$PY" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
"$PY" -m pip install -r requirements_known_good.txt || {
    echo "Those exact versions are not available here; installing the latest ones."
    "$PY" -m pip install -r requirements_noaa.txt; }
"$PY" -c "import torch, numpy, pandas, xarray, netCDF4, scipy, rasterio, geopandas, dash, waitress; print('OK - PyTorch', torch.__version__, '- GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none, the app runs on the CPU')"
echo "Setup finished. Start the app with: bash run_noaa_app.sh"
