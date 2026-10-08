# NOAA-18 / NOAA-19 SST gap filling — quick start

A local web app that fills cloud gaps in NOAA-18 / NOAA-19 AVHRR sea-surface
temperature (SST) images for 14 sectors around India, with one trained model per
satellite and sector (28 models, all included). You pick a sector on a map, pick a
satellite pass from 2014–2016 (or upload your own GeoTIFF), and the model fills the
clouded pixels from the 30 days of passes before it.

## 1. Install Python (once)

Python **3.11 or newer** (tested with 3.14; 3.12 and 3.13 work too):
<https://www.python.org/downloads/>. On Windows, tick **"Add python.exe to PATH"**
in the installer.

## 2. Set up (once, 5–15 minutes)

Unzip this folder anywhere (a path without special characters is safest), then:

* **Windows:** double-click **`setup_windows.bat`**
* **macOS / Linux:** `bash setup_mac_linux.sh`

It creates a private Python environment in `.venv\` inside this folder, so nothing
else on the computer is changed. With an NVIDIA GPU it installs the CUDA build of
PyTorch (~3 GB download); without one, the CPU build (~250 MB). The app works on
either; a reconstruction takes ~2 s on a GPU and a few seconds more on a CPU. To get
the small CPU build even with an NVIDIA GPU, run `set FORCE_CPU=1` in a command
prompt first and start `setup_windows.bat` from that same prompt.

## 3. Run

* **Windows:** double-click **`run_noaa_app.bat`**
* **macOS / Linux:** `bash run_noaa_app.sh`

The first start takes ~30 s (it loads all 28 models and the archives); then the
browser opens **http://localhost:8052** by itself. Keep the black window open while
using the app; close it to stop the app.

## 4. Use it

1. **Satellite & sector** — pick NOAA-18 or NOAA-19, then click a sector on the map
   or type its name in the search box. Hovering a sector shows how good its model
   is. Scroll to zoom the map; *Skill* colours the sectors by model error.
2. **Scene** — *Archive*: choose a date and a time (each time shows how much of the
   sea that pass saw). *Upload*: a GeoTIFF (EPSG:4326, °C or K) or a raw NOAA `.nc`
   whose file name holds the time, e.g. `20161015-120450Z-noaa-18-sst.tif`; ready-made
   examples for every sector are in `demo_inputs\`. Press **Run reconstruction**.
3. **Result** — observed pixels, the gap-filled field and the model's uncertainty (σ)
   side by side. Scroll to zoom (the panels move together), hover any cell for its
   value, switch the view / colour map / SST range, and **Download NetCDF** saves the
   filled field with its per-pixel error estimate. The camera icon saves a PNG.

The map background and the page styling come from the internet; without internet
the page looks plainer, and the **Offline** basemap draws only the coastline, but
everything else works.

## How good are the models

Error of the filled pixels on Jul–Dec 2016, a period never used in training
(average over the 28 models): **0.67 °C**, against 1.32 °C for simply averaging the
previous 30 days. Per model: `epochs200_report.md`, or hover a sector in the app.

## What is in this folder

| | |
|---|---|
| `noaa_app.py`, `run_noaa_app.*`, `setup_*` | the web app and its launchers |
| `models\` | the 28 trained models (PyTorch, 200 epochs) with their metrics |
| `archives\` | 2014–2016 passes per satellite and sector (the models' 30-day context) |
| `assets\` | coastline used to draw the maps |
| `demo_inputs\` | one Oct-2016 test GeoTIFF per satellite and sector, to try **Upload** |
| `gapfill\`, `notebooks\` | training code (retraining also needs the raw NOAA zips, not included) |
| `screenshots\` | what the app looks like for every sector |
| `README_NOAA.md` | full documentation: data, method, training |

## Troubleshooting

* **"An Application Control policy has blocked this file"** (Windows 11 Smart App
  Control) — Windows is blocking a newly released package. Install the previous
  release of the package named in the error, e.g.
  `.venv\Scripts\python -m pip install "rasterio<1.5.1"`, and start again.
* **Port 8052 already in use** — `set NOAA_APP_PORT=8060` in a command prompt, then
  start `run_noaa_app.bat` from it and open http://localhost:8060.
* **`python` not found** — reinstall Python with "Add python.exe to PATH" ticked, or
  on macOS / Linux use `PYTHON=python3.12 bash setup_mac_linux.sh`.
