# NOAA-18 / NOAA-19 SST gap filling

One model per satellite and sector, in PyTorch/CUDA. TensorFlow cannot use a GPU
on native Windows, so this does not use the TF DINCAE code in `app/gapgan`.
It uses the same 14 sector boxes, and the same sector grids (~0.036°) and land
masks, now stored in `gapfill/sector_grids.npz`. With the coastline in `assets/`,
this folder is self-contained.

| Status | Sectors |
|---|---|
| **Trained** on the local GTX 1050 Ti (4 GB): 10 models, ~70 min | 3 Goa, 7 Tamil Nadu (North TN), 8 South AP, 9 North AP, 11 West Bengal |
| **Trained** on a local RTX 4060 Laptop (8 GB, mixed precision): 18 models, 25 epochs, 1–4 min each | 1 Gujarat, 2 Maharashtra, 4 Karnataka, 5 Kerala, 6 South TN, 10 Odisha, 12 Lakshadweep, 13 North Andaman, 14 South Andaman |

All 14 sectors now have a NOAA-18 and a NOAA-19 model.

On Windows with Smart App Control on, the newest wheels of a few packages can be
blocked ("An Application Control policy has blocked this file"). Installing the
previous release avoids it; this machine uses `pandas==3.0.5`, `pyproj==3.7.2`,
`pyogrio==0.12.1` and `rasterio==1.5.0`. The verdict can change later for a
package that loaded before (rasterio 1.5.1 was blocked a day after it worked), so
if the app stops with this error, test each import and step that package back.

## Training on Colab or Kaggle instead

`notebooks/` has two self-contained notebooks. They carry all the code and the
sector grids, so all you upload is the six zips (`n18/*.zip`, `n19/*.zip`):

* **`NOAA_gapfill_kaggle.ipynb`** (recommended). Upload the zips as a Kaggle dataset,
  choose *GPU T4 x2* and turn Internet on. NOAA-18 and NOAA-19 then train in
  parallel, one per GPU. Use *Save & Run All (Commit)* for an unattended run.
  Output: `noaa_gapfill_outputs.zip`.
* **`NOAA_gapfill_colab.ipynb`**. Put the zips in a Drive folder and set
  `DRIVE_FOLDER`. Results go to that folder's `output/`, so they survive a
  disconnect. Run all cells again to continue: finished archives and models are skipped.

To split the work between accounts, set `SATS = ["n18"]` in one and
`SATS = ["n19"]` in the other, or split `SECTORS`. Keep `EPOCHS = 25` so every
sector is trained the same way. On a T4 the models use mixed precision
automatically. If you change anything in `gapfill/`, rebuild the notebooks with
`python build_notebooks.py`.

**Afterwards:** unzip the output and copy its `models/` and `archives/` into this
folder, next to the existing ones. The new sectors turn green in the app.

## Moving to another machine

To only *run* the app, copy `noaa_app.py`, the `run_noaa_app.*` / `setup_*` scripts,
both requirements files and the `gapfill/`, `assets/`, `models/`, `archives/` and
`demo_inputs/` folders (~0.9 GB); `START_HERE.md` walks through it. On the new
machine, `setup_windows.bat` (or `bash setup_mac_linux.sh`) makes a `.venv` in the
folder with PyTorch (CUDA with an NVIDIA GPU, else CPU) and the tested package
versions from `requirements_known_good.txt`; then `run_noaa_app.bat` starts the app
and opens the browser. To also *train*, add the raw zips (`n18_*.zip`, `n19_*.zip`).
`feature_cache/` is a rebuildable training cache; leave it out.

`run_noaa_app.bat` uses `.venv` in this folder, else `..\.venv`, else the `python` on PATH.

## Data split

| | Period | Use |
|---|---|---|
| train | 2014-01-01 → 2016-06-30, except every 10th day | fitting |
| val   | every 10th day of the same span | picking the best epoch |
| test  | 2016-07-01 → 2016-12-31 (6 months) | reported only, never trained on |

## How a file is filled (training and the app do the same thing)

For an image at time *t* the model sees:

* the image itself (the observed pixels),
* the 4 most recent passes of the same satellite before *t*, with their age,
* composites of **the 30 days before *t***: all passes, only passes of the same
  day/night phase, and the last 7 days,
* latitude, longitude, day of year, time of day.

It outputs SST and an error estimate for every sea cell. Observed pixels are
kept as they are and only the gaps take the model's value. During training,
gaps are cut into each image with the real cloud mask of another pass, and the
loss is scored on the pixels that were hidden.

The architecture is a DINCAE-style convolutional encoder/decoder with skip
connections and a Gaussian (mean + variance) output. It has no dense
bottleneck (DINCAE 2 dropped it too). This keeps it at ~4.9 M parameters, which
fits in 4 GB.

## Commands (from this folder)

```
python gapfill\prepare_data.py --sectors 3,7,8,9,11   # zips -> archives\<sat>\<sector>.nc  (~4 min)
python gapfill\train.py --sector 3,7,8,9,11 --skip-done   # models -> models\<sat>\
python gapfill\make_demo_inputs.py                    # Oct-2016 test TIFs for trained sectors -> demo_inputs\
run_noaa_app.bat                                      # web app on http://localhost:8052
python take_screenshots_noaa.py                       # screenshots\<sat>_<sector>.png, every trained sector
```

With no `--sectors` / `--sector`, the scripts run all 14 sectors.
`train.py --sat n18 --sector 8 --epochs 40` retrains a single model.
Data and output folders can be moved with `GAPFILL_DATA_DIR`,
`GAPFILL_WORK_DIR`, `GAPFILL_CACHE_DIR` and `GAPFILL_MODEL_DIR` (the notebooks set these).

`overnight_200ep.bat` retrains all 28 models for 200 epochs instead of 25 (about
8 h on an RTX 4060). It trains into `models_200ep\`, and only when all 28 are
done moves `models\` to `models_25ep\` and installs the new ones;
`epochs200_report.md` compares the two. Re-running it resumes. On one sector,
200 epochs lowered the test RMSE by about 4% (0.442 → 0.425 °C).
Per-model metrics and the per-epoch history are in `models\<sat>\<sector>_metrics.json`.

## Using the app

Pick **NOAA-18 / NOAA-19**, then click a sector on the map or type its name in
the search box above it (the map flies to it). The map zooms with the mouse
wheel and pans by dragging; **⌖ Zoom to sector** and **⤢ All** reset the view.
Hovering a sector shows its model's test RMSE and how much data its archive
has. Below the map, colour the sectors by **Status** or by **Model skill**
(test RMSE), and switch the basemap (Ocean, Light, Satellite, or Offline, which
draws only the bundled coastline and needs no internet). Then either:

* **Scene from the archive**: choose a **Date** and a **Time (UTC)**. Each time
  shows how much of the sector's sea that pass observed. Or:
* **Upload a GeoTIFF** (see below).

Press **Run Reconstruction**. The model fills the gaps from the 30 days of
that satellite's passes before the scene. The result shows the observed scene,
the gap-filled field and the model's uncertainty (σ, only where it filled) side
by side:

* the three panels zoom and pan together (wheel, box-drag, double-click resets);
* hovering any cell shows its value, whether it was observed or filled, and σ;
* **View** switches between all three panels, observed | filled, or one panel;
  **Colour map** and the **SST range** slider restyle it without re-running, and
  **Gap outline** draws the edge of the filled area;
* the camera icon saves a PNG, and **Download NetCDF** saves the filled field
  together with the per-pixel error estimate.

## Uploading your own file

The app accepts a GeoTIFF (EPSG:4326, north-up, °C or K) or a raw NOAA `.nc`.
The observation time is read from the file name: `20161015-120450Z-noaa-18-sst.tif`
(the NOAA style) or `3RIMG_15OCT2016_1204_...` (the ISRO style). The file is filled
with the selected satellite's 30 days of passes before that time. The archive
covers 2014-01-01 → 2016-12-31. For a newer file, add that satellite's zips for
the preceding month and rerun `prepare_data.py`. The models do not need
retraining.
