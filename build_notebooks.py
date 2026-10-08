"""
Generate the self-contained cloud notebooks from the code in gapfill/:

    ..\\.venv\\Scripts\\python build_notebooks.py
        -> notebooks/NOAA_gapfill_colab.ipynb
        -> notebooks/NOAA_gapfill_kaggle.ipynb

Each notebook carries its own copy of gapfill/common.py, prepare_data.py,
train.py and sector_grids.npz, so on Colab/Kaggle you only need the six
n18_*/n19_* zip files. Re-run this script after changing anything in gapfill/.
"""
import os
import json
import base64

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "gapfill")
OUT = os.path.join(HERE, "notebooks")
CODE_FILES = ["common.py", "prepare_data.py", "train.py"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(True)}


def embedded_code_cells():
    cells = [md("## 3. Write the pipeline code\n"
                "Exact copies of `gapfill/*.py` from the project; nothing to upload. "
                "The sector grids + land masks of all 14 sectors are embedded as base64 (31 KB).")]
    for f in CODE_FILES:
        src = open(os.path.join(SRC, f), encoding="utf-8").read()
        cells.append(code(f"%%writefile gapfill/{f}\n{src}"))
    b64 = base64.b64encode(open(os.path.join(SRC, "sector_grids.npz"), "rb").read()).decode()
    cells.append(code(
        "import base64, numpy as np\n"
        f"GRIDS_B64 = '{b64}'\n"
        "open('gapfill/sector_grids.npz', 'wb').write(base64.b64decode(GRIDS_B64))\n"
        "g = np.load('gapfill/sector_grids.npz')\n"
        "print('sector grids:', sorted({k.split('_')[0] for k in g.files}, key=int))"))
    return cells


INTRO = """
# NOAA-18 / NOAA-19 SST gap filling: train the remaining sectors ({platform})

This is the same pipeline that trained **Goa, Tamil Nadu, South AP, North AP and West Bengal**
on the local GTX 1050 Ti, now for the other sectors:

| id | sector | id | sector | id | sector |
|---|---|---|---|---|---|
| 1 | Gujarat | 5 | Kerala | 12 | Lakshadweep |
| 2 | Maharashtra | 6 | South Tamil Nadu | 13 | North Andaman |
| 4 | Karnataka | 10 | Odisha | 14 | South Andaman |

**What it does:** one model per satellite and sector.
* **Train:** 2014-01-01 → 2016-06-30, with every 10th day held out for validation.
* **Test:** 2016-07-01 → 2016-12-31 (6 months, never trained on).
* **Fill:** any scene is filled from the **30 days of that satellite's passes before it**.

**You only need** the six zip files: `n18_2014-*.zip n18_2015-*.zip n18_2016-*.zip
n19_2014-*.zip n19_2015-*.zip n19_2016-*.zip` (~7.7 GB total).

**Runs twice?** Every step skips work that is already done (archives, finished models).
If the session dies, run all cells again and it continues.

**Splitting across accounts:** set `SATS = ["n18"]` in one account and `SATS = ["n19"]` in the
other, or split `SECTORS`. Then copy both `models/` folders together at the end.
"""

HOME = """
## 8. Use the new models on your own machine
Copy the downloaded `models/` and `archives/` folders into the project folder `nova_data/`,
next to the ones already there (`models/n18/…`, `models/n19/…`, `archives/n18/…`).
Then start `run_noaa_app.bat`.

The new sectors turn **green** on the map. Select one, choose
*Scene from the archive* → Date → Time, and press *Run Reconstruction*.

To take the screenshots, run this with the app running:
`..\\.venv\\Scripts\\python take_screenshots_noaa.py`
"""


def summary_cell():
    return code(r'''
import glob, json, pandas as pd
rows = []
for f in sorted(glob.glob(os.path.join(WORK_DIR, "models", "*", "*_metrics.json"))):
    m = json.load(open(f))
    rows.append({"satellite": m["satellite"], "sector": m["sector"], "best_epoch": m["best_epoch"],
                 "val RMSE °C": round(m["validation"]["rmse"], 3),
                 "test RMSE °C": round(m["test"]["rmse"], 3),
                 "30-day composite °C": round(m["test"]["rmse_30day_composite"], 3),
                 "test bias °C": round(m["test"]["bias"], 3), "test passes": m["test"]["n_passes"],
                 "minutes": m["minutes"]})
df = pd.DataFrame(rows)
df.to_csv(os.path.join(WORK_DIR, "metrics_summary.csv"), index=False)
df
''')


TRAIN_CELL = r'''
# One process per satellite. With 2 GPUs (Kaggle T4 x2) they run in parallel, one per GPU;
# with 1 GPU they run one after the other. --skip-done makes a re-run continue where it stopped.
import subprocess, sys, time, torch
env = dict(os.environ, GAPFILL_DATA_DIR=DATA_DIR, GAPFILL_WORK_DIR=WORK_DIR,
           GAPFILL_CACHE_DIR=CACHE_DIR, PYTHONUNBUFFERED="1")
n_gpu = torch.cuda.device_count()
print("GPUs:", n_gpu, [torch.cuda.get_device_name(i) for i in range(n_gpu)])
args = lambda sat: [sys.executable, "gapfill/train.py", "--sat", sat, "--sector", ",".join(SECTORS),
                    "--epochs", str(EPOCHS), "--max-minutes", str(MAX_MINUTES),
                    "--batch-size", str(BATCH_SIZE), "--skip-done"]
os.makedirs(os.path.join(WORK_DIR, "logs"), exist_ok=True)

if n_gpu >= 2 and len(SATS) > 1:
    procs, logs = [], []
    for i, sat in enumerate(SATS):
        log = os.path.join(WORK_DIR, "logs", f"train_{sat}.log")
        procs.append(subprocess.Popen(args(sat), env=dict(env, CUDA_VISIBLE_DEVICES=str(i % n_gpu)),
                                      stdout=open(log, "w"), stderr=subprocess.STDOUT))
        logs.append(log)
    shown = [0] * len(logs)
    while any(p.poll() is None for p in procs):
        time.sleep(30)
        for k, log in enumerate(logs):
            lines = open(log, encoding="utf-8", errors="replace").read().splitlines()
            for line in lines[shown[k]:]:
                if "===" in line or "TEST" in line or "VAL" in line or "Error" in line:
                    print(f"[{SATS[k]}] {line}", flush=True)
            shown[k] = len(lines)
    for k, log in enumerate(logs):
        print(f"\n----- {SATS[k]} (exit {procs[k].returncode}) -----")
        print("\n".join(open(log, encoding="utf-8", errors="replace").read().splitlines()[-14:]))
else:
    for sat in SATS:
        log = os.path.join(WORK_DIR, "logs", f"train_{sat}.log")
        with open(log, "a") as fh:
            p = subprocess.Popen(args(sat), env=env, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            for line in p.stdout:
                fh.write(line)
                if "features" not in line:
                    print(line, end="", flush=True)
            p.wait()
        print(f"{sat}: exit code {p.returncode}")
'''

PREPARE_CELL = r'''
# Builds archives/<sat>/<sector>.nc (the sector time series) from the zips.
# Only missing archives are built, so a re-run after a disconnect is quick.
import subprocess
env = dict(os.environ, GAPFILL_DATA_DIR=DATA_DIR, GAPFILL_WORK_DIR=WORK_DIR,
           GAPFILL_CACHE_DIR=CACHE_DIR, PYTHONUNBUFFERED="1")
sys.path.insert(0, "gapfill")
import common as C
for sat in SATS:
    todo = [s for s in SECTORS if not os.path.exists(
        os.path.join(WORK_DIR, "archives", sat, C.SECTORS[s]["name"] + ".nc"))]
    if not todo:
        print(f"{sat}: all archives already built"); continue
    print(f"{sat}: building {todo}")
    r = subprocess.run([sys.executable, "gapfill/prepare_data.py", "--sats", sat,
                        "--sectors", ",".join(todo)], env=env)
    assert r.returncode == 0, "prepare_data failed - see the output above"
'''


def colab():
    cells = [md(INTRO.format(platform="Google Colab")),
             md("""
## 0. Before you start
1. **Runtime → Change runtime type → T4 GPU** (any GPU works).
2. Put the six zips in one Google Drive folder, e.g. `MyDrive/noaa_sst/` (subfolders are fine).
3. Set `DRIVE_FOLDER` below and run all cells (**Runtime → Run all**).

Results (models, archives, metrics) are written to `DRIVE_FOLDER/output`, so they survive a
disconnect. Rough time on a T4 for all 9 sectors × 2 satellites: about 30 min to build the
archives, then about 1–2 h of training."""),
             md("## 1. Settings"),
             code('''
DRIVE_FOLDER = "/content/drive/MyDrive/noaa_sst"     # folder holding the six n18_/n19_ zips

SATS    = ["n18", "n19"]                              # e.g. ["n18"] on one account, ["n19"] on another
SECTORS = ["1", "2", "4", "5", "6", "10", "12", "13", "14"]   # the sectors not trained yet
EPOCHS      = 25     # same as the first 5 sectors (keep equal so the sectors are comparable)
MAX_MINUTES = 30     # per model safety cap
BATCH_SIZE  = 16
'''),
             md("## 2. Mount Drive, check the GPU, install netCDF4"),
             code('''
import os, sys, glob, shutil
from google.colab import drive
drive.mount("/content/drive")
!nvidia-smi --query-gpu=name,memory.total --format=csv
!pip -q install netCDF4
os.makedirs("/content/gapfill_run/gapfill", exist_ok=True)
%cd /content/gapfill_run
WORK_DIR  = os.path.join(DRIVE_FOLDER, "output")       # persistent (Drive)
CACHE_DIR = "/content/feature_cache"                   # large, rebuilt quickly -> local disk
DATA_DIR  = "/content/data"                            # local copy of the zips (fast random access)
os.makedirs(WORK_DIR, exist_ok=True)
zips = sorted(glob.glob(os.path.join(DRIVE_FOLDER, "**", "n1[89]_*.zip"), recursive=True))
print(len(zips), "zip files found:"); print("\\n".join(zips))
assert zips, f"no n18_/n19_ zips under {DRIVE_FOLDER}"
'''),
             *embedded_code_cells(),
             md("## 4. Copy the zips of the selected satellites to local disk\n"
                "Reading thousands of files out of a zip on Google Drive is slow; a local copy takes a few minutes."),
             code('''
os.makedirs(DATA_DIR, exist_ok=True)
for z in zips:
    if os.path.basename(z)[:3] in SATS:
        dst = os.path.join(DATA_DIR, os.path.basename(z))
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(z):
            print("copying", os.path.basename(z)); shutil.copy(z, dst)
!ls -la {DATA_DIR}
'''),
             md("## 5. Build the sector archives"), code(PREPARE_CELL),
             md("## 6. Train (GPU)"), code(TRAIN_CELL),
             md("## 7. Results\nError on pixels hidden with real cloud masks, test period Jul–Dec 2016. "
                "The *30-day composite* column is the simple baseline the model should beat."),
             summary_cell(),
             code('''
# one zip with everything you need at home (models + archives + metrics)
out_zip = shutil.make_archive("/content/noaa_gapfill_outputs", "zip", WORK_DIR,
                              logger=None)
print(out_zip, round(os.path.getsize(out_zip) / 1e6), "MB  (also in", WORK_DIR, "on Drive)")
from google.colab import files
files.download(out_zip)
'''),
             md(HOME)]
    return cells


def kaggle():
    cells = [md(INTRO.format(platform="Kaggle")),
             md("""
## 0. Before you start
1. **Add Input → Upload → New Dataset**: upload the six zips; name it e.g. `noaa-sst`.
   Kaggle may unpack the zips into folders; either form works.
2. **Settings → Accelerator → GPU T4 x2** (NOAA-18 and NOAA-19 then train in parallel,
   one per GPU). **Settings → Internet → On** (for `pip install netCDF4`).
3. For an unattended run: **Save Version → Save & Run All (Commit)**. Everything in
   `/kaggle/working` becomes the notebook's output and can be downloaded.

Rough time for all 9 sectors × 2 satellites on T4 x2: about 20 min to build the archives,
then about 1 h of training (limit 12 h)."""),
             md("## 1. Settings"),
             code('''
SATS    = ["n18", "n19"]
SECTORS = ["1", "2", "4", "5", "6", "10", "12", "13", "14"]   # the sectors not trained yet
EPOCHS      = 25     # same as the first 5 sectors (keep equal so the sectors are comparable)
MAX_MINUTES = 30     # per model safety cap
BATCH_SIZE  = 16
'''),
             md("## 2. Paths, GPU check, netCDF4"),
             code('''
import os, sys, glob, shutil
DATA_DIR  = "/kaggle/input"                     # searched recursively for the zips / .nc files
WORK_DIR  = "/kaggle/working/output"            # saved as the notebook output
CACHE_DIR = "/tmp/feature_cache"                # large scratch, not saved
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs("/kaggle/working/gapfill_run/gapfill", exist_ok=True)
%cd /kaggle/working/gapfill_run
!nvidia-smi --query-gpu=name,memory.total --format=csv
try:
    import netCDF4
except ImportError:
    !pip -q install netCDF4
n_zip = len(glob.glob(os.path.join(DATA_DIR, "**", "n1[89]_*.zip"), recursive=True))
n_nc  = len(glob.glob(os.path.join(DATA_DIR, "**", "*-noaa-1[89]-sst.nc"), recursive=True))
print(f"found {n_zip} zips and {n_nc} extracted .nc files under {DATA_DIR}")
assert n_zip or n_nc, "attach the dataset with the n18_/n19_ files (Add Input)"
'''),
             *embedded_code_cells(),
             md("## 4. (nothing to copy: Kaggle inputs are already on local disk)"),
             md("## 5. Build the sector archives"), code(PREPARE_CELL),
             md("## 6. Train (GPU)"), code(TRAIN_CELL),
             md("## 7. Results\nError on pixels hidden with real cloud masks, test period Jul–Dec 2016. "
                "The *30-day composite* column is the simple baseline the model should beat."),
             summary_cell(),
             code('''
# one zip with everything you need at home (models + archives + metrics) -> Output tab
out_zip = shutil.make_archive("/kaggle/working/noaa_gapfill_outputs", "zip", WORK_DIR)
print(out_zip, round(os.path.getsize(out_zip) / 1e6), "MB")
'''),
             md(HOME)]
    return cells


def write(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"},
                       "accelerator": "GPU"}}
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", path)


if __name__ == "__main__":
    write("NOAA_gapfill_colab.ipynb", colab())
    write("NOAA_gapfill_kaggle.ipynb", kaggle())
