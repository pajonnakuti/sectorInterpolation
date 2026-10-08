"""
Cut one GeoTIFF per satellite and sector from a raw October-2016 swath (test
period, never seen in training) to upload in the app:

    demo_inputs/<sat>/<sector>__<YYYYMMDD-HHMMSS>Z-noaa-<18|19>-sst.tif

The pass is chosen to be partly cloudy (35-75% of the sector's sea observed)
so there are gaps to fill and enough data to see the result (closest to 55%).
"""
import os
import sys
import glob
import zipfile

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

OUT = os.path.join(C.NOVA_DIR, "demo_inputs")
MARGIN = 0.5


def pick_pass(sat, sid):
    with xr.open_dataset(os.path.join(C.ARCHIVE_DIR, sat, f"{C.SECTORS[sid]['name']}.nc")) as ds:
        t = pd.DatetimeIndex(ds.time.values)
        sea = ds.mask.values == 1
        cov = np.isfinite(ds.SST.values[:, sea]).mean(1)
    for lo, hi in [("2016-10-01", "2016-11-01"), (str(C.TEST_START.date()), "2017-01-01")]:
        sel = (t >= lo) & (t < hi) & (cov >= 0.35) & (cov <= 0.75)
        if sel.any():
            i = np.where(sel)[0][np.argmin(np.abs(cov[sel] - 0.55))]
            return t[i], cov[i]
    raise RuntimeError(f"no suitable test pass for {sat} {sid}")


def main():
    for sat in C.SATELLITES:
        os.makedirs(os.path.join(OUT, sat), exist_ok=True)
        zpaths = glob.glob(os.path.join(C.DATA_DIR, "**", f"{sat}_2016*.zip"), recursive=True)
        if not zpaths:
            print(f"no {sat}_2016 zip under {C.DATA_DIR}; skipping {sat}")
            continue
        with zipfile.ZipFile(zpaths[0]) as z:
            for sid, s in C.SECTORS.items():
                if not os.path.exists(os.path.join(C.MODEL_DIR, sat, f"{s['name']}.pt")):
                    continue
                t, cov = pick_pass(sat, sid)
                stem = f"{t:%Y%m%d-%H%M%S}Z-noaa-{sat[1:]}-sst"
                with netCDF4.Dataset("m", memory=z.read(f"{sat}_2016/{stem}.nc")) as d:
                    lat, lon = d["lat"][:].data, d["lon"][:].data
                    (y0, x0), (y1, x1) = s["bounds"]
                    r = np.where((lat >= y0 - MARGIN) & (lat <= y1 + MARGIN))[0]
                    c = np.where((lon >= x0 - MARGIN) & (lon <= x1 + MARGIN))[0]
                    v = d["SST"]; v.set_auto_mask(False)
                    arr = v[0, r[0]:r[-1] + 1, c[0]:c[-1] + 1].astype("f4")
                arr[(arr < -50) | (arr > 50)] = np.nan
                la, lo = lat[r[0]:r[-1] + 1], lon[c[0]:c[-1] + 1]
                arr = arr[::-1]                                   # north-up
                dx, dy = lo[1] - lo[0], la[1] - la[0]
                tf = from_origin(lo[0] - dx / 2, la[-1] + dy / 2, dx, dy)
                path = os.path.join(OUT, sat, f"{s['name']}__{stem}.tif")
                with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1],
                                   height=arr.shape[0], count=1, dtype="float32",
                                   crs="EPSG:4326", transform=tf, nodata=np.nan,
                                   compress="deflate") as dst:
                    dst.write(arr, 1)
                    dst.update_tags(units="degC", source=f"{C.SATELLITES[sat]} AVHRR {stem}.nc")
                print(f"{sat} {s['name']:22s} {t}  sea observed={cov:.0%}  -> {path}")


if __name__ == "__main__":
    main()
