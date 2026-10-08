"""
Build per-satellite, per-sector SST archives from the raw NOAA data.

    python gapfill/prepare_data.py                          # all 14 sectors, both satellites
    python gapfill/prepare_data.py --sectors 1,2,4 --sats n18

Finds the data anywhere under $GAPFILL_DATA_DIR (default: this folder), either
as the original zips (n18_2014-*.zip ...) or as extracted files
(.../20140124-232440Z-noaa-18-sst.nc) -- Kaggle unpacks uploaded zips. Each
swath is cropped to the sectors, box-averaged onto the sector grid and written
to $GAPFILL_WORK_DIR/archives/<sat>/<sector_name>.nc with dims (time, lat, lon).
Passes that see none of a sector's sea are dropped for that sector.
"""
import os
import re
import sys
import glob
import time
import zipfile
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

FILE_RE = re.compile(r"(\d{8}-\d{6}Z)-noaa-(18|19)-sst\.nc$")


def find_sources(sat):
    """[(zip_path or None, [member names or file paths])], de-duplicated by pass."""
    num = sat[1:]
    seen, tasks = set(), []
    zips = [z for z in glob.glob(os.path.join(C.DATA_DIR, "**", "*.zip"), recursive=True)
            if os.path.basename(z).lower().startswith(sat)]
    for zp in sorted(zips):
        with zipfile.ZipFile(zp) as z:
            names = sorted(n for n in z.namelist()
                           if (m := FILE_RE.search(n)) and m.group(2) == num)
        names = [n for n in names if FILE_RE.search(n).group(1) not in seen]
        seen.update(FILE_RE.search(n).group(1) for n in names)
        for i in range(0, len(names), 200):
            tasks.append((zp, names[i:i + 200]))
    loose = sorted(f for f in glob.glob(os.path.join(C.DATA_DIR, "**", f"*-noaa-{num}-sst.nc"),
                                        recursive=True)
                   if FILE_RE.search(f).group(1) not in seen)
    for i in range(0, len(loose), 200):
        tasks.append((None, loose[i:i + 200]))
    n = sum(len(t[1]) for t in tasks)
    print(f"{sat}: {n} passes in {len(zips)} zip(s) + {len(loose)} loose file(s)", flush=True)
    return tasks


def work(args):
    (zpath, names), sectors = args
    import netCDF4
    grids = {sid: C.load_sector_grid(sid) for sid in sectors}
    binners, out = {}, {sid: ([], []) for sid in sectors}
    z = zipfile.ZipFile(zpath) if zpath else None
    try:
        for name in names:
            try:
                t = C.parse_time(name)
                raw = z.read(name) if z else open(name, "rb").read()
                with netCDF4.Dataset("mem", memory=raw) as d:
                    if not binners:
                        flat, flon = d["lat"][:].data, d["lon"][:].data
                        binners = {sid: C.Binner(flat, flon, g[0], g[1])
                                   for sid, g in grids.items()}
                    var = d["SST"]
                    var.set_auto_mask(False)
                    for sid, b in binners.items():
                        win = var[0, b.r0:b.r1, b.c0:b.c1].astype("f4")
                        win[(win < -50) | (win > 50)] = np.nan
                        g = b(win)
                        sea = grids[sid][2]
                        if np.isfinite(g[sea]).any():
                            g[~sea] = np.nan
                            out[sid][0].append(t)
                            out[sid][1].append(g)
            except Exception as e:
                print(f"  skip {name}: {e}", flush=True)
    finally:
        if z:
            z.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sats", default=",".join(C.SATELLITES))
    ap.add_argument("--sectors", default=",".join(C.SECTORS))
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 2))
    a = ap.parse_args()
    sectors = a.sectors.split(",")
    t0 = time.time()
    for sat in a.sats.split(","):
        tasks = find_sources(sat)
        if not tasks:
            print(f"  no {sat} data found under {C.DATA_DIR}")
            continue
        acc = {sid: ([], []) for sid in sectors}
        with Pool(a.workers) as pool:
            for k, out in enumerate(pool.imap_unordered(work, [(t, sectors) for t in tasks]), 1):
                for sid, (ts, gs) in out.items():
                    acc[sid][0].extend(ts)
                    acc[sid][1].extend(gs)
                if k % 5 == 0 or k == len(tasks):
                    print(f"  [{k}/{len(tasks)}] {time.time()-t0:.0f}s", flush=True)

        for sid in sectors:
            ts, gs = acc.pop(sid)
            if not ts:
                print(f"{sat} {C.SECTORS[sid]['name']}: no passes")
                continue
            lat, lon, sea = C.load_sector_grid(sid)
            order = np.argsort(np.array(ts, dtype="datetime64[ns]"))
            times = pd.DatetimeIndex(np.array(ts, dtype="datetime64[ns]")[order])
            sst = np.stack([gs[i] for i in order])
            del gs
            # a swath split across two files can give duplicate timestamps
            _, first = np.unique(times.values, return_index=True)
            times, sst = times[first], sst[first]
            ds = xr.Dataset(
                {"SST": (("time", "lat", "lon"), sst.astype("f4")),
                 "mask": (("lat", "lon"), sea.astype("i1"))},
                coords={"time": times, "lat": lat, "lon": lon},
                attrs={"satellite": C.SATELLITES[sat], "sector_id": sid,
                       "sector_name": C.SECTORS[sid]["name"],
                       "source": "NOAA AVHRR L2 SST swaths, box-averaged to the sector grid"})
            os.makedirs(os.path.join(C.ARCHIVE_DIR, sat), exist_ok=True)
            path = os.path.join(C.ARCHIVE_DIR, sat, f"{C.SECTORS[sid]['name']}.nc")
            ds.to_netcdf(path, encoding={"SST": {"zlib": True, "complevel": 4}})
            cov = np.isfinite(sst[:, sea]).mean(1)
            print(f"{sat} {C.SECTORS[sid]['name']:22s} passes={len(times):5d} "
                  f"{times[0]:%Y-%m-%d}..{times[-1]:%Y-%m-%d}  mean coverage={cov.mean():.2f}",
                  flush=True)
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
