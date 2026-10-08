"""
Shared pieces of the NOAA-18 / NOAA-19 AVHRR SST gap-filling pipeline.

Everything that must be identical at training time and at inference time lives
here: sector grids, the regridding of a raw swath onto a sector grid, the input
features built from the previous 30 days, and the network itself. The Dash app
imports this module, so the app and the trainer cannot drift apart.
"""
import os
import re
import numpy as np
import pandas as pd

HERE      = os.path.dirname(os.path.abspath(__file__))
NOVA_DIR  = os.path.dirname(HERE)
# Raw data: the n18_<year>-*.zip / n19_<year>-*.zip files (or their extracted
# .nc files), searched recursively. Outputs go under WORK_DIR.
# Both can be overridden, e.g. on Colab/Kaggle.
DATA_DIR    = os.environ.get("GAPFILL_DATA_DIR", NOVA_DIR)
WORK_DIR    = os.environ.get("GAPFILL_WORK_DIR", NOVA_DIR)
ARCHIVE_DIR = os.path.join(WORK_DIR, "archives")      # <sat>/<sector>.nc
MODEL_DIR   = os.environ.get("GAPFILL_MODEL_DIR", os.path.join(WORK_DIR, "models"))  # <sat>/<sector>.pt
CACHE_DIR   = os.environ.get("GAPFILL_CACHE_DIR", os.path.join(WORK_DIR, "feature_cache"))
GRID_FILE   = os.path.join(HERE, "sector_grids.npz")  # lat/lon/sea mask of all 14 sectors

SATELLITES = {"n18": "NOAA-18", "n19": "NOAA-19"}

# All 14 sectors. Ids/names/boxes are the ones in app/gapgan/dashapp.py; the
# grids (~0.036 deg) and land masks come from app/gapgan/nc_data/3R_<id>_SST_DATA.nc,
# copied into sector_grids.npz so this folder is self-contained.
SECTORS = {
    "1":  {"name": "1_Gujarat",            "label": "Gujarat",
           "bounds": [[20.04315567, 67.88010406], [23.75766563, 72.97928619]]},
    "2":  {"name": "2_Maharashtra",        "label": "Maharashtra",
           "bounds": [[15.70783329, 70.50718689], [20.04315567, 73.71035767]]},
    "3":  {"name": "3_Goa",                "label": "Goa",
           "bounds": [[14.85914707, 71.58411407], [15.70611858, 75.03230286]]},
    "4":  {"name": "4_Karnataka",          "label": "Karnataka",
           "bounds": [[12.80841446, 71.58411407], [14.86688137, 75.03580475]]},
    "5":  {"name": "5_Kerala",             "label": "Kerala",
           "bounds": [[7.38235903, 74.23501587], [12.79460812, 77.18965912]]},
    "6":  {"name": "6_SouthTamilNadu",     "label": "South Tamil Nadu",
           "bounds": [[7.01648331, 77.18965912], [10.06777668, 81.33168793]]},
    "7":  {"name": "7_NorthTamilNadu",     "label": "Tamil Nadu",
           "bounds": [[10.08158302, 79.02596283], [13.51026344, 82.38100433]]},
    "8":  {"name": "8_SouthAndhraPradesh", "label": "South Andhra Pradesh",
           "bounds": [[13.51026344, 79.87512970], [16.53046989, 83.54584503]]},
    "9":  {"name": "9_NorthAndhraPradesh", "label": "North Andhra Pradesh",
           "bounds": [[16.53046989, 82.18675232], [19.11697006, 85.88069916]]},
    "10": {"name": "10_Odisha",            "label": "Odisha",
           "bounds": [[19.11697006, 84.77572632], [22.21802711, 89.00102234]]},
    "11": {"name": "11_WestBengal",        "label": "West Bengal",
           "bounds": [[20.71798897, 87.48213196], [22.21802711, 89.00102234]]},
    "12": {"name": "12_lakshadweep",       "label": "Lakshadweep",
           "bounds": [[8.06546974, 71.29454803], [12.32047653, 74.05455780]]},
    "13": {"name": "13_NorthAndaman",      "label": "North Andaman",
           "bounds": [[10.07467461, 91.52107239], [15.23840141, 94.50333405]]},
    "14": {"name": "14_SouthAndaman",      "label": "South Andaman",
           "bounds": [[6.34685087, 92.12857056], [10.07467461, 94.77947235]]},
}
FIRST_BATCH = ["3", "7", "8", "9", "11"]                  # trained on the local GPU
REMAINING   = [s for s in SECTORS if s not in FIRST_BATCH]  # 1 2 4 5 6 10 12 13 14

TRAIN_END   = pd.Timestamp(os.environ.get("GAPFILL_TRAIN_END", "2016-07-01"))
TEST_START  = TRAIN_END                    # train: before TRAIN_END, test: from it on
CONTEXT_DAYS = 30                          # history the model may look at
N_PREV       = 4                           # most recent passes fed individually
MIN_CELL_FRAC = 0.3                        # fine pixels needed for a valid coarse cell


def load_sector_grid(sector_id):
    """lat (descending), lon (ascending), sea mask (bool) of a sector."""
    with np.load(GRID_FILE) as g:
        return (g[f"{sector_id}_lat"].copy(), g[f"{sector_id}_lon"].copy(),
                g[f"{sector_id}_mask"].astype(bool))


def parse_time(filename):
    """
    Observation time from a file name. Accepts the NOAA naming used in the
    archives (20161015-120450Z-noaa-18-sst.nc) and the ISRO naming used by the
    existing app (3RIMG_24NOV2025_2045_...).
    """
    base = os.path.basename(filename)
    m = re.search(r"(\d{8})-(\d{6})Z", base)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M%S")
    m = re.search(r"(\d{2}[A-Za-z]{3}\d{4})_(\d{4})", base)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%d%b%Y%H%M")
    m = re.search(r"(\d{8})[T_-]?(\d{4})", base)
    if m:
        return pd.to_datetime(m.group(1) + m.group(2), format="%Y%m%d%H%M")
    raise ValueError(f"Cannot read the date/time from '{base}'. Expected a name like "
                     f"20161015-120450Z-noaa-18-sst.tif")


def parse_satellite(filename):
    m = re.search(r"noaa-?(18|19)", os.path.basename(filename), re.I)
    return f"n{m.group(1)}" if m else None


# ---------------------------------------------------------------------------
# Regridding: fine swath (~0.01 deg) -> sector grid (~0.036 deg), box average
# ---------------------------------------------------------------------------
class Binner:
    """Averages every fine pixel whose centre falls in a coarse cell."""

    def __init__(self, fine_lat, fine_lon, lat, lon):
        dlat, dlon = abs(lat[1] - lat[0]), abs(lon[1] - lon[0])
        # lat is descending on the sector grid
        ri = np.rint((lat[0] - fine_lat) / dlat).astype(int)
        ci = np.rint((fine_lon - lon[0]) / dlon).astype(int)
        rows = np.where((ri >= 0) & (ri < len(lat)))[0]
        cols = np.where((ci >= 0) & (ci < len(lon)))[0]
        if len(rows) == 0 or len(cols) == 0:
            raise ValueError("The file does not cover this sector.")
        self.r0, self.r1 = rows.min(), rows.max() + 1
        self.c0, self.c1 = cols.min(), cols.max() + 1
        R, C = np.meshgrid(ri[self.r0:self.r1], ci[self.c0:self.c1], indexing="ij")
        self.shape = (len(lat), len(lon))
        self.flat = (R * len(lon) + C).ravel()
        self.total = np.bincount(self.flat, minlength=lat.size * lon.size)

    def __call__(self, fine):
        """fine: 2-D window [r0:r1, c0:c1], NaN = missing. Returns coarse field."""
        v = fine.ravel()
        ok = np.isfinite(v)
        n = self.total.size
        s = np.bincount(self.flat[ok], weights=v[ok], minlength=n)
        c = np.bincount(self.flat[ok], minlength=n)
        out = np.full(n, np.nan, dtype="f4")
        good = (c > 0) & (c >= MIN_CELL_FRAC * np.maximum(self.total, 1))
        out[good] = s[good] / c[good]
        return out.reshape(self.shape)


# ---------------------------------------------------------------------------
# Input features for one target pass (training and inference share this)
# ---------------------------------------------------------------------------
def is_day(times, lon_center):
    """Local solar time between 06 and 18 h -> daytime pass."""
    t = pd.DatetimeIndex(times)
    hour = (t.hour + t.minute / 60.0 + lon_center / 15.0) % 24
    return np.asarray((hour >= 6) & (hour < 18))


def n_channels():
    return 2 + 3 * N_PREV + 6 + 6


def build_features(target_obs, target_time, hist_obs, hist_times, stats, lat, lon):
    """
    target_obs : (H, W) float, NaN = gap (the image to fill)
    target_time: Timestamp
    hist_obs   : (T, H, W) float, NaN = gap, passes strictly BEFORE target_time
                 within CONTEXT_DAYS (older ones are ignored)
    hist_times : (T,) datetime64
    stats      : dict(mean=, std=) of SST over the training period
    Returns (C, H, W) float32.
    """
    mu, sd = stats["mean"], stats["std"]
    H, W = target_obs.shape
    lon_c = float(np.mean(lon))
    hist_times = pd.DatetimeIndex(hist_times)
    tt = pd.Timestamp(target_time)
    keep = (hist_times < tt) & (hist_times >= tt - pd.Timedelta(days=CONTEXT_DAYS))
    hist_obs, hist_times = hist_obs[keep], hist_times[keep]

    def norm(a):
        m = np.isfinite(a)
        return np.where(m, (a - mu) / sd, 0.0).astype("f4"), m.astype("f4")

    feats = []
    x, m = norm(target_obs)
    feats += [x, m]

    # the N_PREV most recent passes that saw anything
    seen = np.isfinite(hist_obs).reshape(len(hist_obs), -1).any(1) if len(hist_obs) else []
    idx = [i for i in range(len(hist_obs) - 1, -1, -1) if seen[i]][:N_PREV]
    for k in range(N_PREV):
        if k < len(idx):
            i = idx[k]
            x, m = norm(hist_obs[i])
            lag = (tt - hist_times[i]).total_seconds() / 86400.0 / CONTEXT_DAYS
            feats += [x, m, np.full((H, W), lag, "f4") * m]
        else:
            feats += [np.zeros((H, W), "f4")] * 3

    # composites: 30 d all passes, 30 d same day/night phase, last 7 days
    day_t = is_day([tt], lon_c)[0]
    day_h = is_day(hist_times, lon_c) if len(hist_times) else np.zeros(0, bool)
    recent = (hist_times >= tt - pd.Timedelta(days=7)) if len(hist_times) else np.zeros(0, bool)
    for sel in (np.ones(len(hist_obs), bool), day_h == day_t, recent):
        sub = hist_obs[sel]
        if len(sub):
            ok = np.isfinite(sub)
            cnt = ok.sum(0)
            s = np.where(ok, sub, 0).sum(0)
            mean = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
            x, m = norm(mean)
            frac = (cnt / max(len(sub), 1)).astype("f4")
            feats += [x, np.minimum(frac * 4, 1).astype("f4") * m]
        else:
            feats += [np.zeros((H, W), "f4")] * 2

    # static / time planes
    LAT = np.broadcast_to(((lat - 15) / 10)[:, None], (H, W))
    LON = np.broadcast_to(((lon - 80) / 10)[None, :], (H, W))
    doy = 2 * np.pi * tt.dayofyear / 365.25
    hr = 2 * np.pi * (tt.hour + tt.minute / 60) / 24
    for v in (LAT, LON):
        feats.append(np.asarray(v, "f4"))
    for v in (np.sin(doy), np.cos(doy), np.sin(hr), np.cos(hr)):
        feats.append(np.full((H, W), v, "f4"))
    out = np.stack(feats).astype("f4")
    assert out.shape[0] == n_channels(), out.shape
    return out


# ---------------------------------------------------------------------------
# Network: DINCAE-style convolutional auto-encoder with skip connections.
# Outputs a mean and a log-variance per pixel (Gaussian NLL), like DINCAE, but
# without DINCAE-1's dense bottleneck (which is >99% of its parameters and does
# not fit a 4 GB GPU); DINCAE 2 dropped it for the same reason.
# ---------------------------------------------------------------------------
def make_model(in_ch=None, base=32):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    in_ch = in_ch or n_channels()

    def block(i, o):
        return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1),
                             nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1))

    class GapFillNet(nn.Module):
        def __init__(self):
            super().__init__()
            f = [base, base * 2, base * 4, base * 8]
            self.enc = nn.ModuleList([block(in_ch, f[0]), block(f[0], f[1]),
                                      block(f[1], f[2]), block(f[2], f[3])])
            self.mid = block(f[3], f[3])
            self.dec = nn.ModuleList([block(f[3] + f[3], f[3]), block(f[3] + f[2], f[2]),
                                      block(f[2] + f[1], f[1]), block(f[1] + f[0], f[0])])
            self.head = nn.Conv2d(f[0], 2, 1)

        def forward(self, x):
            H, W = x.shape[-2:]
            ph, pw = (-H) % 16, (-W) % 16
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            skips = []
            for e in self.enc:
                x = e(x); skips.append(x); x = F.max_pool2d(x, 2)
            x = self.mid(x)
            for d, s in zip(self.dec, reversed(skips)):
                x = F.interpolate(x, size=s.shape[-2:], mode="bilinear", align_corners=False)
                x = d(torch.cat([x, s], 1))
            out = self.head(x)[..., :H, :W]
            mean = out[:, 0]
            logvar = out[:, 1].clamp(-10, 5)
            return mean, logvar

    return GapFillNet()
