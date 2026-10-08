"""
NOAA-18 / NOAA-19 AVHRR SST gap-filling web app.

    ..\\.venv\\Scripts\\python noaa_app.py        ->  http://localhost:8052

Pick the satellite, click a sector on the map (or pick it from the list), choose
a scene from the archive or upload a GeoTIFF (or the raw NOAA .nc) and press Run.
The gaps are filled by the model trained for that satellite and sector, given the
30 days of that satellite's passes before the file's timestamp as context.

The sector map is a zoomable basemap (ocean / light / satellite tiles, or the
bundled coastline when offline). The result is an interactive figure: observed,
gap-filled and the model's uncertainty side by side, with linked zoom and pan and
the value of every cell on hover.
"""
import os
import sys
import json
import uuid
import base64
import warnings
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback, clientside_callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "gapfill"))
import common as C

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEMP_DIR = os.path.join(HERE, "temp_app_files")
os.makedirs(TEMP_DIR, exist_ok=True)
COAST_SHAPEFILE_PATH = os.path.join(HERE, "assets", "indiacoast.shp")

# ---------------------------------------------------------------------------
# Coastline
# ---------------------------------------------------------------------------
COAST_GDF, COAST_XY = None, None
try:
    COAST_GDF = gpd.read_file(COAST_SHAPEFILE_PATH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)       # length in degrees is fine here
        _simple = COAST_GDF.geometry.simplify(0.01)
        _simple = _simple[_simple.length > 0.02]
    xs, ys = [], []
    for g in _simple:
        parts = g.geoms if g.geom_type == "MultiLineString" else [g]
        for p in parts:
            x, y = p.xy
            xs += [round(v, 3) for v in x] + [None]
            ys += [round(v, 3) for v in y] + [None]
    COAST_XY = (xs, ys)
except Exception as e:
    print(f"Warning: could not load coastline: {e}")

COAST_RES, LAND_COLOR, GAP_COLOR, SEEN_COLOR = 0.004, "#dcd5c6", "#ffffff", "#c9d1da"
DISPLAY_FACTOR = 3          # result maps are drawn at COAST_RES * 3 ~ 0.012 deg (3x the model grid)
_COAST_CACHE = {}


def coast_layers(sector_id, lon, lat, sea_mask):
    """Land/sea split of a sector on a fine grid, following the real coastline."""
    if sector_id in _COAST_CACHE:
        return _COAST_CACHE[sector_id]
    if COAST_GDF is None:
        return None
    from rasterio import features
    from rasterio.transform import from_bounds
    from scipy import ndimage
    from shapely.geometry import box
    dx, dy = abs(lon[1] - lon[0]), abs(lat[1] - lat[0])
    x0, x1 = lon.min() - dx / 2, lon.max() + dx / 2
    y0, y1 = lat.min() - dy / 2, lat.max() + dy / 2
    W, H = int(round((x1 - x0) / COAST_RES)), int(round((y1 - y0) / COAST_RES))
    lines = COAST_GDF.geometry.clip(box(x0, y0, x1, y1))
    lines = lines[~lines.is_empty]
    barrier = np.zeros((H, W), bool)
    if len(lines):
        barrier = features.rasterize(((g.buffer(COAST_RES * 1.5), 1) for g in lines),
                                     out_shape=(H, W), transform=from_bounds(x0, y0, x1, y1, W, H),
                                     fill=0, dtype="uint8").astype(bool)
    fx = x0 + (np.arange(W) + 0.5) * COAST_RES
    fy = y1 - (np.arange(H) + 0.5) * COAST_RES
    ii = np.interp(fx, lon, np.arange(len(lon)))
    jj = np.interp(fy, lat[::-1], np.arange(len(lat))[::-1])
    JJ, II = np.meshgrid(jj, ii, indexing="ij")
    nearest = (np.rint(JJ).astype(int), np.rint(II).astype(int))
    labels, n = ndimage.label(~barrier)
    sea = np.zeros((H, W), bool)
    if n:
        frac = ndimage.mean(sea_mask[nearest], labels, index=np.arange(1, n + 1))
        sea[~barrier] = frac[labels[~barrier] - 1] >= 0.5
    lx, ly = [], []
    for g in lines:
        for p in (g.geoms if g.geom_type.startswith("Multi") else [g]):
            x, y = p.xy
            lx += [round(v, 4) for v in x] + [None]
            ly += [round(v, 4) for v in y] + [None]
    out = dict(extent=(x0, x1, y0, y1), sea=sea, land=~sea & ~barrier,
               coords=(JJ, II), nearest=nearest, lines=lines, line_xy=(lx, ly))
    _COAST_CACHE[sector_id] = out
    return out


def to_fine_grid(field, layers, fill_gaps):
    from scipy import ndimage
    missing = np.isnan(field)
    dense = field
    if missing.any() and not missing.all():
        idx = ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True)
        dense = field[tuple(idx)]
    fine = ndimage.map_coordinates(dense, layers["coords"], order=1, mode="nearest")
    if not fill_gaps:
        fine[missing[layers["nearest"]]] = np.nan
    fine[~layers["sea"]] = np.nan
    return fine


def block_mean(a, f):
    """NaN-aware f x f block average (fine coastline grid -> display grid)."""
    H, W = (a.shape[0] // f) * f, (a.shape[1] // f) * f
    b = a[:H, :W].reshape(H // f, f, W // f, f)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN blocks (land) stay NaN
        return np.nanmean(b, axis=(1, 3)).astype("f4")


# ---------------------------------------------------------------------------
# Models and archives
# ---------------------------------------------------------------------------
_MODELS, _ARCHIVES, _SUMMARY = {}, {}, {}


def model_path(sat, sid):
    return os.path.join(C.MODEL_DIR, sat, f"{C.SECTORS[sid]['name']}.pt")


def archive_path(sat, sid):
    return os.path.join(C.ARCHIVE_DIR, sat, f"{C.SECTORS[sid]['name']}.nc")


def load_model(sat, sid):
    key = (sat, sid)
    if key not in _MODELS:
        p = model_path(sat, sid)
        if not os.path.exists(p):
            raise ValueError(f"No trained {C.SATELLITES[sat]} model for sector {sid}. "
                             f"Run gapfill/train.py first.")
        ck = torch.load(p, map_location=DEVICE, weights_only=False)
        m = C.make_model(ck["n_channels"]).to(DEVICE)
        m.load_state_dict(ck["state_dict"])
        m.eval()
        _MODELS[key] = (m, ck["stats"])
    return _MODELS[key]


def load_archive(sat, sid):
    key = (sat, sid)
    if key not in _ARCHIVES:
        with xr.open_dataset(archive_path(sat, sid)) as ds:
            _ARCHIVES[key] = (pd.DatetimeIndex(ds.time.values), ds.SST.values.astype("f4"),
                              ds.lat.values, ds.lon.values, ds.mask.values == 1)
    return _ARCHIVES[key]


def load_metrics(sat, sid):
    p = model_path(sat, sid).replace(".pt", "_metrics.json")
    return json.load(open(p)) if os.path.exists(p) else None


def sector_summary(sat, sid):
    """Passes, time span and mean observed fraction of a sector's archive (None if absent)."""
    key = (sat, sid)
    if key not in _SUMMARY:
        if not os.path.exists(archive_path(sat, sid)):
            return None
        times, sst, _, _, sea = load_archive(sat, sid)
        cov = np.isfinite(sst[:, sea]).mean(1)
        _SUMMARY[key] = dict(n=len(times), t0=times[0], t1=times[-1], cov=float(cov.mean()))
    return _SUMMARY[key]


def model_skill(m):
    """(rmse, 30-day composite rmse, n passes, 'test'|'validation') of a metrics dict."""
    kind = "test" if m["test"]["n_passes"] else "validation"
    s = m[kind]
    return s["rmse"], s["rmse_30day_composite"], s["n_passes"], kind


def read_upload(data, filename, lat, lon):
    """Uploaded GeoTIFF / NOAA netCDF -> SST (deg C) on the sector grid."""
    low = filename.lower()
    if low.endswith((".tif", ".tiff")):
        import rasterio
        from rasterio.io import MemoryFile
        try:
            with MemoryFile(data) as mf, mf.open() as src:
                arr = src.read(1, masked=True).astype("f4").filled(np.nan)
                t = src.transform
                if t.b != 0 or t.d != 0:
                    raise ValueError("rotated GeoTIFFs are not supported")
                flon = t.c + (np.arange(src.width) + 0.5) * t.a
                flat = t.f + (np.arange(src.height) + 0.5) * t.e
        except rasterio.errors.RasterioIOError:
            raise ValueError("Failed to read file. Ensure it is a valid GeoTIFF.")
    elif low.endswith(".nc"):
        import netCDF4
        with netCDF4.Dataset("upload", memory=data) as d:
            v = next(k for k in d.variables if k.lower() in ("sst", "sea_surface_temperature"))
            var = d[v]; var.set_auto_mask(False)
            arr = np.squeeze(var[:]).astype("f4")
            fill = getattr(var, "_FillValue", None)
            if fill is not None:
                arr[arr == fill] = np.nan
            flat, flon = d["lat"][:].data, d["lon"][:].data
    else:
        raise ValueError(f"Unsupported file type '{filename}'. Upload a .tif or .nc file.")
    arr[(arr < -50) | (arr > 400)] = np.nan
    if np.nanmedian(arr) > 200:              # Kelvin
        arr -= 273.15
    if flat[0] < flat[-1]:                   # make latitude descending like the sector grid
        flat, arr = flat[::-1], arr[::-1]
    b = C.Binner(flat, flon, lat, lon)
    return b(arr[b.r0:b.r1, b.c0:b.c1])


def run_inference(data, filename, sat, sid):
    """Fill an uploaded file; its time comes from the file name."""
    _, _, lat, lon, _ = load_archive(sat, sid)
    return fill_scene(read_upload(data, filename, lat, lon), C.parse_time(filename), sat, sid)


def run_archive_scene(time_iso, sat, sid):
    """Fill one pass of the archive, using only the 30 days before it."""
    times, sst, _, _, _ = load_archive(sat, sid)
    ts = pd.Timestamp(time_iso)
    i = times.get_loc(ts)
    return fill_scene(sst[i].copy(), ts, sat, sid)


def scene_options(sat, sid):
    """{date: [(time_iso, 'HH:MM NN% of sea observed'), ...]} for one satellite and sector."""
    times, sst, _, _, sea = load_archive(sat, sid)
    cov = np.isfinite(sst[:, sea]).mean(1)
    out = {}
    for t, c in zip(times, cov):
        out.setdefault(f"{t:%Y-%m-%d}", []).append((t.isoformat(), f"{t:%H:%M} {c:.0%} of sea observed"))
    return out


# Reconstructions are kept on the server (the browser only holds the id), so the
# result figure can be re-drawn with another view / colour map without re-running.
RESULTS = OrderedDict()
MAX_RESULTS = 12


def fill_scene(obs, ts, sat, sid):
    model, stats = load_model(sat, sid)
    times, sst, lat, lon, sea = load_archive(sat, sid)
    obs[~sea] = np.nan
    n_obs = int(np.isfinite(obs[sea]).sum())
    if n_obs == 0:
        raise ValueError(f"The scene has no valid SST over sector {sid} "
                         f"({C.SECTORS[sid]['label']}).")

    lo = times.searchsorted(ts - pd.Timedelta(days=C.CONTEXT_DAYS))
    hi = times.searchsorted(ts)                    # strictly before the upload
    hist, htimes = sst[lo:hi], times[lo:hi]
    if len(hist) == 0:
        raise ValueError(f"No {C.SATELLITES[sat]} passes in the {C.CONTEXT_DAYS} days before "
                         f"{ts:%d-%b-%Y %H:%M}. The archive covers "
                         f"{times[0]:%d-%b-%Y} to {times[-1]:%d-%b-%Y}.")

    x = C.build_features(obs, ts, hist, htimes, stats, lat, lon)
    with torch.no_grad():
        mean, logvar = model(torch.from_numpy(x[None]).to(DEVICE))
    rec = mean[0].cpu().numpy() * stats["std"] + stats["mean"]
    sigma = (np.sqrt(np.exp(logvar[0].cpu().numpy())) * stats["std"]).astype("f4")
    filled = np.where(np.isfinite(obs), obs, rec).astype("f4")
    filled[~sea] = np.nan

    uid = uuid.uuid4().hex[:8]
    dl_path = os.path.join(TEMP_DIR, f"filled_{sat}_{sid}_{uid}.nc")
    xr.Dataset({"SST": (("lat", "lon"), filled), "SST_original": (("lat", "lon"), obs),
                "SST_error_std": (("lat", "lon"), np.where(sea, sigma, np.nan).astype("f4"))},
               coords={"lat": lat, "lon": lon},
               attrs={"satellite": C.SATELLITES[sat], "sector": C.SECTORS[sid]["name"],
                      "time": str(ts), "context_passes": len(hist)}).to_netcdf(dl_path)

    gap = sea & ~np.isfinite(obs)
    cov = 100.0 * n_obs / sea.sum()
    res = dict(uid=uid, sat=sat, sid=sid, ts=ts, lat=lat, lon=lon, sea=sea, obs=obs,
               filled=filled, sigma=sigma, gap=gap, cov=cov, dl=dl_path,
               n_ctx=len(hist), ctx_days=len(np.unique(htimes.normalize())),
               ctx_span=(htimes[0], htimes[-1]))
    res["display"] = display_fields(res)
    RESULTS[uid] = res
    while len(RESULTS) > MAX_RESULTS:
        RESULTS.popitem(last=False)
    return res


def display_fields(res):
    """Arrays the result figure draws, rows ordered south -> north.

    The pictures follow the real coastline at ~0.012 deg (the model grid upsampled
    like the original matplotlib view); hover reads the model's own ~0.036 deg
    cells, so the values shown are exactly what the NetCDF contains."""
    sid, lat, lon, sea = res["sid"], res["lat"], res["lon"], res["sea"]
    obs, filled, sigma, gap = res["obs"], res["filled"], res["sigma"], res["gap"]
    sig_gap = np.where(gap, sigma, np.nan).astype("f4")
    layers = coast_layers(sid, lon, lat, sea)
    if layers is None:                                  # no coastline: draw the model grid
        xd, yd = lon, lat
        obs_d, fill_d, sig_d = obs, filled, sig_gap
        line_xy = ([], [])
    else:
        F = DISPLAY_FACTOR
        obs_d = block_mean(to_fine_grid(obs, layers, False), F)
        fill_d = block_mean(to_fine_grid(filled, layers, True), F)
        sig_d = block_mean(to_fine_grid(sig_gap, layers, False), F)
        x0, x1, y0, y1 = layers["extent"]
        r = COAST_RES * F
        xd = x0 + (np.arange(fill_d.shape[1]) + 0.5) * r
        yd = y1 - (np.arange(fill_d.shape[0]) + 0.5) * r
        line_xy = layers["line_xy"]
    flip = lambda a: np.ascontiguousarray(a[::-1])     # north-up rows -> ascending latitude
    obs_d, fill_d, sig_d, yd = flip(obs_d), flip(fill_d), flip(sig_d), yd[::-1].copy()
    sea_d = np.isfinite(fill_d)

    # hover text on the model grid
    o, f, s, g, se = (flip(a) for a in (obs, filled, sigma, gap, sea))
    fmt = lambda a: np.char.mod("%.2f", np.nan_to_num(a))
    obs_txt = np.where(se, np.where(g, "gap · no observation",
                                    np.char.add(fmt(o), " °C · observed")), "land")
    fill_txt = np.where(se, np.where(g, np.char.add(np.char.add(fmt(f), " °C · model fill  σ ±"),
                                                    fmt(s)),
                                     np.char.add(fmt(o), " °C · observed")), "land")
    sig_txt = np.where(se, np.where(g, np.char.add(np.char.add("σ ±", fmt(s)), " °C · model fill"),
                                    "observed · kept as is"), "land")
    return dict(x=xd, y=yd, obs=obs_d, filled=fill_d, sigma=sig_d,
                gap=(sea_d & ~np.isfinite(obs_d)).astype("u1"),
                seen=(sea_d & ~np.isfinite(sig_d)).astype("u1"),
                hx=lon, hy=lat[::-1].copy(), hz=np.where(se, 0.0, np.nan).astype("f4"),
                gapmask=g.astype("f4"),
                text=dict(obs=obs_txt, filled=fill_txt, sigma=sig_txt), line_xy=line_xy)


def result_stats(res):
    sea, gap, sigma, filled = res["sea"], res["gap"], res["sigma"], res["filled"]
    lat, lon = res["lat"], res["lon"]
    cell_km2 = (abs(lat[1] - lat[0]) * 111.32) * (abs(lon[1] - lon[0]) * 111.32
                                                   * np.cos(np.deg2rad(lat.mean())))
    v = filled[sea]
    return dict(observed=res["cov"], n_gap=int(gap.sum()), area=gap.sum() * cell_km2,
                sig_mean=float(sigma[gap].mean()) if gap.any() else None,
                sig_max=float(sigma[gap].max()) if gap.any() else None,
                p2=float(np.percentile(v, 2)), p98=float(np.percentile(v, 98)))


# ---------------------------------------------------------------------------
# Result figure
# ---------------------------------------------------------------------------
COLORMAPS = {"Turbo": "Turbo", "Jet (classic)": "Jet", "Thermal": "thermal",
             "Red-Yellow-Blue": "RdYlBu_r", "Viridis": "Viridis"}
VIEWS = {"all": ["obs", "filled", "sigma"], "compare": ["obs", "filled"],
         "filled": ["filled"], "sigma": ["sigma"]}
PANEL_TITLES = {"obs": "Observed (gaps in white)", "filled": "Gap-filled",
                "sigma": "Uncertainty of the fill (σ)"}
CLEAR = [[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]]


def default_range(res):
    d = res["display"]["filled"]
    v = d[np.isfinite(d)]
    lo, hi = np.percentile(v, 2), np.percentile(v, 98)
    smin, smax = np.floor(np.percentile(v, 0.1) - 0.5), np.ceil(np.percentile(v, 99.9) + 0.5)
    return [round(float(lo), 1), round(float(hi), 1)], float(smin), float(smax)


FIG_W = 1150      # ~plot width of the result card on a 1920 px wide screen


H_SPACE, V_SPACE = 0.06, 0.08    # gaps between panels, as a fraction of the plot area


def figure_shape(d, n):
    """Panels side by side, or stacked for wide flat sectors (Goa), plus the sector's
    on-screen aspect (height / width). The browser sets the final height from the real
    width (see fit_result_height), so the panels fill the figure at any screen size."""
    lat_c = float(np.mean(d["y"]))
    aspect = ((d["y"][-1] - d["y"][0]) /
              ((d["x"][-1] - d["x"][0]) * np.cos(np.deg2rad(lat_c))))
    return (n > 1 and aspect < 0.45), float(aspect)


def fit_height(width, n, stacked, aspect, l, r, t, b):
    """Figure height at which n panels of this aspect exactly fill `width` (same formula
    as the clientside fit_result_height)."""
    plot_w = width - l - r
    if stacked:
        h = n * plot_w * aspect / (1 - V_SPACE * (n - 1))
    else:
        h = plot_w * (1 - H_SPACE * (n - 1)) / n * aspect
    return int(np.clip(h + t + b, 300, 1000))


def result_figure(res, view, cmap, zrange, outline):
    d = res["display"]
    panels = VIEWS.get(view, VIEWS["all"])
    n = len(panels)
    stacked, aspect = figure_shape(d, n)
    l, r, t, b = 60, (90 if stacked else 20), 40, (50 if stacked else 110)
    height = fit_height(FIG_W, n, stacked, aspect, l, r, t, b)     # first guess; the browser refits
    titles = [PANEL_TITLES[p] + (f" · {res['cov']:.0f}% observed" if p == "obs" else "")
              for p in panels]
    if stacked:
        fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=V_SPACE,
                            subplot_titles=titles)
    else:
        fig = make_subplots(rows=1, cols=n, shared_yaxes=True,
                            horizontal_spacing=H_SPACE if n > 1 else 0, subplot_titles=titles)
    cell = (lambda k: (k, 1)) if stacked else (lambda k: (1, k))
    x, y = d["x"], d["y"]
    for k, p in enumerate(panels, 1):
        r, c = cell(k)
        if p == "obs":
            fig.add_trace(go.Heatmap(x=x, y=y, z=d["gap"], zmin=0, zmax=1, showscale=False,
                                     colorscale=[[0, "rgba(0,0,0,0)"], [1, GAP_COLOR]],
                                     hoverinfo="skip"), r, c)
            fig.add_trace(go.Heatmap(x=x, y=y, z=d["obs"], coloraxis="coloraxis",
                                     hoverinfo="skip"), r, c)
        elif p == "filled":
            fig.add_trace(go.Heatmap(x=x, y=y, z=d["filled"], coloraxis="coloraxis",
                                     hoverinfo="skip"), r, c)
        else:
            fig.add_trace(go.Heatmap(x=x, y=y, z=d["seen"], zmin=0, zmax=1, showscale=False,
                                     colorscale=[[0, "rgba(0,0,0,0)"], [1, SEEN_COLOR]],
                                     hoverinfo="skip"), r, c)
            fig.add_trace(go.Heatmap(x=x, y=y, z=d["sigma"], coloraxis="coloraxis2",
                                     hoverinfo="skip"), r, c)
        if outline and p in ("filled", "sigma") and res["gap"].any():
            fig.add_trace(go.Contour(x=d["hx"], y=d["hy"], z=d["gapmask"], showscale=False,
                                     contours=dict(start=0.5, end=0.5, size=1, coloring="none"),
                                     line=dict(color="rgba(15,15,15,0.85)", width=1.1),
                                     hoverinfo="skip"), r, c)
        lx, ly = d["line_xy"]
        if lx:
            fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", hoverinfo="skip",
                                     line=dict(color="#2b2b2b", width=1)), r, c)
        # invisible layer on the model grid that carries the hover values
        fig.add_trace(go.Heatmap(x=d["hx"], y=d["hy"], z=d["hz"], text=d["text"][p],
                                 colorscale=CLEAR, showscale=False,
                                 hovertemplate="<b>%{text}</b><br>%{y:.3f}°N  %{x:.3f}°E"
                                               "<extra></extra>"), r, c)

    lat_c = float(np.mean(y))
    fig.update_xaxes(range=[x[0] - (x[1] - x[0]) / 2, x[-1] + (x[1] - x[0]) / 2],
                     constrain="domain", ticksuffix="°E", showgrid=True, nticks=5,
                     gridcolor="rgba(0,0,0,0.07)", zeroline=False, showline=True,
                     linecolor="#8a96a3", mirror=True, ticks="outside")
    fig.update_yaxes(range=[y[0] - (y[1] - y[0]) / 2, y[-1] + (y[1] - y[0]) / 2],
                     constrain="domain", ticksuffix="°N", showgrid=True,
                     gridcolor="rgba(0,0,0,0.07)", zeroline=False, showline=True,
                     linecolor="#8a96a3", mirror=True, ticks="outside")
    # true proportions: 1 deg of latitude is 1/cos(lat) times longer than 1 deg of longitude
    fig.update_yaxes(scaleanchor="x", scaleratio=1 / np.cos(np.deg2rad(lat_c)), row=1, col=1)
    for k in range(2, n + 1):                      # all panels zoom and pan together
        fig.update_xaxes(matches="x", row=cell(k)[0], col=cell(k)[1])
        fig.update_yaxes(matches="y", row=cell(k)[0], col=cell(k)[1])

    def cbar(panel_ids, title):
        axis = "yaxis" if stacked else "xaxis"
        doms = [fig.layout[f"{axis}{'' if panels.index(p) == 0 else panels.index(p) + 1}"].domain
                for p in panel_ids]
        a, b_ = min(dm[0] for dm in doms), max(dm[1] for dm in doms)
        if stacked:                                 # vertical bar beside its rows
            return dict(x=1.01, xanchor="left", y=(a + b_) / 2, yanchor="middle",
                        len=(b_ - a) * 0.9, thickness=12, title=dict(text=title, side="right"),
                        outlinewidth=0)
        # horizontal bar pinned to the bottom of the figure, under its panels
        return dict(orientation="h", x=(a + b_) / 2, xanchor="center", yref="container", y=0.01,
                    yanchor="bottom", len=min(0.75, (b_ - a) * 0.7), thickness=12,
                    title=dict(text=title, side="right"), outlinewidth=0)

    sst_panels = [p for p in panels if p in ("obs", "filled")]
    lo, hi = zrange
    fig.update_layout(
        coloraxis=dict(colorscale=cmap, cmin=lo, cmax=hi,
                       colorbar=cbar(sst_panels, "SST (°C)") if sst_panels else None,
                       showscale=bool(sst_panels)),
        coloraxis2=dict(colorscale="YlOrRd", cmin=0,
                        cmax=max(0.2, float(np.nanpercentile(d["sigma"], 98)))
                        if np.isfinite(d["sigma"]).any() else 1,
                        colorbar=cbar(["sigma"], "σ (°C)") if "sigma" in panels else None,
                        showscale="sigma" in panels),
        plot_bgcolor=LAND_COLOR, paper_bgcolor="white", showlegend=False,
        height=height, margin=dict(l=l, r=r, t=t, b=b), dragmode="zoom",
        meta=dict(fit=dict(key=f"{res['uid']}-{view}", n=n, stacked=stacked, aspect=aspect,
                           l=l, r=r, t=t, b=b, hs=H_SPACE, vs=V_SPACE)),
        hoverlabel=dict(bgcolor="white", font_size=12), uirevision=res["uid"],
        font=dict(family="Nunito Sans, Segoe UI, sans-serif", size=12))
    fig.update_annotations(font_size=13)
    return fig


# ---------------------------------------------------------------------------
# Sector map
# ---------------------------------------------------------------------------
def _sector_geometry():
    """Sector polygons as GeoJSON. A box that contains a smaller one (Odisha around
    West Bengal) gets that area cut out, so a click always lands on one sector."""
    from shapely.geometry import box, mapping
    from shapely.ops import polylabel
    boxes = {sid: box(s["bounds"][0][1], s["bounds"][0][0], s["bounds"][1][1], s["bounds"][1][0])
             for sid, s in C.SECTORS.items()}
    feats, label_pts, rings = [], {}, {}
    for sid, b in boxes.items():
        p = b
        for oid, o in boxes.items():
            if oid != sid and o.area < b.area and o.intersects(b):
                p = p.difference(o)
        if p.geom_type != "Polygon":
            p = max(p.geoms, key=lambda g: g.area)
        feats.append({"type": "Feature", "id": sid, "properties": {"id": sid},
                      "geometry": json.loads(json.dumps(mapping(p)))})
        lp = polylabel(p, tolerance=0.01)
        label_pts[sid] = (lp.x, lp.y)
        rx, ry = b.exterior.xy
        rings[sid] = (list(rx), list(ry))
    return {"type": "FeatureCollection", "features": feats}, label_pts, rings


SECTOR_GEOJSON, LABEL_PTS, SECTOR_RINGS = _sector_geometry()
SECTOR_OPTIONS = [{"label": f"{sid} · {s['label']}", "value": sid} for sid, s in C.SECTORS.items()]

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/{}/MapServer/tile/{{z}}/{{y}}/{{x}}"
BASEMAPS = {  # value: (label, map style, raster tile url, attribution)
    "ocean": ("Ocean", "white-bg", ESRI.format("Ocean/World_Ocean_Base"),
              "Tiles © Esri — GEBCO, NOAA, National Geographic"),
    "light": ("Light", "carto-positron", None, None),
    "satellite": ("Satellite", "white-bg", ESRI.format("World_Imagery"),
                  "Tiles © Esri — Maxar, Earthstar Geographics"),
    "offline": ("Offline", "white-bg", None, None),
}
STATUS_COLORS = {"untrained": "#f59e0b", "trained": "#10b981", "selected": "#2563eb"}
MAP_W, MAP_H = 560, 430
DEFAULT_VIEW = {"center": {"lat": 14.6, "lon": 81.2}, "zoom": 3.4, "rev": "all"}


def fit_view(sid):
    (y0, x0), (y1, x1) = C.SECTORS[sid]["bounds"]
    merc = lambda la: np.log(np.tan(np.pi / 4 + np.deg2rad(la) / 2))
    zx = np.log2(MAP_W * 360 / (512 * (x1 - x0)))
    zy = np.log2(MAP_H * 2 * np.pi / (512 * (merc(y1) - merc(y0))))
    return {"center": {"lat": (y0 + y1) / 2, "lon": (x0 + x1) / 2},
            "zoom": float(min(zx, zy) - 0.45), "rev": uuid.uuid4().hex[:6]}


def sector_hover(sat, sid):
    s, m, a = C.SECTORS[sid], load_metrics(sat, sid), sector_summary(sat, sid)
    out = [f"<b>{sid} · {s['label']}</b>"]
    if m:
        rmse, base, npass, kind = model_skill(m)
        out += [f"{C.SATELLITES[sat]} model ✓ trained",
                f"{kind} RMSE <b>{rmse:.2f} °C</b>  (−{100 * (1 - rmse / base):.0f}%)",
                f"30-day composite: {base:.2f} °C"]
    else:
        out.append(f"{C.SATELLITES[sat]} model not trained yet")
    if a:
        out += [f"archive: {a['n']:,} passes, {a['t0']:%b %Y} – {a['t1']:%b %Y}",
                f"{a['cov']:.0%} of the sea observed on average"]
    out.append("<i>click to select</i>")
    return "<br>".join(out)


def build_map(selected=None, sat="n18", color_by="status", basemap="ocean", view=None):
    view = view or DEFAULT_VIEW
    sids = list(C.SECTORS)
    hover = {sid: sector_hover(sat, sid) for sid in sids}
    _, style, url, attrib = BASEMAPS.get(basemap, BASEMAPS["ocean"])
    dark = basemap == "satellite"
    common = dict(geojson=SECTOR_GEOJSON, featureidkey="properties.id",
                  hovertemplate="%{customdata}<extra></extra>")
    traces = []
    if basemap == "offline" and COAST_XY:
        traces.append(go.Scattermap(lon=COAST_XY[0], lat=COAST_XY[1], mode="lines",
                                    line=dict(color="#475569", width=1), hoverinfo="skip"))
    edge = dict(color="rgba(255,255,255,0.9)" if dark else "#1e3a5f", width=1.2)
    if color_by == "skill":
        skill = {}
        for sid in sids:
            m = load_metrics(sat, sid)
            if m:
                skill[sid] = model_skill(m)[0]
        done = [s for s in sids if s in skill]
        rest = [s for s in sids if s not in skill]
        if done:
            traces.append(go.Choroplethmap(
                locations=done, z=[skill[s] for s in done], customdata=[hover[s] for s in done],
                colorscale="RdYlGn_r", zmin=0.4, zmax=1.0, marker=dict(opacity=0.62, line=edge),
                colorbar=dict(title=dict(text="RMSE °C", side="top"), thickness=11, len=0.5,
                              x=0.985, xanchor="right", y=0.03, yanchor="bottom",
                              bgcolor="rgba(255,255,255,0.85)", outlinewidth=0,
                              tickvals=[0.4, 0.6, 0.8, 1.0]),
                **common))
        if rest:
            traces.append(go.Choroplethmap(
                locations=rest, z=[0] * len(rest), customdata=[hover[s] for s in rest],
                colorscale=[[0, "#94a3b8"], [1, "#94a3b8"]], showscale=False,
                marker=dict(opacity=0.5, line=edge), **common))
    else:
        u, t = STATUS_COLORS["untrained"], STATUS_COLORS["trained"]
        rest = [sid for sid in sids if sid != selected]
        traces.append(go.Choroplethmap(
            locations=rest, customdata=[hover[sid] for sid in rest], zmin=0, zmax=1,
            z=[1 if os.path.exists(model_path(sat, sid)) else 0 for sid in rest],
            colorscale=[[0, u], [0.5, u], [0.5, t], [1, t]],
            showscale=False, marker=dict(opacity=0.45, line=edge), **common))
        if selected:                                # light fill: the basemap stays readable
            traces.append(go.Choroplethmap(
                locations=[selected], z=[1], customdata=[hover[selected]], showscale=False,
                colorscale=[[0, STATUS_COLORS["selected"]], [1, STATUS_COLORS["selected"]]],
                marker=dict(opacity=0.22, line=edge), **common))
    if selected:
        rx, ry = SECTOR_RINGS[selected]
        traces.append(go.Scattermap(lon=rx, lat=ry, mode="lines", hoverinfo="skip",
                                    line=dict(color="#fde047" if dark else "#1d4ed8", width=4)))
    traces.append(go.Scattermap(
        lon=[LABEL_PTS[s][0] for s in sids], lat=[LABEL_PTS[s][1] for s in sids],
        mode="text", text=[C.SECTORS[s]["label"] for s in sids], hoverinfo="skip",
        textfont=dict(size=11, color="#ffffff" if dark else "#0f2744")))

    layers = ([dict(below="traces", sourcetype="raster", source=[url], sourceattribution=attrib)]
              if url else [])
    fig = go.Figure(traces)
    fig.update_layout(map=dict(style=style, center=view["center"], zoom=view["zoom"], layers=layers),
                      margin=dict(l=0, r=0, t=0, b=0), height=MAP_H, showlegend=False,
                      uirevision=view["rev"], clickmode="event",
                      hoverlabel=dict(bgcolor="white", font_size=12, align="left"))
    return fig


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])
app.title = "NOAA SST Gap-Filling Tool"
app.index_string = '''<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
.rounded-box{border:2px solid #005c97!important;border-radius:15px!important;
  box-shadow:0 4px 8px rgba(0,0,0,.1)!important;overflow:hidden}
.seg .btn{padding:.3rem .7rem;font-size:.72rem}
.step{display:inline-flex;align-items:center;justify-content:center;width:1.5rem;height:1.5rem;
  border-radius:50%;background:#005c97;color:#fff;font-size:.75rem;margin-right:.5rem}
.stat{border:1px solid #e3e8ee;border-radius:10px;padding:.5rem .75rem;background:#f8fafc;height:100%}
.stat .v{font-size:1.15rem;font-weight:700;color:#0f2744;letter-spacing:0}
.stat .k{font-size:.68rem;text-transform:uppercase;color:#64748b;letter-spacing:.06em}
.stat .s{font-size:.72rem;color:#64748b}
.legend-dot{display:inline-block;width:.8rem;height:.8rem;border-radius:3px;margin:0 .3rem 0 .8rem;
  vertical-align:-1px;opacity:.8}
.map-tools .form-select{font-size:.8rem;padding:.3rem 2rem .3rem .6rem}
.map-tools .btn{font-size:.72rem;padding:.35rem .6rem}
</style></head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>'''

header = html.Div(
    [html.H1("NOAA AVHRR SST Gap-Filling Tool", className="text-white text-center fw-bold mb-0"),
     html.Div("NOAA-18 | NOAA-19  ·  pick a sector, pick a scene, fill the clouds",
              className="text-white-50 text-center pb-2")],
    style={"background": "linear-gradient(90deg, #005c97 0%, #363795 100%)", "padding": "16px",
           "boxShadow": "0 4px 6px -1px rgba(0,0,0,0.3)", "marginBottom": "20px"})


def step(n, text):
    return html.Span([html.Span(str(n), className="step"), text])


def stat(key, value, sub=""):
    return dbc.Col(html.Div([html.Div(key, className="k"), html.Div(value, className="v"),
                             html.Div(sub, className="s")], className="stat"), xs=6, md=True)


map_card = dbc.Card([
    dbc.CardHeader(html.Div([step(1, "Satellite & sector"),
                             dbc.Badge(id="map-sat-label", color="light", text_color="primary",
                                       className="ms-auto")],
                            className="d-flex align-items-center"), className="bg-light fw-bold"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(dbc.RadioItems(
                id="sat-radio", className="btn-group seg",
                inputClassName="btn-check", labelClassName="btn btn-outline-primary",
                labelCheckedClassName="active",
                options=[{"label": "NOAA-18", "value": "n18"},
                         {"label": "NOAA-19", "value": "n19"}], value="n18"), width="auto"),
            dbc.Col(dcc.Dropdown(id="sector-dd", options=SECTOR_OPTIONS,
                                 placeholder="Search a sector…", clearable=False), width=True),
        ], className="g-2 mb-2 align-items-center map-tools"),
        dcc.Graph(id="sector-map", figure=build_map(),
                  config={"displayModeBar": False, "scrollZoom": True},
                  style={"height": f"{MAP_H}px", "width": "100%", "borderRadius": "8px",
                         "overflow": "hidden"}),
        dbc.Row([
            dbc.Col(dbc.RadioItems(
                id="map-color", value="status", className="btn-group seg",
                inputClassName="btn-check", labelClassName="btn btn-outline-secondary",
                labelCheckedClassName="active",
                options=[{"label": "Status", "value": "status"},
                         {"label": "Skill", "value": "skill"}]), width="auto"),
            dbc.Col(dbc.Select(id="map-basemap", value="ocean", size="sm",
                               options=[{"label": v[0], "value": k}
                                        for k, v in BASEMAPS.items()]), width=True),
            dbc.Col(dbc.ButtonGroup([
                dbc.Button("⌖ Sector", id="btn-zoom-sector", outline=True, color="primary",
                           size="sm", title="Zoom to the selected sector"),
                dbc.Button("⤢ All", id="btn-show-all", outline=True, color="primary", size="sm",
                           title="Show all sectors")]), width="auto"),
        ], className="g-2 mt-2 align-items-center map-tools"),
        html.Div(id="map-legend", className="small text-muted mt-2"),
        html.Div(id="selection-output", className="small fw-bold text-primary mt-1"),
    ])], className="rounded-box mb-3")

model_card = dbc.Card([
    dbc.CardHeader("Model", className="bg-light fw-bold"),
    dbc.CardBody(html.Div(id="model-info", className="small",
                          children=html.Span("Select a sector to see its model.",
                                             className="text-muted"))),
], className="rounded-box mb-3")

# Step 2 sits right above the result, so picking a scene and pressing Run needs no scrolling
scene_card = dbc.Card([
    dbc.CardHeader(step(2, "Scene"), className="bg-light fw-bold"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([html.Div("Source", className="small fw-bold text-muted"),
                     dbc.RadioItems(id="source-radio", value="archive", className="btn-group seg",
                                    inputClassName="btn-check",
                                    labelClassName="btn btn-outline-primary",
                                    labelCheckedClassName="active",
                                    options=[{"label": "Archive", "value": "archive"},
                                             {"label": "Upload", "value": "upload"}])],
                    width="auto"),
            dbc.Col([
                html.Div(id="archive-box", children=dbc.Row([
                    dbc.Col([html.Div("Date", className="small fw-bold text-muted"),
                             dcc.Dropdown(id="date-dd", placeholder="Select a sector first")],
                            md=5),
                    dbc.Col([html.Div("Time (UTC)", className="small fw-bold text-muted"),
                             dcc.Dropdown(id="time-dd", placeholder="Select a date")], md=7),
                ], className="g-2")),
                html.Div(id="upload-box", style={"display": "none"}, children=[
                    html.Div("GeoTIFF (°C or K) or raw NOAA .nc, time in the file name, "
                             "e.g. 20161015-120450Z-noaa-18-sst.tif",
                             className="small fw-bold text-muted"),
                    dcc.Upload(id="upload-data",
                               children=html.Div(["Drag & drop or ", html.A("select a file")]),
                               style={"width": "100%", "height": "38px", "lineHeight": "36px",
                                      "borderWidth": "2px", "borderStyle": "dashed",
                                      "borderRadius": "6px", "textAlign": "center"}),
                ]),
            ], width=True),
            dbc.Col([html.Div(" ", className="small"),
                     dbc.Button("▶ Run reconstruction", id="run-btn", color="primary",
                                disabled=True)], width="auto"),
        ], className="g-3 align-items-start"),
        html.Div([html.Span(id="upload-status", className="me-3"),
                  html.Span(f"The model fills the gaps from the {C.CONTEXT_DAYS} days of the "
                            f"satellite's passes before the scene.")],
                 className="small text-muted mt-2"),
        dcc.Loading(type="circle", children=html.Div(id="loading-output")),
    ])], className="rounded-box mb-3")

result_controls = html.Div([
    dbc.Row([
        dbc.Col([html.Div("View", className="small fw-bold text-muted"),
                 dbc.RadioItems(id="res-view", value="all", className="btn-group seg",
                                inputClassName="btn-check", labelClassName="btn btn-outline-primary",
                                labelCheckedClassName="active",
                                options=[{"label": "All three", "value": "all"},
                                         {"label": "Observed | filled", "value": "compare"},
                                         {"label": "Filled", "value": "filled"},
                                         {"label": "Uncertainty", "value": "sigma"}])], width="auto"),
        dbc.Col([html.Div("Colour map", className="small fw-bold text-muted"),
                 dbc.Select(id="res-cmap", value="Turbo", size="sm", style={"minWidth": "170px"},
                            options=[{"label": k, "value": v} for k, v in COLORMAPS.items()])],
                width="auto"),
        dbc.Col([html.Div("Gap outline", className="small fw-bold text-muted"),
                 dbc.Switch(id="res-outline", value=False, className="mt-1")], width="auto"),
    ], className="g-3 align-items-start"),
    html.Div("SST colour range (°C)", className="small fw-bold text-muted mt-2"),
    dcc.RangeSlider(id="res-range", min=20, max=32, step=0.1, value=[25, 30],
                    marks=None, allowCross=False),
], className="mb-2")

result_card = dbc.Card([
    dbc.CardHeader(step(3, "Result"), className="bg-light fw-bold"),
    dbc.CardBody([
        dbc.Alert(id="error-alert", is_open=False, color="danger", dismissable=True),
        html.Div(id="result-container", style={"display": "none"}, children=[
            html.Div([html.H5(id="res-title", className="text-primary mb-0"),
                      html.Div(id="res-date", className="text-muted small")],
                     className="text-center mb-2"),
            dbc.Row(id="res-stats", className="g-2 mb-3"),
            result_controls,
            dcc.Loading(dcc.Graph(id="result-graph",
                                  config={"scrollZoom": True, "displaylogo": False}),
                        type="dot"),
            html.Div(id="res-info", className="text-center small text-muted mt-1"),
            html.Div("Scroll to zoom, drag to zoom a box, double-click to reset. The three panels "
                     "move together. Hover any cell for its value. Observed pixels are kept as "
                     "they are; only the gaps take the model's value. The camera icon saves a PNG.",
                     className="text-center small text-muted"),
            html.Hr(),
            dbc.Button("⬇ Download NetCDF (filled SST + error estimate)", id="btn-download",
                       color="success", className="w-100"),
        ]),
        html.Div(id="placeholder-res", children=html.Div([
            html.Div("🌊", style={"fontSize": "3rem"}),
            html.Div("No reconstruction yet.", className="fw-bold"),
            html.Div("Choose a sector on the map, pick a scene above and press Run.",
                     className="text-muted small")], className="text-center mt-5 mb-5")),
    ])], className="rounded-box mb-3", id="results-card")

app.layout = html.Div([
    header,
    dcc.Store(id="sector-store"), dcc.Store(id="result-store"), dcc.Store(id="map-view"),
    dcc.Download(id="download-nc"),
    dbc.Container([dbc.Row([
        dbc.Col([map_card, model_card], lg=4),
        dbc.Col([scene_card, result_card], lg=8),
    ])], fluid=True, style={"paddingBottom": "40px"}),
])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback([Output("sector-store", "data"), Output("sector-map", "figure"),
           Output("selection-output", "children"), Output("sector-dd", "value"),
           Output("map-view", "data"), Output("map-sat-label", "children"),
           Output("map-legend", "children")],
          [Input("sector-map", "clickData"), Input("sector-dd", "value"),
           Input("sat-radio", "value"), Input("map-color", "value"), Input("map-basemap", "value"),
           Input("btn-zoom-sector", "n_clicks"), Input("btn-show-all", "n_clicks")],
          [State("sector-store", "data"), State("map-view", "data")])
def update_map(click, dd, sat, color_by, basemap, _zoom, _all, current, view):
    trig = dash.ctx.triggered_id
    sid, view = current, view or DEFAULT_VIEW
    if trig == "sector-map" and click:
        loc = str(click["points"][0].get("location", ""))
        if loc in C.SECTORS:
            sid = loc
    elif trig == "sector-dd" and dd in C.SECTORS and dd != current:
        sid, view = dd, fit_view(dd)                  # picked from the list: fly to it
    elif trig == "btn-zoom-sector" and sid:
        view = fit_view(sid)
    elif trig == "btn-show-all":
        view = dict(DEFAULT_VIEW, rev=uuid.uuid4().hex[:6])
    fig = build_map(sid, sat, color_by, basemap, view)
    text = (f"Selected: Sector {sid} ({C.SECTORS[sid]['label']}) · {C.SATELLITES[sat]}"
            if sid else "No Sector Selected")
    n_ready = sum(os.path.exists(model_path(sat, s)) for s in C.SECTORS)
    if color_by == "skill":
        legend = html.Span("Colour = test RMSE (Jul–Dec 2016), greener is better")
    else:
        legend = html.Span([
            html.Span(className="legend-dot", style={"background": STATUS_COLORS["trained"],
                                                     "marginLeft": 0}), "trained",
            html.Span(className="legend-dot", style={"background": STATUS_COLORS["untrained"]}),
            "not trained yet",
            html.Span(className="legend-dot", style={"background": STATUS_COLORS["selected"]}),
            "selected"])
    return (sid, fig, text, sid, view,
            f"{C.SATELLITES[sat]} · {n_ready}/{len(C.SECTORS)} models", legend)


@callback(Output("model-info", "children"),
          [Input("sector-store", "data"), Input("sat-radio", "value")])
def show_model_info(sid, sat):
    if not sid:
        return html.Span("Select a sector to see its model.", className="text-muted")
    m = load_metrics(sat, sid)
    if not m:
        return dbc.Alert(f"{C.SATELLITES[sat]} model for this sector is not trained yet.",
                         color="warning", className="p-2 mb-0")
    rmse, base, npass, kind = model_skill(m)
    a = sector_summary(sat, sid)
    period = "Jul – Dec 2016" if kind == "test" else "held-out days of the training period"
    return html.Div([
        html.Div(f"{C.SATELLITES[sat]} model · {C.SECTORS[sid]['label']}", className="fw-bold"),
        html.Div(f"Trained on {m['train_passes']:,} passes, {m['train_period'].split(' ')[0]}"
                 .replace("..", " → ")),
        html.Div(f"{kind.capitalize()} ({period}, {npass} passes):"),
        html.Div(f"RMSE {rmse:.2f} °C  vs  {base:.2f} °C for a plain 30-day composite "
                 f"(−{100 * (1 - rmse / base):.0f}%)", className="text-success fw-bold"),
        html.Div(f"Archive: {a['n']:,} passes, {a['t0']:%d %b %Y} – {a['t1']:%d %b %Y}")
        if a else None,
    ])


@callback([Output("archive-box", "style"), Output("upload-box", "style")],
          Input("source-radio", "value"))
def toggle_source(source):
    show, hide = {"display": "block"}, {"display": "none"}
    return (show, hide) if source == "archive" else (hide, show)


@callback([Output("date-dd", "options"), Output("date-dd", "value"),
           Output("date-dd", "placeholder")],
          [Input("sector-store", "data"), Input("sat-radio", "value")],
          State("date-dd", "value"))
def date_options(sid, sat, current):
    if not sid:
        return [], None, "Select a sector first"
    if not os.path.exists(archive_path(sat, sid)):
        return [], None, f"No {C.SATELLITES[sat]} archive for this sector yet"
    dates = list(scene_options(sat, sid))
    keep = current if current in dates else None
    return [{"label": d, "value": d} for d in dates], keep, "Select a date"


@callback([Output("time-dd", "options"), Output("time-dd", "value")],
          [Input("date-dd", "value"), Input("sector-store", "data"), Input("sat-radio", "value")])
def time_options(date, sid, sat):
    if not (date and sid):
        return [], None
    opts = scene_options(sat, sid).get(date, [])
    return [{"label": lab, "value": v} for v, lab in opts], None


@callback([Output("run-btn", "disabled"), Output("upload-status", "children")],
          [Input("upload-data", "contents"), Input("sector-store", "data"),
           Input("source-radio", "value"), Input("time-dd", "value")],
          State("upload-data", "filename"))
def toggle_button(contents, sid, source, scene, filename):
    ready = bool(sid) and bool(scene if source == "archive" else contents)
    if source == "archive":
        return not ready, ""
    return not ready, (f"📄 {filename}" if filename else "No file uploaded yet.")


@callback([Output("result-container", "style"), Output("placeholder-res", "style"),
           Output("result-store", "data"), Output("res-title", "children"),
           Output("res-date", "children"), Output("res-info", "children"),
           Output("res-stats", "children"), Output("loading-output", "children"),
           Output("error-alert", "children"), Output("error-alert", "is_open")],
          Input("run-btn", "n_clicks"),
          [State("upload-data", "contents"), State("upload-data", "filename"),
           State("sector-store", "data"), State("sat-radio", "value"),
           State("source-radio", "value"), State("time-dd", "value")],
          prevent_initial_call=True)
def run(n, contents, filename, sid, sat, source, scene):
    if not sid or not (scene if source == "archive" else contents):
        raise PreventUpdate
    fail = lambda msg: ({"display": "none"}, {"display": "block"}, None, "", "", "", [], "",
                        msg, True)
    try:
        warn = ""
        if source == "archive":
            res = run_archive_scene(scene, sat, sid)
        else:
            data = base64.b64decode(contents.split(",", 1)[1])
            res = run_inference(data, filename, sat, sid)
            file_sat = C.parse_satellite(filename)
            if file_sat and file_sat != sat:
                warn = f"  ·  ⚠ file name says {C.SATELLITES[file_sat]}"
    except ValueError as e:
        return fail(str(e))
    except Exception as e:
        import traceback; traceback.print_exc()
        return fail(f"An unexpected error occurred: {e}")

    st = result_stats(res)
    c0, c1 = res["ctx_span"]
    info = (f"Context: {res['n_ctx']} {C.SATELLITES[sat]} passes on {res['ctx_days']} days, "
            f"{c0:%d-%b-%Y} → {c1:%d-%b-%Y}{warn}")
    tiles = [
        stat("Observed", f"{st['observed']:.0f}%", "of the sector's sea cells"),
        stat("Filled by the model", f"{st['n_gap']:,} cells", f"≈ {st['area']:,.0f} km²"),
        stat("Mean uncertainty", f"±{st['sig_mean']:.2f} °C" if st["sig_mean"] is not None else "—",
             f"max ±{st['sig_max']:.2f} °C in the gaps" if st["sig_max"] is not None else "no gaps"),
        stat("Context", f"{res['n_ctx']} passes", f"{res['ctx_days']} days before the scene"),
        stat("SST", f"{st['p2']:.1f} – {st['p98']:.1f} °C", "2nd – 98th percentile"),
    ]
    return ({"display": "block"}, {"display": "none"}, res["uid"],
            f"{C.SATELLITES[sat]} · Sector {sid}: {C.SECTORS[sid]['label']}",
            f"Date: {res['ts']:%d-%b-%Y %H:%M} UTC", info, tiles, "", "", False)


@callback([Output("result-graph", "figure"), Output("result-graph", "config"),
           Output("res-range", "min"), Output("res-range", "max"), Output("res-range", "value"),
           Output("res-range", "marks")],
          [Input("result-store", "data"), Input("res-view", "value"), Input("res-cmap", "value"),
           Input("res-range", "value"), Input("res-outline", "value")],
          [State("res-range", "min"), State("res-range", "max")])
def render_result(uid, view, cmap, zrange, outline, smin, smax):
    res = RESULTS.get(uid)
    if res is None:
        raise PreventUpdate
    if dash.ctx.triggered_id in (None, "result-store"):   # a new result: reset the colour range
        zrange, smin, smax = default_range(res)
    marks = {float(v): f"{v:g}"
             for v in np.arange(np.ceil(smin), smax + 0.01, 2 if smax - smin > 5 else 1)}
    fig = result_figure(res, view, cmap, zrange, outline)
    config = {"scrollZoom": True, "displaylogo": False,
              "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
              "toImageButtonOptions": {"format": "png", "scale": 2,
                                       "filename": f"{res['sat']}_{C.SECTORS[res['sid']]['name']}_"
                                                   f"{res['ts']:%Y%m%d-%H%M}_gapfill"}}
    return fig, config, smin, smax, zrange, marks


clientside_callback(
    """function(uid) {
        if (uid) { setTimeout(() => { const e = document.getElementById('results-card');
                   if (e && window.innerWidth < 992) e.scrollIntoView({behavior: 'smooth'}); }, 300); }
        return window.dash_clientside.no_update;
    }""",
    Output("results-card", "className"), Input("result-store", "data"), prevent_initial_call=True)


# Fit the result figure's height to its real width, so the panels fill it at any screen
# size (same formula as fit_height); refits when the window is resized.
clientside_callback(
    """function(fig) {
        const want = fig && fig.layout && fig.layout.meta && fig.layout.meta.fit;
        if (!want) { return window.dash_clientside.no_update; }
        if (!window.fitResultHeight) {
            window.fitResultHeight = function (key, tries) {
                const gd = document.querySelector('#result-graph .js-plotly-plot');
                const m = gd && gd.layout && gd.layout.meta && gd.layout.meta.fit;
                if (!m || (key && m.key !== key) || !gd.clientWidth) {
                    if (tries > 0) { setTimeout(() => window.fitResultHeight(key, tries - 1), 100); }
                    return;
                }
                const w = gd.clientWidth - m.l - m.r;
                let h = m.stacked ? m.n * w * m.aspect / (1 - m.vs * (m.n - 1))
                                  : w * (1 - m.hs * (m.n - 1)) / m.n * m.aspect;
                h = Math.max(300, Math.min(1000, Math.round(h + m.t + m.b)));
                if (Math.abs((gd.layout.height || 0) - h) > 3) { window.Plotly.relayout(gd, {height: h}); }
            };
            let timer;
            window.addEventListener('resize', () => {
                clearTimeout(timer); timer = setTimeout(() => window.fitResultHeight(null, 0), 150);
            });
        }
        setTimeout(() => window.fitResultHeight(want.key, 40), 30);
        return window.dash_clientside.no_update;
    }""",
    Output("result-graph", "className"), Input("result-graph", "figure"), prevent_initial_call=True)


@callback(Output("download-nc", "data"), Input("btn-download", "n_clicks"),
          State("result-store", "data"), prevent_initial_call=True)
def download(n, uid):
    res = RESULTS.get(uid)
    path = res["dl"] if res else None
    return dcc.send_file(path) if path and os.path.exists(path) else dash.no_update


if __name__ == "__main__":
    port = int(os.environ.get("NOAA_APP_PORT", "8052"))
    # Warm up once so the first Run is not slowed by cold imports / model loads.
    import rasterio.io, scipy.ndimage, rasterio.features  # noqa: F401
    for _sid in C.SECTORS:
        for _sat in C.SATELLITES:
            if os.path.exists(model_path(_sat, _sid)):
                load_model(_sat, _sid); load_archive(_sat, _sid)
        if os.path.exists(archive_path("n18", _sid)):
            _t, _s, _lat, _lon, _sea = load_archive("n18", _sid)
            coast_layers(_sid, _lon, _lat, _sea)
    print(f"device: {DEVICE}.  Serving on http://localhost:{port}", flush=True)
    if os.environ.get("NOAA_APP_OPEN_BROWSER"):        # set by run_noaa_app.bat / .sh
        import webbrowser
        threading.Timer(1.5, webbrowser.open, args=[f"http://localhost:{port}"]).start()
    try:
        from waitress import serve
        try:
            serve(app.server, listen=f"127.0.0.1:{port} [::1]:{port}", threads=8)
        except OSError:
            serve(app.server, listen=f"127.0.0.1:{port}", threads=8)
    except ImportError:
        app.run(host="127.0.0.1", port=port)
