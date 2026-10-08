"""
Train one gap-filling model per satellite and sector on the GPU.

    python gapfill/train.py                              # every sector, both satellites
    python gapfill/train.py --sat n18 --sector 1,2,4     # a list of sectors
    python gapfill/train.py --skip-done                  # resume: skip finished models

Split (by observation time):
    train  2014-01-01 .. 2016-06-30, except every 10th day
    val    every 10th day of that span   (model selection)
    test   2016-07-01 .. 2016-12-31      (reported only, never used to train)

Each target pass is given the 30 days of passes before it as context (see
common.build_features). Extra gaps are cut into the target using the real
cloud mask of another pass; the loss is the Gaussian negative log-likelihood
on the pixels that were hidden (plus a small weight on the visible ones).
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd
import xarray as xr
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

CACHE_DIR = C.CACHE_DIR
MIN_COV_TRAIN = 0.05
MIN_COV_TEST = 0.10


def load_archive(sat, sid):
    path = os.path.join(C.ARCHIVE_DIR, sat, f"{C.SECTORS[sid]['name']}.nc")
    with xr.open_dataset(path) as ds:
        return (pd.DatetimeIndex(ds.time.values), ds.SST.values.astype("f4"),
                ds.lat.values, ds.lon.values, ds.mask.values == 1)


def context_features(times, sst, lat, lon, stats, targets, cache_path):
    """(N, C-2, H, W) float16 memmap of the context channels of every target."""
    N, (H, W) = len(targets), sst.shape[1:]
    shape = (N, C.n_channels() - 2, H, W)
    meta = cache_path + ".json"
    if os.path.exists(cache_path) and os.path.exists(meta):
        m = json.load(open(meta))
        if m.get("shape") == list(shape) and m.get("stats") == stats:
            return np.lib.format.open_memmap(cache_path, mode="r")
    mm = np.lib.format.open_memmap(cache_path, mode="w+", dtype="f2", shape=shape)
    t0 = time.time()
    for k, i in enumerate(targets):
        lo = times.searchsorted(times[i] - pd.Timedelta(days=C.CONTEXT_DAYS))
        f = C.build_features(sst[i], times[i], sst[lo:i], times[lo:i], stats, lat, lon)
        mm[k] = f[2:]
        if k % 500 == 0:
            print(f"    features {k}/{N}  {time.time()-t0:.0f}s", flush=True)
    mm.flush()
    json.dump({"shape": list(shape), "stats": stats}, open(meta, "w"))
    return np.lib.format.open_memmap(cache_path, mode="r")


def make_batch(idx, ctx, tgt, tgt_ok, donors_ok, sea, rng, fixed=None):
    """Returns inputs, target, hidden mask, visible mask (torch, on CPU)."""
    B = len(idx)
    H, W = tgt.shape[1:]
    x = np.empty((B, C.n_channels(), H, W), "f4")
    hid = np.zeros((B, H, W), bool)
    vis = np.zeros((B, H, W), bool)
    for b, i in enumerate(idx):
        ok = tgt_ok[i]
        for _ in range(10):
            j = fixed[i] if fixed is not None else rng.integers(len(donors_ok))
            v = ok & donors_ok[j]
            h = ok & ~donors_ok[j]
            if h.sum() >= 0.05 * ok.sum() and v.sum() >= 0.02 * sea.sum():
                break
            if fixed is not None:
                break
        vis[b], hid[b] = v, h
        x[b, 0] = np.where(v, tgt[i], 0)
        x[b, 1] = v
        x[b, 2:] = ctx[i]
    return (torch.from_numpy(x), torch.from_numpy(np.nan_to_num(tgt[idx])),
            torch.from_numpy(hid), torch.from_numpy(vis))


def nll(mean, logvar, y, w):
    l = 0.5 * (logvar + (y - mean) ** 2 * torch.exp(-logvar))
    return (l * w).sum() / w.sum().clamp(min=1)


def evaluate(model, sel, ctx, tgt, tgt_ok, donors_ok, sea, stats, fixed, dev, bs=16):
    """RMSE (deg C) on the hidden pixels, model vs. 30-day composite baseline."""
    model.eval()
    se_m = se_b = ae_m = bias = n = 0.0
    with torch.no_grad():
        for s in range(0, len(sel), bs):
            idx = sel[s:s + bs]
            x, y, h, v = make_batch(idx, ctx, tgt, tgt_ok, donors_ok, sea, None, fixed)
            mean, _ = model(x.to(dev))
            mean = mean.cpu().numpy() * stats["std"]
            y = y.numpy() * stats["std"]
            h = h.numpy()
            # baseline: 30-day same day/night composite, else 30-day all, else mean
            # channels 14/15 = 30-day mean/frac, 16/17 = same-phase mean/frac
            xn = x.numpy()
            base = np.where(xn[:, 17] > 0, xn[:, 16],
                            np.where(xn[:, 15] > 0, xn[:, 14], 0)) * stats["std"]
            d = (mean - y)[h]
            se_m += (d ** 2).sum(); ae_m += np.abs(d).sum(); bias += d.sum()
            se_b += ((base - y)[h] ** 2).sum(); n += h.sum()
    model.train()
    n = max(n, 1)
    return dict(rmse=float(np.sqrt(se_m / n)), mae=float(ae_m / n), bias=float(bias / n),
                rmse_30day_composite=float(np.sqrt(se_b / n)), n_pixels=int(n),
                n_passes=int(len(sel)))


def train_one(sat, sid, epochs, bs, dev, max_minutes, amp=False):
    name = C.SECTORS[sid]["name"]
    print(f"\n=== {sat} {name} ===", flush=True)
    times, sst, lat, lon, sea = load_archive(sat, sid)
    ok = np.isfinite(sst) & sea
    cov = ok.reshape(len(sst), -1).sum(1) / sea.sum()

    is_train_period = times < C.TRAIN_END
    day_no = (times.normalize() - pd.Timestamp("2014-01-01")).days
    is_val = is_train_period & (day_no % 10 == 0)
    tr = np.where(is_train_period & ~is_val & (cov >= MIN_COV_TRAIN))[0]
    va = np.where(is_val & (cov >= MIN_COV_TRAIN))[0]
    te = np.where((times >= C.TEST_START) & (cov >= MIN_COV_TEST))[0]
    print(f"passes: train={len(tr)} val={len(va)} test={len(te)}  grid={sst.shape[1:]}", flush=True)

    vals = sst[is_train_period][ok[is_train_period]]
    stats = {"mean": float(vals.mean()), "std": float(vals.std())}

    targets = np.concatenate([tr, va, te])
    os.makedirs(CACHE_DIR, exist_ok=True)
    ctx_all = context_features(times, sst, lat, lon, stats, targets,
                               os.path.join(CACHE_DIR, f"{sat}_{name}.npy"))
    pos = {t: k for k, t in enumerate(targets)}
    # everything indexed by position in `targets`
    tgt = np.where(ok[targets], (sst[targets] - stats["mean"]) / stats["std"], np.nan).astype("f4")
    tgt_ok = ok[targets]
    donors_ok = ok[tr[cov[tr] < 0.95]] | ~sea           # real cloud masks from training passes
    P_tr = np.array([pos[t] for t in tr], dtype=int)
    P_va = np.array([pos[t] for t in va], dtype=int)
    P_te = np.array([pos[t] for t in te], dtype=int)
    rng_fix = np.random.default_rng(0)
    fixed = rng_fix.integers(len(donors_ok), size=len(targets))

    ctx = ctx_all                      # disk-backed; the OS page cache keeps it warm

    torch.manual_seed(0)
    model = C.make_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(P_tr) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    rng = np.random.default_rng(1)
    # mixed precision on tensor-core GPUs (T4, V100, A100...): same model, ~2x faster
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    out_dir = os.path.join(C.MODEL_DIR, sat)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}.pt")
    best, hist, t0 = 1e9, [], time.time()
    for ep in range(epochs):
        perm = rng.permutation(P_tr)
        tot = 0.0
        for s in range(0, len(perm), bs):
            x, y, h, v = make_batch(perm[s:s + bs], ctx, tgt, tgt_ok, donors_ok, sea, rng)
            x, y, h, v = x.to(dev), y.to(dev), h.to(dev), v.to(dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                mean, logvar = model(x)
            loss = nll(mean.float(), logvar.float(), y, h.float() + 0.2 * v.float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(x)
        vm = evaluate(model, P_va, ctx, tgt, tgt_ok, donors_ok, sea, stats, fixed, dev)
        el = (time.time() - t0) / 60
        hist.append({"epoch": ep + 1, "train_nll": tot / len(perm), **{f"val_{k}": v for k, v in vm.items()}})
        flag = ""
        if vm["rmse"] < best:
            best = vm["rmse"]; flag = "  *saved"
            torch.save({"state_dict": model.state_dict(), "stats": stats, "sat": sat,
                        "sector": sid, "epoch": ep + 1, "n_channels": C.n_channels()}, out_path)
        print(f"  ep {ep+1:3d}/{epochs} nll={tot/len(perm):.3f} val_rmse={vm['rmse']:.3f} "
              f"(30d-composite {vm['rmse_30day_composite']:.3f})  {el:.1f} min{flag}", flush=True)
        if el > max_minutes:
            print(f"  time budget of {max_minutes} min reached", flush=True)
            break

    ck = torch.load(out_path, map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    val = evaluate(model, P_va, ctx, tgt, tgt_ok, donors_ok, sea, stats, fixed, dev)
    test = evaluate(model, P_te, ctx, tgt, tgt_ok, donors_ok, sea, stats, fixed, dev)
    metrics = {"satellite": C.SATELLITES[sat], "sector": name, "best_epoch": ck["epoch"],
               "train_passes": int(len(tr)), "stats": stats,
               "train_period": f"{times[0]:%Y-%m-%d}..{C.TRAIN_END - pd.Timedelta(days=1):%Y-%m-%d}"
                               f" (every 10th day held out)",
               "test_period": f"{C.TEST_START:%Y-%m-%d}..{times[-1]:%Y-%m-%d}",
               "validation": val, "test": test, "history": hist,
               "minutes": round((time.time() - t0) / 60, 1)}
    json.dump(metrics, open(out_path.replace(".pt", "_metrics.json"), "w"), indent=1)
    print(f"  VAL  rmse={val['rmse']:.3f} C  (30-day composite {val['rmse_30day_composite']:.3f})")
    print(f"  TEST rmse={test['rmse']:.3f} C  (30-day composite {test['rmse_30day_composite']:.3f})"
          f"  bias={test['bias']:+.3f}  passes={test['n_passes']}", flush=True)
    del ctx
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", choices=list(C.SATELLITES), default=None)
    ap.add_argument("--sector", default=None, help="sector id, or a comma-separated list")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-minutes", type=float, default=25)
    ap.add_argument("--skip-done", action="store_true",
                    help="skip models that already have a _metrics.json (resume after a disconnect)")
    ap.add_argument("--no-amp", action="store_true", help="disable mixed precision")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (dev == "cuda" and not a.no_amp and torch.cuda.get_device_capability()[0] >= 7)
    print(f"device: {dev} {torch.cuda.get_device_name(0) if dev == 'cuda' else ''}"
          f"  mixed precision: {amp}", flush=True)
    sats = [a.sat] if a.sat else list(C.SATELLITES)
    secs = a.sector.split(",") if a.sector else list(C.SECTORS)
    summary = []
    for sat in sats:
        for sid in secs:
            name = C.SECTORS[sid]["name"]
            done = os.path.join(C.MODEL_DIR, sat, f"{name}_metrics.json")
            if a.skip_done and os.path.exists(done):
                print(f"\n=== {sat} {name}: already trained, skipping ===", flush=True)
                m = json.load(open(done))
            elif not os.path.exists(os.path.join(C.ARCHIVE_DIR, sat, f"{name}.nc")):
                print(f"\n=== {sat} {name}: no archive, run prepare_data.py first ===", flush=True)
                continue
            else:
                m = train_one(sat, sid, a.epochs, a.batch_size, dev, a.max_minutes, amp)
            summary.append({k: m[k] for k in ("satellite", "sector", "best_epoch", "minutes")} |
                           {"val_rmse": m["validation"]["rmse"], "test_rmse": m["test"]["rmse"],
                            "test_rmse_30day_composite": m["test"]["rmse_30day_composite"],
                            "test_bias": m["test"]["bias"], "test_passes": m["test"]["n_passes"]})
    print("\n" + pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
