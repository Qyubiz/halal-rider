"""Lens "crowding": when, inside each hour, each coin's resting 12% buy would have filled (from 1-minute candles).

The hourly engine only knows THAT a limit buy filled in an hour (the hourly low traded eps through it), not WHEN.
A cap on fills per hour ("cancel the other resting buys once F have filled") needs the order of the fills, so this
module computes, for every hour t and coin k, the minute of the first 1-minute low at or below lim*(1-eps), with
lim = ref[t,k]*(1-depth) (ref = 24h max close shifted 1, exactly the hourly engine's trigger):
  -1   the hour's open is already at/below lim (market buy at the hour open, the very first fill)
  0-59 minute of the first 1-minute low that trades eps through lim
  60   the hourly candle says filled but the 1-minute data shows no crossing (data mismatch: ranked last)
  nan  no fill this hour (or no data)
This is portfolio-independent (it only depends on ref and depth), uses only the hour's own minutes to order fills
that the hourly engine already counts as filled in that hour, and adds no information about later hours.
"""
import os

import numpy as np
import pandas as pd

import engine as E
import study as S


def _read1m(s):
    p = os.path.join(E.DATA, "1m", f"{s}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({"l": z["l"].astype(np.float64)}, index=pd.to_datetime(z["t"], unit="s"))


def fill_minutes(syms, idx, P, depth=0.12, eps=E.EPS, tag=""):
    cache = os.path.join(S.OUT, f"fm_{tag}_{len(syms)}_{depth:.3f}.npy")
    if os.path.exists(cache):
        FM = np.load(cache)
        if FM.shape == P["o"].shape:
            return FM
    O, Lh, REF = P["o"].values, P["l"].values, P["ref"].values
    lim = REF * (1.0 - depth)
    hourly_fill = (O <= lim) | (Lh <= lim * (1.0 - eps))
    n, K = O.shape
    FM = np.full((n, K), np.nan)
    t0 = idx[0].value // 10**9
    for k, s in enumerate(syms):
        d = _read1m(s)
        if s == "RENDERUSDT":
            r = _read1m("RNDRUSDT")
            if r is not None:
                d = pd.concat([r, d])
                d = d[~d.index.duplicated(keep="last")].sort_index()
        sec = d.index.values.astype("datetime64[s]").astype(np.int64)
        hpos = (sec - t0) // 3600
        mnt = (sec % 3600) // 60
        ok = (hpos >= 0) & (hpos < n)
        hpos, mnt, lo = hpos[ok], mnt[ok], d["l"].values[ok]
        cross = lo <= lim[hpos, k] * (1.0 - eps)
        hc, mc = hpos[cross], mnt[cross]
        first = np.full(n, np.nan)
        if len(hc):
            u, ii = np.unique(hc, return_index=True)        # minutes are time-sorted, so the first index is the earliest
            first[u] = mc[ii]
        col = np.where(hourly_fill[:, k], np.where(np.isnan(first), 60.0, first), np.nan)
        col = np.where(hourly_fill[:, k] & (O[:, k] <= lim[:, k]), -1.0, col)
        FM[:, k] = col
    np.save(cache, FM)
    return FM


if __name__ == "__main__":
    syms, idx, P = S.market(False)
    FM = fill_minutes(syms, idx, P, tag="h19")
    v = FM[~np.isnan(FM)]
    print(f"hour-coin fills: {len(v)} | at open {np.mean(v == -1) * 100:.1f}% | intra-hour {np.mean((v >= 0) & (v < 60)) * 100:.1f}% "
          f"| no 1m crossing (mismatch) {np.mean(v == 60) * 100:.2f}%")
