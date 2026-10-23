"""Lever "cycle": 1-minute bars (a bot that checks every minute), stored as float32.

The raw 1-minute candles are float32, so float32 storage is exact; ckernel/ckernel2 promote to float64 inside and
give bit-identical results on float32 or float64 inputs (checked on 15m bars in the session log). main() checks that
these 1-minute bars aggregate exactly to the 5-minute bars of x10_cycle_lib.build_bars.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_cycle_lib as L  # noqa: E402


def path(dead):
    return os.path.join(L.OUT, f"bars_1m{'_dead' if dead else ''}_f32.npz")


def build(dead=False, start="2021-01-01"):
    if os.path.exists(path(dead)):
        return
    syms = list(E.H19) + (list(E.DEAD) if dead else [])
    firsts, lasts = [], []
    for s in syms:
        d = L._coin1m(s, start)
        firsts.append(d.index[0]); lasts.append(d.index[-1])
        del d
    idx = pd.date_range(min(firsts).floor("1h"), max(lasts).floor("1h") + pd.Timedelta(minutes=59), freq="1min")
    out = {f: np.full((len(idx), len(syms)), np.nan, np.float32) for f in ("o", "h", "l", "c")}
    for j, s in enumerate(syms):
        d = L._coin1m(s, start).reindex(idx)
        first = d.c.first_valid_index()
        cc = d.c.ffill()
        live = idx >= first
        for f in ("o", "h", "l"):
            col = d[f].where(d[f].notna(), cc)
            out[f][:, j] = np.where(live, col.values, np.nan)
        out["c"][:, j] = np.where(live, cc.values, np.nan)
        if dead and s in E.DEAD:
            gone = idx > pd.Timestamp(E.DEAD[s]) + pd.Timedelta(minutes=59)
            for f in out:
                out[f][gone, j] = np.nan
        del d, cc
    np.savez(path(dead), syms=np.array(syms), t=idx.values.astype("datetime64[s]").astype(np.int64), **out)


def load(dead=False):
    build(dead)
    z = np.load(path(dead), allow_pickle=False)
    return {"syms": [str(x) for x in z["syms"]], "t": pd.to_datetime(z["t"], unit="s"), "C": 1,
            **{f: z[f] for f in ("o", "h", "l", "c")}}


def ref32(B, W):
    return pd.DataFrame(B["c"]).rolling(W, min_periods=1).max().shift(1).values.astype(np.float32)


def cf32(B):
    return pd.DataFrame(B["c"]).ffill().values.astype(np.float32)


def main():
    ok = True
    for dead in (False, True):
        B = load(dead); b5 = L.load_bars(5, dead)
        nb = B["o"].shape[0] // 5
        agg = {"o": B["o"][::5], "c": B["c"][4::5], "h": np.fmax.reduce(B["h"].reshape(nb, 5, -1), axis=1),
               "l": np.fmin.reduce(B["l"].reshape(nb, 5, -1), axis=1)}
        same = all(np.array_equal(agg[f].astype(np.float64), b5[f], equal_nan=True) for f in agg)
        ok &= same
        print(f"1m{' +dead' if dead else ''} -> 5m: {'IDENTICAL' if same else 'DIFFERENT'}")
        del B, b5, agg
    print("ALL CHECKS PASS" if ok else "SOME CHECK FAILED")


if __name__ == "__main__":
    main()
