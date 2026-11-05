"""Task "ten-wider", grid 1 (fixed before any wider-universe result of this round was seen; 2021-23 and 2024-26 are
both visible to the selection rule, so everything here counts as tuned on 2024-26).

Core = 10% mode on the user's 19 coins: depth 14%, take-profit +5%, 48h time stop, 24h cooldown, crash guard 35%
(7-day high), hourly resting bids, fmax 100%.
Tier = other screened coins in the monthly top M of the non-19 coins by trailing volume.

GRID (96 designs + 2 controls, each run normal and stressed):
  ranking rk {v30, v365} x M {30, 50} x tier depth dt {18, 20, 22%} x tier size st {0.5, 1.0}
  x tier take-profit tt {+5%, +8%} x bids N {1, 2}
  controls: the 19 coins alone (M=0) with N 1 and 2.

SELECTION RULE (this round, fixed): 10%-goal qualifier = %/month >= 10% in 2021-23 AND 2024-26 (normal run);
among qualifiers the highest worst-of (2021-23, 2024-26, since 2025) %/month, ties -> shallower worst drop.
"""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_lib as L  # noqa: E402


def designs():
    out = [dict(L.TENMODE, M=0, N=1), dict(L.TENMODE, M=0, N=2)]
    for rk, M, dt, st, tt, N in itertools.product(("v30", "v365"), (30, 50), (0.18, 0.20, 0.22), (0.5, 1.0),
                                                  (0.05, 0.08), (1, 2)):
        out.append(dict(L.TENMODE, rk=rk, M=M, dt=dt, st=st, tt=tt, N=N))
    return out


def run_grid(ds, path, W=None):
    W = L.build() if W is None else W
    rows = []
    t0 = time.time()
    for i, d in enumerate(ds):
        r, *_ = L.evaluate(d, W=W)
        rs, *_ = L.evaluate(d, stress=True, W=W)
        row = dict(name=L.name(d), **{f"p_{k}": v for k, v in d.items()}, **r,
                   **{f"S{k}": v for k, v in rs.items()})
        rows.append(row)
        if i % 8 == 7 or i == len(ds) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{i + 1}/{len(ds)} {time.time() - t0:.0f}s | {row['name']} | {L.fmt(r)}", flush=True)
    return pd.DataFrame(rows)


def show(df, n=25, key="min_mo"):
    cols = ["name", "d_win", "d_mo", "d_dd", "t_win", "t_mo", "t_dd", "s_win", "s_mo", "s_dd", "Sd_mo", "St_mo",
            "Ss_mo", "Smin_dd", "t_pnl_out", "t_pnl"]
    cols = [c for c in cols if c in df.columns]
    x = df.sort_values(key, ascending=False).head(n)[cols].copy()
    for c in cols[1:]:
        x[c] = (x[c] * 100).round(1)
    return x.to_string(index=False)


if __name__ == "__main__":
    path = os.path.join(L.OUT, "grid1.csv")
    df = run_grid(designs(), path)
    pd.set_option("display.width", 250)
    print("\nTOP by worst-of-three %/month (normal):")
    print(show(df))
    q = df[df.ten]
    print(f"\n10% qualifiers (both 2021-23 and 2024-26 >= 10%): {len(q)}")
    if len(q):
        print(show(q))
