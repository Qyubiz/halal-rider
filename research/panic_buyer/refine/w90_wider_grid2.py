"""Task "ten-wider", grid 2 (HINDSIGHT: placed after grid 1, around its closest misses
'v30 top30 @22% x0.5 N1' = 12.4/9.9/8.6 %/month and 'v365 top30 @20% x1 N1' = 13.0/9.1/8.7).

Core = 10% mode (14%, +5%, 48h, 24h cooldown, guard 35%, N1, fmax 100%); tier take-profit +5%.
GRID (192): rk {v30, v365} x M {20, 25, 30, 40} x dt {20, 22, 24, 26%} x st {0.5, 0.75, 1.0}
  x pump filter on tier coins {off, skip when last close > 2x its 30-day low}.
Stressed version run only when both 2021-23 and 2024-26 make >= 8.5%/month (others cannot qualify).
Selection rule as grid 1.
"""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_grid as G  # noqa: E402
import w90_wider_lib as L  # noqa: E402


def designs():
    return [dict(L.TENMODE, rk=rk, M=M, dt=dt, st=st, tt=0.05, pg=pg)
            for rk, M, dt, st, pg in itertools.product(("v30", "v365"), (20, 25, 30, 40), (0.20, 0.22, 0.24, 0.26),
                                                       (0.5, 0.75, 1.0), (0.0, 2.0))]


def run_grid(ds, path, W=None, gate=0.085):
    W = L.build() if W is None else W
    rows = []
    t0 = time.time()
    for i, d in enumerate(ds):
        r, *_ = L.evaluate(d, W=W)
        row = dict(name=L.name(d), **{f"p_{k}": v for k, v in d.items()}, **r)
        if min(r["d_mo"], r["t_mo"]) >= gate:
            rs, *_ = L.evaluate(d, stress=True, W=W)
            row.update({f"S{k}": v for k, v in rs.items()})
        rows.append(row)
        if i % 16 == 15 or i == len(ds) - 1:
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{i + 1}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    path = os.path.join(L.OUT, "grid2.csv")
    df = run_grid(designs(), path)
    pd.set_option("display.width", 250)
    print("\nTOP by worst-of-three %/month (normal):")
    print(G.show(df.fillna(float("nan")), n=30))
    q = df[df.ten]
    print(f"\n10% qualifiers (both 2021-23 and 2024-26 >= 10%): {len(q)}")
    if len(q):
        print(G.show(q, n=40))
