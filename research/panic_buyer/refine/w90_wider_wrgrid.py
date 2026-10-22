"""Task "ten-wider", win-rate frontier: does the wider universe help at the high-win-rate end (small take-profits)?
Fixed before running. 48h time stop, 24h cooldown everywhere.

GRID (54 + 2 references): sizing {N8 x 20% (live), N3 x 33%, N1 x 100%} x take-profit {+1.5, +2, +3%}
  x core depth {12, 14%} x tier {none, v365 top30 at core+6 pts x1, v30 top20 at core+8 pts x0.75}; crash guard 35%.
References: live settings exactly (12%, N8 x 20%, +1.5%, no guard) on the 19 coins, and the same + v365 tier.
Qualifier for the 90% goal: win rate >= 90% in 2021-23 AND 2024-26 (trades bought in the period).
"""
import itertools
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_grid as G  # noqa: E402
import w90_wider_grid2 as G2  # noqa: E402
import w90_wider_lib as L  # noqa: E402


def designs():
    base = dict(hm=48, cd=24, g=0.35)
    ds = [dict(base, dc=0.12, tp=0.015, N=8, fmax=0.2, g=0.0, M=0),
          dict(base, dc=0.12, tp=0.015, N=8, fmax=0.2, g=0.0, rk="v365", M=30, dt=0.18, st=1.0)]
    for (N, f), tp, dc, tier in itertools.product(((8, 0.2), (3, 1 / 3), (1, 1.0)), (0.015, 0.02, 0.03), (0.12, 0.14),
                                                  ("none", "v365", "v30")):
        d = dict(base, dc=dc, tp=tp, N=N, fmax=f)
        if tier == "none":
            d["M"] = 0
        elif tier == "v365":
            d.update(rk="v365", M=30, dt=dc + 0.06, st=1.0)
        else:
            d.update(rk="v30", M=20, dt=dc + 0.08, st=0.75)
        ds.append(d)
    return ds


if __name__ == "__main__":
    path = os.path.join(L.OUT, "wrgrid.csv")
    df = G2.run_grid(designs(), path, gate=-1.0)
    pd.set_option("display.width", 250)
    print(G.show(df, n=60, key="min_win"))
    print(f"\n90% win-rate qualifiers (both periods): {int(df.wr90.sum())} of {len(df)}")
