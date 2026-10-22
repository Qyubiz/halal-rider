"""Task "ten-wider", grid 3 (HINDSIGHT: placed after grid 2, whose best design sat on the grid edge,
'v365 top30 @26% x1' = 15.1/11.0/11.4 %/month).

Core = 10% mode (14%, +5%, 48h, 24h cooldown, guard 35%, N1, fmax 100%).
GRID (78):
  v365: M {25, 30, 35, 40} x dt {24, 26, 28, 30, 32%} x st {0.75, 1.0} x tt {+5%}          = 40
        plus M 30 x dt {26, 28, 30%} x st 1.0 x tt {+6%, +8%}                                = 6
  v30:  M {15, 20, 25} x dt {20, 22, 24, 26%} x st {0.75, 1.0} x tt +5%                         = 24
  v365 M30 dt 26/28 st 1 with core depth 13% / 15%                                              = 4  (core check)
  v365 M30 dt 26/28 st 1 with cooldown 12h / 36h                                                = 4
Stressed version only when both 2021-23 and 2024-26 make >= 8.5%/month. Selection rule as grid 1.
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

T = L.TENMODE


def designs():
    ds = [dict(T, rk="v365", M=M, dt=dt, st=st, tt=0.05)
          for M, dt, st in itertools.product((25, 30, 35, 40), (0.24, 0.26, 0.28, 0.30, 0.32), (0.75, 1.0))]
    ds += [dict(T, rk="v365", M=30, dt=dt, st=1.0, tt=tt) for dt, tt in itertools.product((0.26, 0.28, 0.30), (0.06, 0.08))]
    ds += [dict(T, rk="v30", M=M, dt=dt, st=st, tt=0.05)
           for M, dt, st in itertools.product((15, 20, 25), (0.20, 0.22, 0.24, 0.26), (0.75, 1.0))]
    ds += [dict(T, rk="v365", M=30, dt=dt, st=1.0, tt=0.05, dc=dc) for dt, dc in itertools.product((0.26, 0.28), (0.13, 0.15))]
    ds += [dict(T, rk="v365", M=30, dt=dt, st=1.0, tt=0.05, cd=cd) for dt, cd in itertools.product((0.26, 0.28), (12, 36))]
    return ds


if __name__ == "__main__":
    path = os.path.join(L.OUT, "grid3.csv")
    df = G2.run_grid(designs(), path)
    pd.set_option("display.width", 250)
    print("\nTOP by worst-of-three %/month (normal):")
    print(G.show(df, n=40))
    q = df[df.ten]
    print(f"\n10% qualifiers (both 2021-23 and 2024-26 >= 10%): {len(q)} of {len(df)}")
