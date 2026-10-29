"""Grid 2 (written after grid 1, before running this): per-coin depth from each coin's own past volatility
(30-day hourly vol, known at the hour's open), crash guard 35%/168h, N=1 x 100%. Motivation (anatomy.log): since
2025 the account holds a coin only 9% of the hours and 48 of 56 trades are in the wildest third of the coins;
the calm coins almost never dip 14%.
Part A, tiers by volatility rank among the 19 (calm third / middle / wild third), 4 bases from grid 1
  (d14 tp5 hm48 cd24w1, d15 tp6 hm48 cd24w1, d15 tp4 hm96 cd24w1, d14 tp5 hm48 cd24 = 10% mode)
  x calm depth {+0, -2, -4, -6 pts} x wild depth {-1, +0, +1, +2 pts} x calm take-profit {base, base -2 pts} = 128.
Part B, depth = vm x own daily vol, clipped: vm {2.2, 2.6, 3.0, 3.4} x floor {6, 8, 10%} x cap {15, 17%}
  x take-profit {4%, 5%, 0.3 x depth, 0.4 x depth} x time stop {48, 96h}, cooldown 24h / re-arm after a win = 192.
Same fixed selection rule as grid 1 (HINDSIGHT: it looks at 2024-26)."""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402
import w90_tenh_report as R  # noqa: E402

BASES = {"d14 tp5 hm48 w1": dict(depth=0.14, tp=0.05, hm=48, cd=24, cdw=1),
         "d15 tp6 hm48 w1": dict(depth=0.15, tp=0.06, hm=48, cd=24, cdw=1),
         "d15 tp4 hm96 w1": dict(depth=0.15, tp=0.04, hm=96, cd=24, cdw=1),
         "d14 tp5 hm48 cd24": dict(depth=0.14, tp=0.05, hm=48, cd=24)}
COMMON = dict(N=1, fmax=1.0, g=0.35, gL=168)


def designs():
    out = {}
    for (bn, b), lo, hi, tl in itertools.product(BASES.items(), [0, -0.02, -0.04, -0.06], [-0.01, 0, 0.01, 0.02],
                                                 [0, -0.02]):
        out[f"A {bn} calm{lo*100:+.0f} wild{hi*100:+.0f} tpcalm{tl*100:+.0f}"] = dict(
            COMMON, **b, dlo=lo, dhi=hi, tlo=tl, vW=720)
    for vm, mn, mx, (tn, tpk), hm in itertools.product([2.2, 2.6, 3.0, 3.4], [0.06, 0.08, 0.10], [0.15, 0.17],
                                                       {"tp4": dict(tp=0.04), "tp5": dict(tp=0.05),
                                                        "tpr.3": dict(tp=0.05, tpr=0.3),
                                                        "tpr.4": dict(tp=0.05, tpr=0.4)}.items(), [48, 96]):
        out[f"B vm{vm} [{mn*100:.0f},{mx*100:.0f}] {tn} hm{hm} w1"] = dict(
            COMMON, depth=mx, vm=vm, vmin=mn, vmax=mx, vW=720, hm=hm, cd=24, cdw=1, **tpk)
    return out


if __name__ == "__main__":
    t0 = time.time()
    D = designs()
    rows = []
    for name, s in D.items():
        r, _, _ = T.evaluate(s)
        rows.append(T.row(name, s, r))
    G = pd.DataFrame(rows)
    R.report(G, "grid2", D, stress_top=6)
    print(f"{len(G)} designs, {time.time()-t0:.0f}s")
