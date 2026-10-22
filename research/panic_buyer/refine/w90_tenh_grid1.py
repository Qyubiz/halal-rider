"""Grid 1 (written before running): the plain hourly 10%-mode knobs on the 19 coins, crash guard 35%/168h fixed.
  N=1 x 100% | depth {12,13,14,15,16}% | take-profit {1.5,2,3,4,5,6,8}% | time stop {24,48,96}h |
  cooldown scheme {24h; 6h; 24h after a loss but re-arm at the next hour after a win (cdw=1)}   = 315 designs.
Selection rule (fixed by the task): a design qualifies for "ten" if %/month >= 10% in BOTH 2021-23 and 2024-26,
for "wr90" if the win rate >= 90% in both. Among qualifiers prefer the highest worst-of(2021-23, 2024-26,
since 2025) %/month; ties -> shallower worst drop. Everything here looks at 2024-26, so any pick is HINDSIGHT.
Output: grid1.csv, grid1.log (top designs, frontier by win-rate level)."""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402
import w90_tenh_report as R  # noqa: E402

DEPTH = [0.12, 0.13, 0.14, 0.15, 0.16]
TP = [0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
HM = [24, 48, 96]
CDS = {"cd24": dict(cd=24), "cd6": dict(cd=6), "cd24w1": dict(cd=24, cdw=1)}


def designs():
    out = {}
    for d, tp, hm, (cn, c) in itertools.product(DEPTH, TP, HM, CDS.items()):
        s = dict(depth=d, N=1, fmax=1.0, tp=tp, hm=hm, g=0.35, gL=168, **c)
        out[f"N1 d{d*100:.0f} tp{tp*100:g} hm{hm} {cn}"] = s
    return out


if __name__ == "__main__":
    t0 = time.time()
    D = designs()
    rows = []
    for name, s in D.items():
        r, _, _ = T.evaluate(s)
        rows.append(T.row(name, s, r))
    G = pd.DataFrame(rows)
    G.to_csv(os.path.join(T.OUT, "grid1.csv"), index=False)
    R.report(G, "grid1", D, stress_top=6)
    print(f"{len(G)} designs, {time.time()-t0:.0f}s")
