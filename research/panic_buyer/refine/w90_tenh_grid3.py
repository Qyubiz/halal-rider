"""Grid 3 (written after grids 1-2, before running this): crash-guard threshold and lookback on the grid-1/2 leaders.
Bases (N=1 x 100%): d14 tp5 hm48 cd24 + re-arm after a win (cdw=1); d15 tp6 hm48 cdw1; d15 tp4 hm96 cdw1;
d14 tp5 hm48 cd24 (10% mode); d13 tp1.5 hm96 cdw1 (high win rate); d15 tp6 hm48 cd24.
Guard: off, or skip a coin whose last close is more than g {25, 30, 35, 40, 50}% under its highest close of the
previous L {72, 168, 336} hours. 6 x 16 = 96 designs. Same fixed selection rule (HINDSIGHT)."""
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
         "d14 tp5 hm48 cd24": dict(depth=0.14, tp=0.05, hm=48, cd=24),
         "d13 tp1.5 hm96 w1": dict(depth=0.13, tp=0.015, hm=96, cd=24, cdw=1),
         "d15 tp6 hm48 cd24": dict(depth=0.15, tp=0.06, hm=48, cd=24)}
GUARDS = [(0.0, 168)] + list(itertools.product([0.25, 0.30, 0.35, 0.40, 0.50], [72, 168, 336]))


def designs():
    out = {}
    for (bn, b), (g, L) in itertools.product(BASES.items(), GUARDS):
        gn = "noguard" if g == 0 else f"g{g*100:.0f}/{L}h"
        out[f"{bn} {gn}"] = dict(b, N=1, fmax=1.0, g=g, gL=L)
    return out


if __name__ == "__main__":
    t0 = time.time()
    D = designs()
    rows = []
    for name, s in D.items():
        r, _, _ = T.evaluate(s)
        rows.append(T.row(name, s, r))
    G = pd.DataFrame(rows)
    G = R.report(G, "grid3", D, stress_top=6)
    # guard table per base: worst-of-three %/month and 2021-23 drop
    with open(os.path.join(T.OUT, "grid3.log"), "a") as f:
        for bn in BASES:
            g = G[G.name.str.startswith(bn + " ")]
            f.write(f"\n{bn}: " + " | ".join(f"{n[len(bn)+1:]} min {r.min_mo*100:+.1f} 21-23dd {r.d_dd*100:.0f} 24-26 {r.t_mo*100:+.1f} 25- {r.s_mo*100:+.1f}"
                                           for n, r in zip(g.name, g.itertuples())) + "\n")
    print(f"{len(G)} designs, {time.time()-t0:.0f}s")
