"""Grid 4 (written after grids 1-3, before running this): exits, a second order and allocation on the leaders,
crash guard 35%/168h (grid 3: 35-40% over 168h is best for every base).
Bases: d14 tp5 hm48 cdw1; d15 tp6 hm48 cdw1; d15 tp4 hm96 cdw1; d14 tp5 hm48 cd24 (10% mode); d13 tp1.5 hm96 cdw1.
Variants per base (16):
  take-profit relative to the 24h high: target max(entry x (1+tp), entry + f x (24h high - entry)),
     f {0.25, 0.40, 0.55}, for market buys only or for every fill (6)
  second smaller order at a deeper level on the same coin (ladder): first leg w {60, 80}% at the trigger, the rest
     resting at depth + {4, 7} pts, take-profit on the average entry (4)
  N=2: 2 x 50% (fmax 0.5); or fmax 1 with bids only on coins within {1, 3, 6}% of their trigger at the open
     (100% on one coin when it is the only one near, 50/50 when two are) (4)
  cooldown after a loss {12, 48}h, re-arm after a win at the next hour (2)
5 x 16 = 80 designs + 5 bases. Same fixed selection rule (HINDSIGHT)."""
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
         "d13 tp1.5 hm96 w1": dict(depth=0.13, tp=0.015, hm=96, cd=24, cdw=1)}
COMMON = dict(N=1, fmax=1.0, g=0.35, gL=168)


def variants(b):
    v = {"base": {}}
    for f, m in itertools.product([0.25, 0.40, 0.55], [1, 2]):
        v[f"tpf{f} {'mkt' if m == 1 else 'all'}"] = dict(tpf=f, tpfm=m)
    for w, dd in itertools.product([0.6, 0.8], [0.04, 0.07]):
        v[f"ladder {w*100:.0f}% +{dd*100:.0f}pts"] = dict(w=w, d2=b["depth"] + dd)
    v["N2 x50%"] = dict(N=2, fmax=0.5)
    for nr in [0.01, 0.03, 0.06]:
        v[f"N2 f1 near{nr*100:.0f}%"] = dict(N=2, fmax=1.0, near=nr)
    if b.get("cdw", -1) == 1:
        for c in [12, 48]:
            v[f"cd{c} w1"] = dict(cd=c)
    else:
        for c in [12, 48]:
            v[f"cd{c} w1"] = dict(cd=c, cdw=1)
    return v


def designs():
    out = {}
    for bn, b in BASES.items():
        for vn, v in variants(b).items():
            out[f"{bn} | {vn}"] = {**COMMON, **b, **v}
    return out


if __name__ == "__main__":
    t0 = time.time()
    D = designs()
    rows = []
    for name, s in D.items():
        r, _, _ = T.evaluate(s)
        rows.append(T.row(name, s, r))
    G = pd.DataFrame(rows)
    G = R.report(G, "grid4", D, stress_top=6)
    with open(os.path.join(T.OUT, "grid4.log"), "a") as f:
        for bn in BASES:
            g = G[G.name.str.startswith(bn + " |")]
            f.write(f"\n{bn}:\n" + "\n".join("   " + R.line(r) for _, r in g.iterrows()) + "\n")
    print(f"{len(G)} designs, {time.time()-t0:.0f}s")
