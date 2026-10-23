"""Grid 5 (written after grids 1-4, before running this): the remaining small changes and a plateau check on the
three leaders, crash guard 35%/168h. Bases: d14 tp5 hm48 cdw1 (best worst-of-three), d15 tp4 hm96 cdw1 (best
with a 90% win rate), d13 tp1.5 hm96 cdw1 (highest win rate near the top).
Per base (15):
  trigger window: highest close of the last {12, 36, 48, 72}h instead of 24h (EXTRA lever, not on the task list)
  slower take-profit: placed at the 2nd or 3rd hourly check; or no resting order, sell only at an hourly open at
     or above target from the 1st / 2nd check (EXTRA lever)
  take-profit relative to the 24h high only: target max(entry +2%, entry + f x (24h high - entry)), f {0.3, 0.4, 0.5}
  time stop {72, 120}h
  depth -0.5 / +0.5 point (plateau)
3 x 15 = 45 designs + 3 bases. Same fixed selection rule (HINDSIGHT)."""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402
import w90_tenh_report as R  # noqa: E402

BASES = {"d14 tp5 hm48 w1": dict(depth=0.14, tp=0.05, hm=48, cd=24, cdw=1),
         "d15 tp4 hm96 w1": dict(depth=0.15, tp=0.04, hm=96, cd=24, cdw=1),
         "d13 tp1.5 hm96 w1": dict(depth=0.13, tp=0.015, hm=96, cd=24, cdw=1)}
COMMON = dict(N=1, fmax=1.0, g=0.35, gL=168)


def variants(b):
    v = {"base": {}}
    for W in [12, 36, 48, 72]:
        v[f"window {W}h"] = dict(refW=W)
    v["tpwait2"] = dict(tpwait=2)
    v["tpwait3"] = dict(tpwait=3)
    v["no rest tpwait1"] = dict(tprest=False)
    v["no rest tpwait2"] = dict(tprest=False, tpwait=2)
    for f in [0.3, 0.4, 0.5]:
        v[f"tp vs 24h high f{f}"] = dict(tp=0.02, tpf=f, tpfm=2)
    for hm in [72, 120]:
        v[f"hm{hm}"] = dict(hm=hm)
    for dd in [-0.005, 0.005]:
        v[f"depth{dd*100:+.1f}"] = dict(depth=round(b["depth"] + dd, 4))
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
    G = R.report(G, "grid5", D, stress_top=4)
    with open(os.path.join(T.OUT, "grid5.log"), "a") as f:
        for bn in BASES:
            g = G[G.name.str.startswith(bn + " |")]
            f.write(f"\n{bn}:\n" + "\n".join("   " + R.line(r) for _, r in g.iterrows()) + "\n")
    print(f"{len(G)} designs, {time.time()-t0:.0f}s")
