"""Task "wr90" grids. Each design is run normal (19 coins) and stressed (real bot timing: market-buy take-profit at once
and cooldown +1h; every fill 0.2% worse on buys AND sells; LUNA/FTT/SRM added with liquidation at their cut).

Usage: python w90_wr90_grid.py A|B|C|D   (results -> $PB_OUT/grid_<X>.csv)

Grid A (336, written before any wr90 result was seen): concentration x depth x take-profit x time stop, cooldown 24h,
  crash guard 35% (the 10% mode guard):
  sizing (N, fmax) {1x100%, 2x50%, 2x100% (one coin gets everything when only one is near), 3x33%, 3x50%, 4x25%, 4x50%}
  x depth {10, 12, 14, 16%} x tp {1, 1.5, 2, 3%} x hm {48, 96, 168h}.
Grid B (cooldown x guard): the 24 best grid-A qualifiers (win rate >= 90% in 2021-23 and 2024-26, ranked by the
  selection rule) x cd {6, 12, 24h} x guard {0, 25, 35, 50%} = 288 (24 of them repeat grid A).
Grid C (profit-only time stop, pmode 1): the 12 best qualifiers of A+B x {hm 48, 96h after which the target drops to
  break-even +0.25%} x {hard stop off, 25%, 40%} x {hmax off, 720h} = 144.
Grid D (HINDSIGHT refinement, placed after seeing A-C results on all periods incl. 2024-26), high-return corner:
  sizing {1x100%, 2x50%} x depth {12, 13, 14, 15%} x tp {1, 1.25, 1.5, 2, 2.5, 3%} x hm {72, 96, 120h} x cd {3, 6h},
  guard 35% = 288.
Grid E (HINDSIGHT refinement), low-drop corner: sizing {3x33%, 4x25%, 4x50%, 5x20%, 5x40%} x depth {15, 16, 17, 18%}
  x tp {1.5, 2, 2.5, 3%} x hm {96, 168h} x cd {6, 24h}, guard 35% = 320.
Grid F (HINDSIGHT, after seeing that N2 designs lose ~50% of the account to LUNA in the stressed run): 5 leading
  designs x catastrophe stop hs {off, 30, 40, 50%} (sell at an hourly close that far under entry) x guard {25, 35%} = 40.
Selection rule (fixed by the task): qualifies if win rate >= 90% in 2021-23 AND 2024-26 (trades bought in the period,
open positions marked at the end); among qualifiers the highest min(2021-23, 2024-26, since-2025) %/month; ties ->
shallower worst drop of the three periods.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wr90_lib as W  # noqa: E402

SIZING = [(1, 1.0), (2, 0.5), (2, 1.0), (3, 1 / 3), (3, 0.5), (4, 0.25), (4, 0.5)]
KEYS = ["N", "fmax", "depth", "tp", "hm", "cd", "guard", "pmode", "be", "hs", "hsa", "hmax"]
BASE = dict(pmode=0, be=0.0025, hs=0.0, hsa=0, hmax=0)
COLS = ["d_wr", "t_wr", "s_wr", "d_mo", "t_mo", "s_mo", "d_dd", "t_dd", "s_dd", "d_n", "t_n", "s_n", "d_wt", "t_wt",
        "s_wt", "d_avgw", "d_avgl", "t_avgw", "t_avgl", "s_open", "y2022", "minmo", "mindd", "minwr", "hold_max_h"]


def run_all(designs, tag):
    rows = []
    t0 = time.time()
    for i, d in enumerate(designs):
        kw = {k: d[k] for k in KEYS}
        r, _, _ = W.evaluate(kw)
        rs, _, _ = W.evaluate(kw, stress=True)
        row = dict(d)
        row.update({c: r[c] for c in COLS})
        row.update({f"x_{c}": rs[c] for c in COLS})
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(designs)} {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(W.OUT, f"grid_{tag}.csv"), index=False)
    return R


def qualify(R, wr=0.90):
    return R[(R.d_wr >= wr) & (R.t_wr >= wr)]


def rank(R):
    """Selection rule order: highest min %/month of the three periods, then shallower worst drop."""
    return R.sort_values(["minmo", "mindd"], ascending=[False, False])


def grid_a():
    return [dict(N=N, fmax=f, depth=d, tp=tp, hm=hm, cd=24, guard=0.35, **BASE)
            for (N, f), d, tp, hm in itertools.product(SIZING, (0.10, 0.12, 0.14, 0.16), (0.01, 0.015, 0.02, 0.03),
                                                       (48, 96, 168))]


def grid_b():
    A = pd.read_csv(os.path.join(W.OUT, "grid_A.csv"))
    top = rank(qualify(A)).head(24)
    out = []
    for _, r in top.iterrows():
        for cd, g in itertools.product((6, 12, 24), (0.0, 0.25, 0.35, 0.50)):
            out.append(dict(N=int(r.N), fmax=float(r.fmax), depth=float(r.depth), tp=float(r.tp), hm=int(r.hm), cd=cd,
                            guard=g, **BASE))
    return out


def grid_c():
    AB = pd.concat([pd.read_csv(os.path.join(W.OUT, f"grid_{x}.csv")) for x in ("A", "B")])
    AB = AB.drop_duplicates(subset=KEYS)
    top = rank(qualify(AB)).head(12)
    out = []
    for _, r in top.iterrows():
        for hm, hs, hmax in itertools.product((48, 96), (0.0, 0.25, 0.40), (0, 720)):
            out.append(dict(N=int(r.N), fmax=float(r.fmax), depth=float(r.depth), tp=float(r.tp), hm=hm, cd=int(r.cd),
                            guard=float(r.guard), pmode=1, be=0.0025, hs=hs, hsa=0, hmax=hmax))
    return out


def grid_d():
    return [dict(N=N, fmax=f, depth=d, tp=tp, hm=hm, cd=cd, guard=0.35, **BASE)
            for (N, f), d, tp, hm, cd in itertools.product(((1, 1.0), (2, 0.5)), (0.12, 0.13, 0.14, 0.15),
                                                           (0.01, 0.0125, 0.015, 0.02, 0.025, 0.03), (72, 96, 120), (3, 6))]


def grid_e():
    return [dict(N=N, fmax=f, depth=d, tp=tp, hm=hm, cd=cd, guard=0.35, **BASE)
            for (N, f), d, tp, hm, cd in itertools.product(((3, 1 / 3), (4, 0.25), (4, 0.5), (5, 0.2), (5, 0.4)),
                                                           (0.15, 0.16, 0.17, 0.18), (0.015, 0.02, 0.025, 0.03),
                                                           (96, 168), (6, 24))]


def grid_f():
    lead = [(1, 1.0, 0.13, 0.0125, 96, 3), (1, 1.0, 0.14, 0.025, 96, 3), (2, 0.5, 0.14, 0.025, 96, 3),
            (2, 0.5, 0.15, 0.03, 96, 3), (3, 1 / 3, 0.14, 0.02, 96, 6)]
    return [dict(N=N, fmax=f, depth=d, tp=tp, hm=hm, cd=cd, guard=g, pmode=0, be=0.0025, hs=hs, hsa=0, hmax=0)
            for (N, f, d, tp, hm, cd), hs, g in itertools.product(lead, (0.0, 0.30, 0.40, 0.50), (0.25, 0.35))]


def main(which):
    g = {"A": grid_a, "B": grid_b, "C": grid_c, "D": grid_d, "E": grid_e, "F": grid_f}[which]()
    print(f"grid {which}: {len(g)} designs", flush=True)
    R = run_all(g, which)
    Q = rank(qualify(R))
    print(f"grid {which}: {len(Q)} of {len(R)} designs have win rate >= 90% in 2021-23 and 2024-26")
    show = ["N", "fmax", "depth", "tp", "hm", "cd", "guard", "pmode", "hs", "hmax", "d_wr", "t_wr", "s_wr", "d_mo", "t_mo",
            "s_mo", "mindd", "x_d_mo", "x_t_mo", "x_s_mo", "x_mindd", "x_d_wr", "x_t_wr"]
    with pd.option_context("display.width", 250, "display.max_columns", 40, "display.float_format", "{:.4f}".format):
        print(Q[show].head(25).to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1])
