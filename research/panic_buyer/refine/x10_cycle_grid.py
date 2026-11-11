"""Lever "cycle", phase 1: a faster Panic Buyer (checks every 5 or 15 minutes, short-window trigger).

PRE-REGISTRATION (written 22 Oct 2026 before any grid result was seen; not changed after the run)
---------------------------------------------------------------------------------------------------
Only the six sample runs used to time the kernel were seen before this was written.

GRID (7,680 settings; 19 live coins; the bot checks every C minutes, places fully funded resting bids on the N
coins closest to their trigger, take-profit placed at the next check, time stop, cooldown):
    C (check interval)        {5, 15} minutes
    W (trigger window)        {1, 2, 3, 6} hours: trigger = highest C-minute close of the last W hours x (1 - depth)
    depth                     {4, 5, 6, 8, 10} %
    tp (take-profit)          {1, 1.5, 2, 3} %
    hm (time stop)            {2, 4, 6, 12} hours
    cd (cooldown after sale)  {0, 1, 2} hours
    sizing (N coins x fmax)   {1 x 100%, 2 x 50%, 3 x 33.3%, 4 x 25%}
Each setting is run twice: normal (19 coins, 0.1% fee/side, 0.02% market slippage, limit buys need 0.1%
through) and stressed (every buy 0.2% worse, as engine `pen`, AND LUNA/FTT/SRM added with 0.1% slippage).

RULE A (primary; uses only the 2021-23 columns of the normal run):
    keep settings with calendar-2022 return >= 0% and 2021-23 worst drop >= -50%;
    pick the highest 2021-23 %/month (tie: shallower 2021-23 drop).
    If none qualifies: the highest 2021-23 CAGR / |2021-23 worst drop|.
RULE B (secondary, survivorship-aware; uses only the 2021-23 columns of the STRESSED run): same as rule A.
For each pick, 2024-26 and since-2025 are then reported once, with all four reports.
HINDSIGHT (labelled as such): the grid's best 2024-26 %/month (normal run), and the best with a 2024-26 drop
no deeper than -50%; plus the share of rule-A-qualifying settings that reach 10%/month in 2024-26.
%/month = multiple ** (1 / months) - 1. 10%/month over 2024-01-01..2026-10-22 = x24.9.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_lib as L  # noqa: E402

CS = [5, 15]
WS = [1, 2, 3, 6]
DEPTHS = [0.04, 0.05, 0.06, 0.08, 0.10]
TPS = [0.01, 0.015, 0.02, 0.03]
HMS = [2, 4, 6, 12]
CDS = [0, 1, 2]
SIZING = [(1, 1.0), (2, 0.5), (3, 1.0 / 3.0), (4, 0.25)]
STRESS_PEN = 0.002


def job(args):
    C, W, stressed = args
    B = L.load_bars(C, dead=stressed)
    bph = 60 // C
    cf = L.ffill_close(B); ref = L.make_ref(B, W * bph); clk = L.Clock(B["t"])
    rows = []
    for d in DEPTHS:
        for tp in TPS:
            for hm in HMS:
                for cd in CDS:
                    for N, f in SIZING:
                        eq, lg = L.crun(B, ref, cf, d, N, f, tp, hm * bph, cd * bph,
                                        pen=STRESS_PEN if stressed else 0.0)
                        r = L.summarize(clk, eq, lg)
                        r.update(C=C, W=W, depth=d, tp=tp, hm=hm, cd=cd, N=N, fmax=round(f, 4))
                        rows.append(r)
    return stressed, rows


def main(workers=2):
    tasks = [(C, W, s) for s in (False, True) for C in CS for W in WS]
    t0 = time.time()
    res = {False: [], True: []}
    with Pool(workers) as pool:
        for s, rows in pool.imap_unordered(job, tasks):
            res[s] += rows
            print(f"  done {len(res[False]) + len(res[True])} runs, {time.time() - t0:.0f}s", flush=True)
    keys = ["C", "W", "depth", "tp", "hm", "cd", "N", "fmax"]
    A = pd.DataFrame(res[False]).sort_values(keys).reset_index(drop=True)
    S = pd.DataFrame(res[True]).sort_values(keys).reset_index(drop=True)
    A.to_csv(os.path.join(L.OUT, "grid_normal.csv"), index=False)
    S.to_csv(os.path.join(L.OUT, "grid_stressed.csv"), index=False)
    print(f"saved {len(A)} + {len(S)} runs to {L.OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
