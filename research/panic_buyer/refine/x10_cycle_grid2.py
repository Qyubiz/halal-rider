"""Lever "cycle", phase 2: collapse guards, exit variants and a second (slow) trigger on the fast bot.

PRE-REGISTRATION (written 22 Oct 2026 after phase 1 and before any phase-2 run; not changed after the run)
---------------------------------------------------------------------------------------------------------
Caveat known before running: this grid was designed AFTER seeing all phase-1 results, including 2024-26. The entry
ranges are the phase-1 values with the best 2021-23 medians (W 1-3h, depth 6-10%, tp 1.5-3%, hm 6-12h, cd 0,
N 1-3; the 2024-26 medians happen to point the same way), so the grid's location is not fully blind. The choice
inside it is made on 2021-23 only.

Phase 1 showed the honest all-in pick is wiped out by LUNA (the fast trigger re-buys a collapsing coin). So the
PRIMARY rule is now survivorship-aware (rule B2).

GRID (6,912 settings, kernel x10_cycle_guard.ckernel3):
  entries (144): C {5, 15} min x W {1, 3} h x depth {6, 8, 10}% x tp {1.5, 3}% x hm {6, 12} h x cd 0
                 x sizing {1 x 100%, 2 x 50%, 3 x 33.3%}
  guards (8):    lcd {off, 72h block after a losing exit} x stop {off, sell at a close 15% under entry}
                 x skip {off, no bid if the open is >30% under the 7-day high close}
  exits (3):     plain (take-profit at the next check) | trail (when the take-profit is hit, hold the whole
                 position and sell at a check close 1% under its high since, part=0 trail=0.01) | wait (take-profit
                 active only 1 hour after the fill)
  slow trigger (2): off | on (bid at the higher of the fast trigger and 24h-high close -12%)
Each setting runs normal (19 coins) and stressed (every buy 0.2% worse + LUNA/FTT/SRM with 0.1% slippage).

RULE B2 (primary; only the 2021-23 columns of the STRESSED run): keep 2022 >= 0% and 2021-23 worst drop >= -50%;
  pick the highest 2021-23 %/month (tie: shallower drop). If none: highest 2021-23 CAGR / |drop|.
RULE A2 (secondary; only the 2021-23 columns of the NORMAL run): same rule.
Each pick: 2024-26 and since 2025 reported once, all four reports. HINDSIGHT (labelled): best 2024-26 %/month of
the normal runs; best 2024-26 %/month among settings whose stressed run never loses more than 50%.
PHASE 3 (also fixed now): the B2 and A2 picks re-run with 1-minute checks (C=1, same hours), all four reports.
"""
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_guard as G3  # noqa: E402
import x10_cycle_lib as L  # noqa: E402

CS = [5, 15]
WS = [1, 3]
DEPTHS = [0.06, 0.08, 0.10]
TPS = [0.015, 0.03]
HMS = [6, 12]
SIZING = [(1, 1.0), (2, 0.5), (3, 1.0 / 3.0)]
LCDS = [0, 72]
STOPS = [0.0, 0.15]
SKIPS = [0.0, 0.30]
EXITS = ["plain", "trail", "wait"]
DUALS = [0.0, 0.12]
STRESS_PEN = 0.002


def exit_kw(x, bph):
    if x == "trail":
        return dict(part=0.0, trail=0.01)
    if x == "wait":
        return dict(tpwait=bph)
    return {}


def run_one(B, refs, cf, bph, s, pen=0.0, pens=0.0, liquidate=False):
    """cd from s (phase-1 settings have cd 1-2h; phase 2 always 0). liquidate: see x10_cycle_guard (added after the
    phase-2 run; off reproduces the phase-2 grid)."""
    ref = refs[s["W"]]
    return G3.crun3(B, ref, cf, s["depth"], s["N"], s["fmax"], s["tp"], s["hm"] * bph, int(s.get("cd", 0)) * bph,
                    ref2=refs[24], depth2=s["dual"], ref3=refs[168], skip=s["skip"], lcd=s["lcd"] * bph,
                    stop=s["stop"], pen=pen, pens=pens, liquidate=liquidate, **exit_kw(s["exit"], bph))


def job(args):
    C, W, stressed = args
    B = L.load_bars(C, dead=stressed)
    bph = 60 // C
    cf = L.ffill_close(B); clk = L.Clock(B["t"])
    refs = {W: L.make_ref(B, W * bph), 24: L.make_ref(B, 24 * bph), 168: L.make_ref(B, 168 * bph)}
    rows = []
    for d in DEPTHS:
        for tp in TPS:
            for hm in HMS:
                for N, f in SIZING:
                    for lcd in LCDS:
                        for stop in STOPS:
                            for skip in SKIPS:
                                for x in EXITS:
                                    for dual in DUALS:
                                        s = dict(C=C, W=W, depth=d, tp=tp, hm=hm, cd=0, N=N, fmax=f, lcd=lcd,
                                                 stop=stop, skip=skip, exit=x, dual=dual)
                                        eq, lg = run_one(B, refs, cf, bph, s, pen=STRESS_PEN if stressed else 0.0)
                                        r = L.summarize(clk, eq, lg)
                                        r["run_dd"] = float((eq / pd.Series(eq).cummax().values - 1).min())
                                        r.update(s); r["fmax"] = round(f, 4)
                                        rows.append(r)
    return stressed, rows


KEYS = ["C", "W", "depth", "tp", "hm", "cd", "N", "fmax", "lcd", "stop", "skip", "exit", "dual"]


def main(workers=2):
    tasks = [(C, W, s) for C in CS for W in WS for s in (False, True)]
    t0 = time.time()
    res = {False: [], True: []}
    with Pool(workers) as pool:
        for s, rows in pool.imap_unordered(job, tasks):
            res[s] += rows
            print(f"  done {len(res[False]) + len(res[True])} runs, {time.time() - t0:.0f}s", flush=True)
    A = pd.DataFrame(res[False]).sort_values(KEYS).reset_index(drop=True)
    S = pd.DataFrame(res[True]).sort_values(KEYS).reset_index(drop=True)
    A.to_csv(os.path.join(L.OUT, "grid2_normal.csv"), index=False)
    S.to_csv(os.path.join(L.OUT, "grid2_stressed.csv"), index=False)
    print(f"saved {len(A)} + {len(S)} runs to {L.OUT}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
