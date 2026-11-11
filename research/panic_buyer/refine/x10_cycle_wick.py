"""Lever "cycle", phase 4: a deep "wick layer" in its own sub-account next to the fast main bot.

PRE-REGISTRATION (written 22 Oct 2026 after phases 1-3 and before any phase-4 run; not changed after the run)
---------------------------------------------------------------------------------------------------------------
Caveat known before running: this idea comes from looking at 2024-26 trades. The hindsight 10%/month designs owe a large
part of 2024-26 to one flash crash (10 Oct 2025: RENDER -85%, LINK -62%, LTC -57% inside 5-15 minutes). So the idea is
hindsight-motivated; the choice inside the grid is made on stressed 2021-23 only (which has its own flash wicks:
19 May 2021, 4 Dec 2021, plus the LUNA/FTT/SRM collapses that a deep-bid bot buys).

WICK LAYER (512 settings, x10_cycle_guard.ckernel3, cd 0, plain exit, no skip/stop guards since they block wicks):
  C {5, 15} min x W {1, 3} h x depth {15, 20, 30, 40}% x tp {3, 5, 10, 20}% x hm {6, 24} h
  x sizing {1 x 100%, 3 x 33.3%} x lcd {off, 72h block after a loss}
MAIN BOT (honest picks only): phase-2 rule-B2 pick (5m) and phase-1 rule-B pick (15m), unchanged.
COMBINATION: two separately funded sub-accounts, no rebalancing: account = (1 - a) x main + a x wick, a in
  {0.25, 0.5, 1.0}; plus each main alone (a = 0). Equity is compared on the 15-minute grid whenever either part is 15m.
Every part runs normal and stressed (buys 0.2% worse + LUNA/FTT/SRM, liquidation at their cut).
RULE B4 (only the stressed 2021-23 columns): keep 2022 >= 0% and 2021-23 worst drop >= -50%; pick the highest 2021-23
  %/month (tie: shallower drop). 2024-26 and since 2025 reported once, all four reports. HINDSIGHT: best normal 2024-26.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_grid2 as G  # noqa: E402
import x10_cycle_lib as L  # noqa: E402

MAINS = {
    "phase-2 B2 pick": dict(C=5, W=1, depth=0.10, tp=0.015, hm=6, cd=0, N=1, fmax=1.0, lcd=72, stop=0.0, skip=0.0,
                            exit="plain", dual=0.0),
    "phase-1 B pick": dict(C=15, W=2, depth=0.08, tp=0.03, hm=6, cd=2, N=3, fmax=1.0 / 3.0, lcd=0, stop=0.0, skip=0.0,
                           exit="plain", dual=0.0),
}
WEIGHTS = [0.25, 0.5, 1.0]


def wick_settings():
    for C in (5, 15):
        for W in (1, 3):
            for d in (0.15, 0.20, 0.30, 0.40):
                for tp in (0.03, 0.05, 0.10, 0.20):
                    for hm in (6, 24):
                        for N, f in ((1, 1.0), (3, 1.0 / 3.0)):
                            for lcd in (0, 72):
                                yield dict(C=C, W=W, depth=d, tp=tp, hm=hm, cd=0, N=N, fmax=f, lcd=lcd, stop=0.0,
                                           skip=0.0, exit="plain", dual=0.0)


class Env:
    def __init__(self, C, dead):
        self.B = L.load_bars(C, dead); self.bph = 60 // C
        self.cf = L.ffill_close(self.B); self.clk = L.Clock(self.B["t"]); self.refs = {}
        self.dead = dead

    def run(self, s, pen=0.0, pens=0.0):
        for w in (s["W"], 24, 168):
            if w not in self.refs:
                self.refs[w] = L.make_ref(self.B, w * self.bph)
        return G.run_one(self.B, self.refs, self.cf, self.bph, s, pen=pen, pens=pens, liquidate=self.dead)


def to15(eq, C):
    """Equity at 15-minute bar closes (5m bar 3i+2 closes with 15m bar i)."""
    return eq if C == 15 else eq[2::3]


def combine(main_eq, mainC, wick_eq, wickC, a):
    if a == 0.0:
        return main_eq, mainC
    if a == 1.0:
        return wick_eq, wickC
    if mainC == wickC:
        return (1 - a) * main_eq + a * wick_eq, mainC
    return (1 - a) * to15(main_eq, mainC) + a * to15(wick_eq, wickC), 15


def main():
    t0 = time.time()
    clks = {}
    eqs = {}            # (stressed, kind, idx) -> (eq, C)
    for stressed in (False, True):
        pen = G.STRESS_PEN if stressed else 0.0
        for C in (5, 15):
            env = Env(C, stressed)
            clks[C] = env.clk
            for nm, s in MAINS.items():
                if s["C"] == C:
                    eqs[(stressed, "main", nm)] = (env.run(s, pen=pen)[0], C)
            for i, s in enumerate(wick_settings()):
                if s["C"] == C:
                    eqs[(stressed, "wick", i)] = (env.run(s, pen=pen)[0], C)
            del env
            print(f"  runs done: stressed={stressed} C={C} ({time.time() - t0:.0f}s)", flush=True)
    W = list(wick_settings())
    rows = []
    for mn in MAINS:
        for a in [0.0] + WEIGHTS:
            for i in (range(len(W)) if a > 0 else [None]):
                if a == 1.0 and mn != list(MAINS)[0]:
                    continue                                  # wick alone: count once
                r = {"main": mn if a < 1.0 else "-", "a": a}
                if i is not None:
                    r.update({f"w_{k}": v for k, v in W[i].items() if k in ("C", "W", "depth", "tp", "hm", "N", "lcd")})
                    r["wick_i"] = i
                for stressed, tag in ((False, ""), (True, "_s")):
                    me, mc = eqs[(stressed, "main", mn)]
                    we, wc = eqs[(stressed, "wick", i)] if i is not None else (None, None)
                    e, C = combine(me, mc, we, wc, a)
                    m = L.summarize(clks[C], e)
                    m["run_dd"] = float((e / np.maximum.accumulate(e) - 1).min())
                    r.update({k + tag: v for k, v in m.items()})
                rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(L.OUT, "wick_grid.csv"), index=False)
    print(f"saved {len(R)} combinations ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
