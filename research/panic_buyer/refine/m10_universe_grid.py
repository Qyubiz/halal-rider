"""m10 "universe" grid. Everything below was fixed before any grid result was seen.

Core = 90% mode on the user's 19 coins: 14% under the 24h-high close, +2.5% take-profit, 96h time stop, 3h cooldown,
crash guard 35% (skip coins > 35% under their 7-day high). Tier = monthly top M other screened coins.

STAGE A (132 designs): max positions N in {1, 2, 3, 4} (slots, each 1/N of the money)
  x tier {none} + M {20, 40, 60, 100} x rank {v30, v365} x tier depth {14, 18, 22, 26%}; tier size x1, tier tp 2.5%.
STAGE B (<= 144 designs): the 3 best tier designs (M > 0) of stage A by the score below (qualifiers first)
  x liquidity floor lmin {0, 2M, 10M, 30M USDT/day (30-day average)} x age floor amin {0, 90, 365 days}
  x tier size st {0.5, 1} x tier cap tcap {none, 1 tier position at a time}.

Each design runs twice: normal fills, and every buy and sell fill 0.2% worse.

PICK RULE (2021-23 only):
  qualify: 2021-23 win rate >= 92% (normal) and >= 90% (fills 0.2% worse); 2021-23 worst drop no deeper than -65%
           (90% mode: -64%); worst single trade no worse than -50% of the account (90% mode: -50%).
  score:   2021-23 %/month with fills 0.2% worse. Highest score among qualifiers (stages A and B together) = the pick.
2024-26 and since 2025 are only printed in the final script (m10_universe_final.py), once, for the pick.
This script writes the 2024-26 columns to the csv but prints only 2021-23.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_universe_lib as M  # noqa: E402

CORE = dict(M.MODE90)


def stage_a():
    ds = []
    for N in (1, 2, 3, 4):
        ds.append(dict(CORE, N=N, M=0))
        for Mt, rk, dt in itertools.product((20, 40, 60, 100), ("v30", "v365"), (0.14, 0.18, 0.22, 0.26)):
            ds.append(dict(CORE, N=N, M=Mt, rk=rk, dt=dt))
    return ds


def stage_b(bases):
    ds = []
    for b in bases:
        for lmin, amin, st, tcap in itertools.product((0, 2e6, 10e6, 30e6), (0, 90, 365), (0.5, 1.0), (-1, 1)):
            if lmin == 0 and amin == 0 and st == 1.0 and tcap == -1:
                continue                                 # = the base itself (already in stage A)
            ds.append(dict(b, lmin=lmin, amin=amin, st=st, tcap=tcap))
    return ds


def qualifies(r):
    return bool(r["d_win"] >= 0.92 and r["P_d_win"] >= 0.90 and r["d_dd"] >= -0.65 and r["d_wacct"] >= -0.50)


def run_all(W, ds, stage, t0):
    rows = []
    for i, d in enumerate(ds):
        r, *_ = M.evaluate(d, W=W)
        rp, *_ = M.evaluate(d, W=W, pen=0.002, pens=0.002)
        row = dict(stage=stage, name=M.name(d), design=repr(d), **r, **{f"P_{k}": v for k, v in rp.items()})
        row["qual"] = qualifies(row)
        row["score"] = rp["d_mo"]
        rows.append(row)
        if (i + 1) % 20 == 0:
            print(f"  stage {stage}: {i + 1}/{len(ds)} {time.time() - t0:.0f}s", flush=True)
    return rows


def show(df, n=25):
    cols = ["name", "qual", "d_win", "P_d_win", "d_mo", "P_d_mo", "d_dd", "d_wacct", "d_tpm", "d_n_out", "d_pnl_out"]
    x = df.sort_values(["qual", "score"], ascending=False).head(n)[cols].copy()
    for c in ("d_win", "P_d_win", "d_mo", "P_d_mo", "d_dd", "d_wacct", "d_pnl_out"):
        x[c] = (x[c] * 100).round(2)
    x["d_tpm"] = x["d_tpm"].round(1)
    return x.to_string(index=False)


def main():
    W = M.build()
    log = open(os.path.join(M.OUT, "grid.log"), "w")

    def say(s=""):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    t0 = time.time()
    rows = run_all(W, stage_a(), "A", t0)
    A = pd.DataFrame(rows)
    say(f"STAGE A: {len(A)} designs, {int(A.qual.sum())} qualify (2021-23 only; P_ = fills 0.2% worse; "
        f"d_pnl_out = account P&L points from coins outside the 19)")
    say(show(A, 30))
    tier = A[A.design.str.contains("'M': [1-9]", regex=True)]
    bases = tier.sort_values(["qual", "score"], ascending=False).head(3)
    say("\nstage B bases: " + "; ".join(bases.name))
    rows += run_all(W, stage_b([eval(x) for x in bases.design]), "B", t0)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(M.OUT, "grid.csv"), index=False)
    B = df[df.stage == "B"]
    say(f"\nSTAGE B: {len(B)} designs, {int(B.qual.sum())} qualify")
    say(show(B, 30))
    q = df[df.qual]
    if len(q):
        p = q.sort_values("score", ascending=False).iloc[0]
        say(f"\nPICK (rule, 2021-23 only): {p['name']}\n  design {p['design']}\n  2021-23 WR {p['d_win'] * 100:.1f}% "
            f"(fills 0.2% worse {p['P_d_win'] * 100:.1f}%), {p['d_mo'] * 100:+.2f}%/mo (worse fills "
            f"{p['P_d_mo'] * 100:+.2f}%), drop {p['d_dd'] * 100:.0f}%, {p['d_tpm']:.1f} trades/mo")
        ref = df[(df.stage == "A") & (df.name == M.name(dict(CORE, N=1, M=0)))].iloc[0]
        say(f"  90% mode on the same panel, 2021-23: WR {ref['d_win'] * 100:.1f}% ({ref['P_d_win'] * 100:.1f}%), "
            f"{ref['d_mo'] * 100:+.2f}%/mo ({ref['P_d_mo'] * 100:+.2f}%), drop {ref['d_dd'] * 100:.0f}%, "
            f"{ref['d_tpm']:.1f} trades/mo")
        open(os.path.join(M.OUT, "pick.txt"), "w").write(p["design"])
    else:
        say("\nno design qualifies")
    say(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
