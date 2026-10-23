"""m10 ladder: the ONE look at 2024-26, since 2025 and the 2018-20 holdout for the setup chosen on 2021-23.

The pick (PICK) comes from m10_ladder_grid2.py's rule, applied to 2021-23 only (grid2.log / grid3.log in $PB_OUT).
It is run once, unchanged, on:
  2021-26 continuous (19 coins; 2021-23, 2024-26 and since Jan 2025 are read off the same equity curve),
  2018-20 holdout (16 coins that existed then, engine.load_pre),
each as: normal | every fill 0.2% worse (buys and sells) | bot timing (market-buy take-profit at once, cooldown +1h)
+ fills 0.2% worse | +LUNA/FTT/SRM (sold at their last trading hour).
90% mode (published: 1 coin x 100%, 14% dip, +2.5%, 96h, 3h cooldown, crash guard 35%) is run through the same code
and the same metrics for comparison.

Metrics per period (ladders BOUGHT in the period): strict win rate (a win = take-profit exit above cost, or still
open at the end marked above cost; time stops / max-loss exits / collapse cuts count as losses), %/month compounded,
worst drop (hourly equity), worst trade as % of the account, trades per month, days held (mean per trade, mean
weighted by money, longest), share of hours with a position older than 30 days.

Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 m10_ladder_final.py
"""
import os

import numpy as np
import pandas as pd

import engine as E
import m10_ladder_lib as M

PICK = None          # set below from the stage 2-3 rule pick (2021-23 only)
REF90 = dict(legs=((0.14, 1),), H=1, fmax=1.0, tp=0.025, hm=96, cd=3, guard=0.35)
PERIODS = (("2021-23", M.T_D0, M.T_T0), ("2024-26", M.T_T0, None), ("since 2025", M.T_S0, None))


def variants(spec):
    cd = int(spec.get("cd", 3))
    return (("normal", {}, False), ("fills 0.2% worse", dict(pen=0.002, pens=0.002), False),
            ("bot timing + fills 0.2% worse", dict(mtp_now=True, cd=cd + 1, pen=0.002, pens=0.002), False),
            ("+LUNA/FTT/SRM", dict(liquidate=True), True))


def report(spec, label):
    rows = []
    for lab, over, dead in variants(spec):
        syms, idx, P = M.market(dead)
        e, lg = M.lrun(P, idx, **{**spec, **over})
        for per, a, b in PERIODS:
            rows.append({"setup": label, "test": lab, "period": per, **M.stats(e, lg, idx, a, b)})
        if lab == "normal":
            yr = {y: M.period(e, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01") if y < 2026 else None)[0] - 1
                  for y in range(2021, 2027)}
            print(f"  {label} years: " + " ".join(f"{y} {v*100:+.0f}%" for y, v in yr.items()))
            mo = e.resample("ME").last()
            mo = pd.concat([pd.Series([1.0], [e.index[0]]), mo]).pct_change().dropna()
            mo.to_csv(os.path.join(M.OUT, f"monthly_{label.split()[0]}.csv"))
        Ph = M.holdout_market(dead)
        eh, lh = M.lrun(Ph[2], Ph[1], **{**spec, **over})
        rows.append({"setup": label, "test": lab, "period": "2018-20 holdout",
                     **M.stats(eh, lh, Ph[1], E.H_START, E.H_END)})
        if lab == "normal":
            yr = {y: M.period(eh, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01"))[0] - 1 for y in (2018, 2019, 2020)}
            print(f"  {label} holdout years: " + " ".join(f"{y} {v*100:+.0f}%" for y, v in yr.items())
                  + f" ({len(Ph[0])} coins)")
    return rows


def show(R):
    print(f"{'setup':10s} {'test':30s} {'period':16s} {'WR':>5s} {'WR>0':>5s} {'%/mo':>7s} {'drop':>5s} {'worst':>6s} "
          f"{'tr/mo':>5s} {'days':>5s} {'d$w':>5s} {'dmax':>5s} {'>30d':>5s} {'ts':>3s} {'ml':>3s} {'open':>4s}")
    for _, r in R.iterrows():
        print(f"{r.setup[:10]:10s} {r.test[:30]:30s} {r.period:16s} {r.wr*100:5.1f} {r.wr_pos*100:5.1f} {r.mo*100:+7.2f} "
              f"{r.dd*100:5.0f} {r.wt*100:+6.1f} {r.tpm:5.1f} {r.hold:5.1f} {r.hold_cw:5.1f} {r.hold_max:5.0f} "
              f"{r.stuck30*100:4.1f}% {int(r.ts):3d} {int(r.mlx):3d} {int(r.open):4d}")


if __name__ == "__main__":
    import m10_ladder_pick as PK
    PICK = PK.PICK
    print(f"PICK (chosen on 2021-23 only): {M.name(PICK)}\n  spec {PICK}")
    print("90% mode reference: " + M.name(REF90))
    R = pd.DataFrame(report(PICK, "LADDER pick") + report(REF90, "90% mode"))
    R.to_csv(os.path.join(M.OUT, "final.csv"), index=False)
    print("\nWR = strict win rate; worst = worst trade as % of the account; days = mean days held per trade; d$w = "
          "money-weighted; dmax = longest; >30d = share of hours with a position older than 30 days")
    for setup in ("LADDER pick", "90% mode"):
        print()
        show(R[R.setup == setup])
