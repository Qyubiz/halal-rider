"""Lever "idle cash", round 2: less friction. Pre-registered grid, pick on 2021-23 only, one look at 2024-26.

PRE-REGISTRATION (written 22 Oct 2026 after round 1 (x10_idle_grid.py) was seen, before this script was run;
not changed after the run)
-----------------------------------------------------------------------------------------------------------
Round 1 showed: momentum parking is ruinous (drops -55% to -100%); BTC/ETH with a 50-day filter and a 24h
re-buy wait were the least bad; parking costs (fees on forced sales and re-buys) were up to 77% of equity
summed over 2021-26. This round therefore only varies the friction and the BTC/ETH engine. The grid's LOCATION
is informed by round 1 (including its 2024-26 numbers); only the choice inside it is honest.

GRID: 4 bases (as round 1) x (no parking + 5 parking engines x 5 arming rules x 3 re-buy waits x 2 fractions)
      = 604 runs, 19 coins, normal costs, band 5% of equity.
  parking engines (decided daily at 00:00 UTC unless noted):
     BTC f50              BTC above its 50-day average
     ETH f50              ETH above its 50-day average
     BTC+ETH f50          50/50, each with its own filter
     BTC/ETH f50 renorm   the ones above their 50-day average, equal weight (100% in one if only one is on)
     BTC+ETH f50 calm5    BTC+ETH f50, and no parking in any hour when BTC's last close is > 5% under its 24h high
  arming: (mode 0, 3%) (mode 0, 5%) (mode 0, 7%) (mode 1 watcher, 1%) (mode 1, 2%)
  re-buy wait after a forced/watcher sale: 24, 72, 168 hours
  parking fraction of the idle cash: 100%, 50%

SELECTION RULE (unchanged from round 1; 2021-23 columns only):
  keep runs with calendar-2022 return >= 0% and 2021-23 worst drop >= -50%;
  pick the highest 2021-23 %/month (tie: shallower drop). Then one look at 2024-26 / since 2025 / stress, and
  the paired comparison with the pick's base without parking.
HINDSIGHT (labelled): the grid's best 2024-26 %/month and each base's best parking variant by 2024-26.
"""
import os
import time

import numpy as np
import pandas as pd

import x10_idle_lib as X

BE = {"BTCUSDT": 0.5, "ETHUSDT": 0.5}
PARKS = {
    "BTC f50": dict(kind="fixed", assets={"BTCUSDT": 1.0}, filt=50),
    "ETH f50": dict(kind="fixed", assets={"ETHUSDT": 1.0}, filt=50),
    "BTC+ETH f50": dict(kind="fixed", assets=BE, filt=50),
    "BTC/ETH f50 renorm": dict(kind="fixed", assets=BE, filt=50, renorm=True),
    "BTC+ETH f50 calm5": dict(kind="fixed", assets=BE, filt=50, calm=0.05),
}
ARMS = [(0, 0.03), (0, 0.05), (0, 0.07), (1, 0.01), (1, 0.02)]
WAITS = [24, 72, 168]
PFS = [1.0, 0.5]
MIN_2022, MAX_DD = 0.0, -0.50


def park_spec(name, pf):
    s = dict(PARKS[name]); s["pf"] = pf
    return s


def run_grid():
    syms, idx, P = X.market(False)
    W = {(p, pf): X.weights_for(P, idx, park_spec(p, pf)) for p in PARKS for pf in PFS}
    rows = []
    t0 = time.time()
    for bname, b in X.BASES.items():
        combos = [None] + [(p, pf, m, x, w) for p in PARKS for pf in PFS for (m, x) in ARMS for w in WAITS]
        for cmb in combos:
            kw = {k: b[k] for k in X.BOT_KEYS}
            if cmb:
                p, pf, m, x, w = cmb
                kw.update(mode=m, armx=x, band=0.05, rebuy_wait=w)
            e, lg, aux, st = X.irun(P, idx, pw=None if cmb is None else W[(cmb[0], cmb[1])], **kw)
            r = X.summarize(e, lg, idx, aux, st if cmb else None)
            if cmb:
                r.update(base=bname, park=p, pf=pf, mode=m, armx=x, rebuy_wait=w)
            else:
                r.update(base=bname, park="none", pf=0.0, mode=-1, armx=np.nan, rebuy_wait=-1)
            rows.append(r)
    print(f"{len(rows)} runs in {time.time() - t0:.0f}s")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(X.OUT, "grid2.csv"), index=False)
    return R


def spec_of(row):
    b = X.BASES[row["base"]]
    if row["park"] == "none":
        return b, None, None
    return b, park_spec(row["park"], float(row["pf"])), dict(mode=int(row["mode"]), armx=float(row["armx"]),
                                                             band=0.05, rebuy_wait=int(row["rebuy_wait"]))


def label(row):
    if row["park"] == "none":
        return f"{row['base']} | no parking"
    return (f"{row['base']} | park {row['park']} x{row['pf']:.0%} | mode {int(row['mode'])} arm {row['armx']:.0%} "
            f"| re-buy wait {int(row['rebuy_wait'])}h")


def full_report(tag, row):
    b, park, pk = spec_of(row)
    r, e, lg, aux = X.evaluate(b, park, pk)
    rs, _, _, _ = X.evaluate(b, park, pk, stress=True)
    print(X.fmt4(f"{tag}: {label(row)}", r, rs))
    y = " ".join(f"{yy} {r[f'y{yy}']*100:+.0f}%" for yy in range(2021, 2027))
    print(f"      years: {y}")
    return r, rs


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    R = run_grid()
    D = R[["base", "park", "pf", "mode", "armx", "rebuy_wait", "d_mo", "d_dd", "y2022"]]   # selection: 2021-23 only
    ok = D[(D.y2022 >= MIN_2022) & (D.d_dd >= MAX_DD)]
    print(f"\nqualify on 2021-23 (2022 >= 0%, drop >= -50%): {len(ok)} of {len(D)}")
    print(ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).head(12).round(4).to_string(index=False))
    if len(ok):
        pick = R.loc[ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).index[0]]
        print("\n==== ONE LOOK: honest pick (2021-23 only) and its base without parking")
        full_report("PICK", pick)
        full_report("ITS BASE, NO PARKING", R[(R.base == pick.base) & (R.park == "none")].iloc[0])

    print("\n==== Does parking add? per base: share of the 150 parking variants that beat the base without parking")
    for bname in X.BASES:
        S = R[(R.base == bname) & (R.park != "none")]
        b0 = R[(R.base == bname) & (R.park == "none")].iloc[0]
        print(f"{bname:18s} base 21-23 {b0.d_mo*100:+.2f}%/mo dd {b0.d_dd*100:.0f}% | 24-26 {b0.t_mo*100:+.2f}%/mo "
              f"dd {b0.t_dd*100:.0f}% | since 2025 {b0.s_mo*100:+.2f}%/mo | parking variants higher: 21-23 "
              f"{(S.d_mo > b0.d_mo).mean()*100:.0f}%, 24-26 {(S.t_mo > b0.t_mo).mean()*100:.0f}%, since 2025 "
              f"{(S.s_mo > b0.s_mo).mean()*100:.0f}%; with drop no deeper (1pt) in 24-26: "
              f"{((S.t_mo > b0.t_mo) & (S.t_dd >= b0.t_dd - 0.01)).mean()*100:.0f}%; median 24-26 "
              f"{S.t_mo.median()*100:+.2f}%/mo, median 24-26 drop {S.t_dd.median()*100:.0f}%")

    print("\n==== Median 2024-26 %/month by factor (all bases pooled, parking runs only)")
    S = R[R.park != "none"]
    for f in ("park", "pf", "mode", "armx", "rebuy_wait"):
        g = S.groupby(f).agg(d_mo=("d_mo", "median"), t_mo=("t_mo", "median"), s_mo=("s_mo", "median"),
                             t_dd=("t_dd", "median"), costs=("pk_costs", "median"))
        print((g * 100).round(2).to_string())

    print("\n==== HINDSIGHT: best of the grid by 2024-26 %/month, and each base's best parking variant by 2024-26")
    full_report("HINDSIGHT BEST OF GRID 2", R.sort_values("t_mo", ascending=False).iloc[0])
    for bname in X.BASES:
        S = R[(R.base == bname) & (R.park != "none")].sort_values("t_mo", ascending=False).iloc[0]
        full_report(f"HINDSIGHT best parking for {bname}", S)
    print(f"\nMAX 2024-26 %/month of any round-2 run: {R.t_mo.max()*100:.2f}%")
