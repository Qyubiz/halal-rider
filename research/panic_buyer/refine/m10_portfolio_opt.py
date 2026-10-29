"""Step 2 of the portfolio angle: sleeve weights chosen on 2021-23 ONLY, then every period once.

Sleeves: DIP = Panic Buyer 90% mode (ens_lib.MODE90, crash guard 35%); MOM = rider.py simplified on 5-minute candles
(variant M1, the best of the three in m10_portfolio_sleeves.py, although none made money in 2021-23); TREND = 50/50
BTC + ETH, each held while its daily close > its 200-day SMA (m10_portfolio_sleeves.py pick); CASH. One account,
monthly rebalance rb=2 (m10_portfolio_lib docstring).

Rules (fixed before running):
  grid: every (DIP, MOM, TREND, CASH) weight combination in 10% steps summing to 100% (286 designs), 2021-23 only.
  MAIN pick: the best 2021-23 CAGR / |max drop| among designs whose 2021-23 share of positive months is at least
             that of 90% mode alone (fallback: the highest share of positive months).
  ALT pick:  the highest 2021-23 share of positive months (ties: higher %/month).
Then, for 90% mode alone, each sleeve alone and both picks: 2018-20 holdout, 2021-23, 2024-26, since 2025, with normal
fills and with every buy and sell 0.2% worse (all sleeves). Holdout caveat: the data start 2017-12-01, so the 200-day
SMA is unknown until mid-2018; the default treats unknown as "not held" (cash) and a second row treats it as "held".
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 m10_portfolio_opt.py
"""
import itertools
import os

import numpy as np
import pandas as pd

import m10_portfolio_lib as M

MOM = dict(vmode=1, vm=3.0)
TREND = dict(assets=("BTCUSDT", "ETHUSDT"), aw=(0.5, 0.5), rule="sma", n=200)
KW = dict(mom=MOM, trend=TREND)
_, a21, b21 = M.PERIODS[1]
syms, idx, P = M.market(False)


def d_row(w):
    e, sq, lg = M.prun(False, list(w), **KW)
    r = M.stats(e, lg, idx, a21, b21)
    r["calmar"] = ((r["mult"] ** (1 / 3) - 1) / max(1e-9, -r["dd"] / 100)) if r["mult"] > 0 else -1.0
    r["w"] = tuple(w)
    return r


grid = [np.array(c) / 10 for c in itertools.product(range(11), repeat=4) if sum(c) == 10]
rows = [d_row(w) for w in grid]
G = pd.DataFrame(rows)
G[["wDIP", "wMOM", "wTREND", "wCASH"]] = pd.DataFrame(G.w.tolist())
G.drop(columns="w").to_csv(os.path.join(M.OUT, "grid_2021_23.csv"), index=False)
base = G[(G.wDIP == 1.0)].iloc[0]
ok = G[G.pos_mo >= base.pos_mo - 1e-9]
main = ok.loc[ok.calmar.idxmax()] if len(ok) else G.loc[G.pos_mo.idxmax()]
alt = G.sort_values(["pos_mo", "pct_mo"], ascending=False).iloc[0]
print(f"grid: {len(G)} designs on 2021-23. 90% mode alone: up-months {base.pos_mo:.1f}%, {base.pct_mo:+.2f}%/mo, "
      f"DD {base.dd:.1f}%, CAGR/DD {base.calmar:.2f}")
print(f"designs with at least as many up-months as 90% mode: {len(ok)}")
print("top 8 by 2021-23 CAGR/|DD| among them:")
print(ok.sort_values("calmar", ascending=False).head(8)[["wDIP", "wMOM", "wTREND", "wCASH", "pos_mo", "pct_mo", "dd",
                                                         "worst_mo", "wr", "calmar"]].to_string(index=False))
print("top 8 by 2021-23 up-months:")
print(G.sort_values(["pos_mo", "pct_mo"], ascending=False).head(8)[["wDIP", "wMOM", "wTREND", "wCASH", "pos_mo",
                                                                    "pct_mo", "dd", "worst_mo", "wr"]].to_string(index=False))
print("best 2021-23 %/month with MOM > 0:",
      G[G.wMOM > 0].sort_values("pct_mo", ascending=False).head(1)[["wDIP", "wMOM", "wTREND", "wCASH", "pct_mo", "dd"]]
      .to_string(index=False, header=False))
wm = [float(x) for x in main[["wDIP", "wMOM", "wTREND", "wCASH"]]]
wa = [float(x) for x in alt[["wDIP", "wMOM", "wTREND", "wCASH"]]]
print(f"MAIN pick: DIP/MOM/TREND/CASH = {wm}")
print(f"ALT pick:  DIP/MOM/TREND/CASH = {wa}")

DES = {"90% mode alone": [1, 0, 0, 0], "MOM alone": [0, 1, 0, 0], "TREND alone": [0, 0, 1, 0],
       f"MAIN {wm}": wm, f"ALT {wa}": wa}
out = []
for name, w in DES.items():
    for ftag, extra in (("normal", {}), ("fills 0.2% worse", M.STRESS)):
        print(f"\n{name} [{ftag}]")
        for r in M.all_periods(w, name, **KW, **extra):
            r["fills"] = ftag
            out.append(r)
            print("  " + M.fmt_row(r))
# sleeve trade win rates inside the MAIN pick (2024-26, normal fills)
e, sq, lg = M.prun(False, wm, **KW)
for nm, s in (("DIP", 0), ("MOM", 1), ("TREND", 2)):
    r = M.stats(e, lg, idx, pd.Timestamp("2024-01-01"), None, sleeve=s)
    print(f"MAIN, 2024-26 trades of the {nm} sleeve: {r['trades']}, win rate {r['wr']:.1f}%")
# holdout with the 200-day SMA treated as 'held' while unknown, and no-rebalance sensitivity
hidx = M.market(True)[1]
for tag, kw in (("holdout, SMA200 unknown = held", dict(trend=dict(TREND, unknown_on=True))),
                ("holdout, no rebalance (rb=0)", dict(trend=TREND, rb=0))):
    e, sq, lg = M.prun(True, wm, mom=MOM, **kw)
    r = M.stats(e, lg, hidx, M.PERIODS[0][1], M.PERIODS[0][2]); r["period"] = "2018-20"
    print(f"MAIN {tag}: " + M.fmt_row(r))
e, sq, lg = M.prun(False, wm, mom=MOM, trend=TREND, rb=0)
for name, a, b in M.PERIODS[1:]:
    r = M.stats(e, lg, idx, a, b); r["period"] = name
    print(f"MAIN no rebalance (rb=0, one start in 2021): " + M.fmt_row(r))
# monthly-return correlation of the sleeves alone, 2021-23
cur = {n: M.prun(False, w, **KW)[0] for n, w in list(DES.items())[:3]}
print("\n2021-23 monthly-return correlation, sleeves alone:")
print(pd.DataFrame({n: M.months(v, a21, b21) for n, v in cur.items()}).corr().round(2).to_string())
pd.DataFrame(out).to_csv(os.path.join(M.OUT, "periods.csv"), index=False)
print(f"\nsaved {M.OUT}/grid_2021_23.csv, periods.csv")
