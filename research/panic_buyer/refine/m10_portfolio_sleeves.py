"""Step 1 of the portfolio angle: choose each extra sleeve's variant on 2021-23 ONLY (rule fixed before running).

Variants (written before any result was seen):
  MOM   M0 rider rules, no volume stand-in (price only) | M1 range >= 3x 48-candle average | M2 range >= 2x
  TREND T1 BTC > SMA50 | T2 BTC > SMA200 | T3 BTC SMA50 > SMA200 | T4-T6 same, 50/50 BTC + ETH (each on its own signal)
Pick per sleeve: the best 2021-23 CAGR / |max drop| (each sleeve alone, all money in it), among variants that made money
in 2021-23. Only the picks are then shown on the other periods. Also prints the 2021-23 monthly-return correlation of
the sleeves (90% mode, MOM pick, TREND pick).
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 m10_portfolio_sleeves.py
"""
import json
import os

import numpy as np
import pandas as pd

import m10_portfolio_lib as M

MOMV = {"M0 price only": dict(vmode=0), "M1 range>=3x": dict(vmode=1, vm=3.0), "M2 range>=2x": dict(vmode=1, vm=2.0)}
TRV = {}
for a, (assets, aw) in {"BTC": (("BTCUSDT", "ETHUSDT"), (1.0, 0.0)),
                        "BTC+ETH": (("BTCUSDT", "ETHUSDT"), (0.5, 0.5))}.items():
    for r, (rule, n) in {"close>SMA50": ("sma", 50), "close>SMA200": ("sma", 200), "SMA50>SMA200": ("cross", 200)}.items():
        TRV[f"{a} {r}"] = dict(assets=assets, aw=aw, rule=rule, n=n)

_, a21, b21 = M.PERIODS[1]
syms, idx, P = M.market(False)


def design_row(w, **kw):
    e, sq, lg = M.prun(False, w, **kw)
    r = M.stats(e, lg, idx, a21, b21)
    yrs = 3.0
    r["calmar"] = ((r["mult"] ** (1 / yrs) - 1) / max(1e-9, -r["dd"] / 100)) if r["mult"] > 0 else -1
    return r, e


picks = {}
curves = {}
for sleeve, V, w in (("MOM", MOMV, [0, 1, 0, 0]), ("TREND", TRV, [0, 0, 1, 0])):
    print(f"\n==== {sleeve} variants, all money in the sleeve, 2021-23 only")
    rows = []
    for name, v in V.items():
        kw = {"mom": v} if sleeve == "MOM" else {"trend": v}
        r, e = design_row(w, **kw)
        r["name"] = name
        rows.append(r)
        print(f"  {name:22s} up-months {r['pos_mo']:5.1f}%  WR {r['wr']:5.1f}% ({r['trades']:4d})  {r['pct_mo']:+6.2f}%/mo  "
              f"maxDD {r['dd']:6.1f}%  worst mo {r['worst_mo']:+6.1f}%  CAGR/DD {r['calmar']:5.2f}")
    ok = [r for r in rows if r["mult"] > 1]
    pk = max(ok, key=lambda r: r["calmar"]) if ok else None
    picks[sleeve] = pk["name"] if pk else None
    print(f"  PICK ({sleeve}, 2021-23 CAGR/|DD|): {picks[sleeve]}")

print("\n==== picks on every period (all money in the sleeve), normal fills / fills 0.2% worse")
for label, w, kw in (("90% mode alone", [1, 0, 0, 0], {}),
                     (f"MOM {picks['MOM']}", [0, 1, 0, 0], {"mom": MOMV[picks["MOM"]]}),
                     (f"TREND {picks['TREND']}", [0, 0, 1, 0], {"trend": TRV[picks["TREND"]]})):
    for ftag, extra in (("normal", {}), ("fills 0.2% worse", M.STRESS)):
        print(f"  {label} [{ftag}]")
        for r in M.all_periods(w, label, **kw, **extra):
            print("    " + M.fmt_row(r))
    e, sq, lg = M.prun(False, w, **kw)
    curves[label] = e

mr = pd.DataFrame({k: M.months(v, a21, b21) for k, v in curves.items()})
print("\n2021-23 monthly-return correlation of the sleeves (each alone):")
print(mr.corr().round(2).to_string())
both = pd.DataFrame({k: M.months(v, pd.Timestamp("2024-01-01"), None) for k, v in curves.items()})
print("2024-26 (information only, not used for any choice):")
print(both.corr().round(2).to_string())
with open(os.path.join(M.OUT, "sleeve_picks.json"), "w") as f:
    json.dump({"MOM": [picks["MOM"], MOMV[picks["MOM"]]], "TREND": [picks["TREND"], TRV[picks["TREND"]]]}, f)
print(f"\nsaved {os.path.join(M.OUT, 'sleeve_picks.json')}")
