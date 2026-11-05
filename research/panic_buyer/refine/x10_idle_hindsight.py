"""HINDSIGHT analyses for the idle-cash lever (none of this is an honest pick; everything here has seen 2024-26).

A. Ceiling: park in each single coin of the 19 (no filter / 50-day filter) behind the two strongest bases, with
   the round-2 hindsight mechanics (mode 0, arm 7%, re-buy wait 24h) and the round-2 honest mechanics (mode 1,
   arm 2%, wait 168h). Shows how far the best parking coin, chosen with hindsight, could take the combination.
B. Plateau around the round-2 hindsight best (1x100% +3%, ETH f50, mode 0, arm 7%, wait 24h): one setting moved
   at a time. A real effect should not depend on one exact setting.
C. Cost and survivorship sensitivity, split: fills 0.1% / 0.2% worse (all fills incl. parking), LUNA/FTT/SRM
   added, both; for the honest picks, the hindsight best and their bases.
D. Parking engines alone, per year (ETH f50, BTC f50, BTC+ETH f50), to show where the parking return comes from.
"""
import os

import numpy as np
import pandas as pd

import x10_idle_grid2 as G2
import x10_idle_lib as X

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
syms, idx, P = X.market(False)
B1, B2 = X.BASES["1x100% +3%"], X.BASES["2x50% +5%"]
MECHS = {"m0 a7% w24": dict(mode=0, armx=0.07, band=0.05, rebuy_wait=24),
         "m1 a2% w168": dict(mode=1, armx=0.02, band=0.05, rebuy_wait=168)}


def run(b, park, pk, P=P, idx=idx, **extra):
    kw = {k: b[k] for k in X.BOT_KEYS}
    kw.update(pk or {}); kw.update(extra)
    pw = X.weights_for(P, idx, park)
    e, lg, aux, st = X.irun(P, idx, pw=pw, **kw)
    return X.summarize(e, lg, idx, aux, st if pw is not None else None)


print("######## A. HINDSIGHT ceiling: single-coin parking")
rows = []
for bn, b in (("1x100% +3%", B1), ("2x50% +5%", B2)):
    for s in X.E.H19:
        for f in (0, 50):
            for mn, pk in MECHS.items():
                r = run(b, dict(kind="fixed", assets={s: 1.0}, filt=f), pk)
                r.update(base=bn, coin=s, filt=f, mech=mn)
                rows.append(r)
A = pd.DataFrame(rows)
A.to_csv(os.path.join(X.OUT, "hindsight_single_coin.csv"), index=False)
show = ["base", "coin", "filt", "mech", "d_mo", "d_dd", "y2022", "t_mo", "t_dd", "s_mo", "s_dd", "t_park"]
print(A.sort_values("t_mo", ascending=False)[show].head(15).round(4).to_string(index=False))
print(f"single-coin parking runs: {len(A)}; reaching 10%/month in 2024-26: {(A.t_mo >= 0.10).sum()}; "
      f"max {A.t_mo.max()*100:.2f}%/mo")
best = A.sort_values("t_mo", ascending=False).iloc[0]
park_best = dict(kind="fixed", assets={best.coin: 1.0}, filt=int(best.filt))
bb = B1 if best.base == "1x100% +3%" else B2
r = run(bb, park_best, MECHS[best.mech])
syms_d, idx_d, P_d = X.market(True)
rs = run(bb, park_best, MECHS[best.mech], P=P_d, idx=idx_d, pen=0.002, pens=0.002)
print(X.fmt4(f"HINDSIGHT single-coin best: {best.base} | park {best.coin} filt {best.filt} | {best.mech}", r, rs))

print("\n######## B. Plateau around the round-2 hindsight best (1x100% +3%, ETH f50, m0 a7% w24); HINDSIGHT")
base_pk = dict(mode=0, armx=0.07, band=0.05, rebuy_wait=24)
eth = lambda f=50: dict(kind="fixed", assets={"ETHUSDT": 1.0}, filt=f)  # noqa: E731
var = [("as is", eth(), base_pk)]
for f in (30, 40, 60, 70):
    var.append((f"filter {f}d", eth(f), base_pk))
for a in (0.05, 0.06, 0.08, 0.10):
    var.append((f"arm {a:.0%}", eth(), {**base_pk, "armx": a}))
for w in (0, 12, 48, 72, 168):
    var.append((f"wait {w}h", eth(), {**base_pk, "rebuy_wait": w}))
for bd in (0.02, 0.10):
    var.append((f"band {bd:.0%}", eth(), {**base_pk, "band": bd}))
var.append(("mode 1 arm 2%", eth(), {**base_pk, "mode": 1, "armx": 0.02}))
var.append(("parking 50%", {**eth(), "pf": 0.5}, base_pk))
for name, bdict in (("base 1x100% +2%", dict(B1, tp=0.02)), ("base 1x100% +5%", dict(B1, tp=0.05)),
                    ("base depth 10%", dict(B1, depth=0.10)), ("base depth 15%", dict(B1, depth=0.15))):
    var.append((name, eth(), base_pk, bdict))
prow = []
for v in var:
    name, park, pk = v[:3]
    b = v[3] if len(v) > 3 else B1
    r = run(b, park, pk)
    prow.append(dict(variant=name, d_mo=r["d_mo"], d_dd=r["d_dd"], y2022=r["y2022"], t_mo=r["t_mo"], t_dd=r["t_dd"],
                     s_mo=r["s_mo"], s_dd=r["s_dd"], costs=r.get("pk_costs", np.nan)))
PL = pd.DataFrame(prow)
PL.to_csv(os.path.join(X.OUT, "hindsight_plateau.csv"), index=False)
print(PL.round(4).to_string(index=False))
print(f"neighbours >= 10%/mo in 2024-26: {(PL.t_mo.iloc[1:] >= 0.10).sum()} of {len(PL) - 1}; >= 9%: "
      f"{(PL.t_mo.iloc[1:] >= 0.09).sum()}; median {PL.t_mo.iloc[1:].median()*100:.2f}%/mo")

print("\n######## C. Cost and survivorship sensitivity (2024-26 %/mo, drop | since 2025 | 2021-23)")
CASES = {
    "R1 honest pick: 2x50% +5% BTC f50 m0 a5% w24": (B2, G2.park_spec("BTC f50", 1.0),
                                                     dict(mode=0, armx=0.05, band=0.05, rebuy_wait=24)),
    "R2 honest pick: 2x50% +5% BTC f50 m1 a2% w168": (B2, G2.park_spec("BTC f50", 1.0),
                                                      dict(mode=1, armx=0.02, band=0.05, rebuy_wait=168)),
    "base 2x50% +5% (no parking)": (B2, None, None),
    "R2 HINDSIGHT best: 1x100% +3% ETH f50 m0 a7% w24": (B1, eth(), base_pk),
    "base 1x100% +3% (no parking)": (B1, None, None),
    "live 8x20% (no parking)": (X.BASES["live 8x20% +1.5%"], None, None),
}
crow = []
for name, (b, park, pk) in CASES.items():
    for lab, dead, pen in (("normal", False, 0.0), ("fills 0.1% worse", False, 0.001), ("fills 0.2% worse", False, 0.002),
                           ("+LUNA/FTT/SRM", True, 0.0), ("both (report 4)", True, 0.002)):
        _, ix, PP = X.market(dead)
        r = run(b, park, pk, P=PP, idx=ix, pen=pen, pens=pen)
        crow.append(dict(case=name, stress=lab, t_mo=r["t_mo"], t_dd=r["t_dd"], s_mo=r["s_mo"], d_mo=r["d_mo"],
                         d_dd=r["d_dd"], worst_acct=r.get("worst_acct", np.nan)))
C = pd.DataFrame(crow)
C.to_csv(os.path.join(X.OUT, "cost_sensitivity.csv"), index=False)
print(C.round(4).to_string(index=False))

print("\n######## D. Parking engines alone, per calendar year (N=0, fees, band 0)")
for name, park in (("BTC f50", G2.PARKS["BTC f50"]), ("ETH f50", G2.PARKS["ETH f50"]),
                   ("BTC+ETH f50", G2.PARKS["BTC+ETH f50"]), ("ETH buy and hold", dict(kind="fixed", assets={"ETHUSDT": 1.0}))):
    pw = X.weights_for(P, idx, park)
    e, lg, aux, st = X.irun(P, idx, pw=pw, N=0, band=0.0)
    r = X.summarize(e, lg, idx, aux, st)
    y = " ".join(f"{yy} {r[f'y{yy}']*100:+5.0f}%" for yy in range(2021, 2027))
    print(f"{name:18s} {y} | 2021-23 {r['d_mo']*100:+.2f}%/mo dd {r['d_dd']*100:.0f}% | 2024-26 {r['t_mo']*100:+.2f}%/mo "
          f"dd {r['t_dd']*100:.0f}% | since 2025 {r['s_mo']*100:+.2f}%/mo")
