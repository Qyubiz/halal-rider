"""Agent 3 (a): ensembles of 90% mode and 10% mode, confluence, and dynamic sizing. No new guardrails here (the crash
guard 35%/168h of both modes stays on). Each design runs on the 19 coins and on the 22-coin panel (+LUNA/FTT/SRM,
positions sold at their last bar). Periods 2021-23 / 2024-26 / since 2025.

Pick rule, fixed before any result was seen: the architecture with the best 2021-23 (%/month / |max drop|), taking
the WORSE of the two panels for each design. 2024-26 and since-2025 are reported, not used for the pick.

Usage: PB_DATA=... PB_OUT=... python3 ens_designs.py
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_ens/research"))
os.makedirs(OUT, exist_ok=True)
import ens_lib as L  # noqa: E402

M90, M10 = L.MODE90, L.MODE10
D = {"90% mode": ([M90], {}), "10% mode": ([M10], {})}
for w in (0.5, 0.7, 0.3):
    for rb in (0, 1):
        D[f"sleeves 90/10 {int(w*100)}/{int(100-w*100)}{' rebal mo' if rb else ''}"] = ([M90, M10], dict(weights=[w, 1 - w], rb=rb))
for part in (0.5, 0.7):
    for hm in (48, 96):
        for cd in (3, 24):
            D[f"confluence 14% {int(part*100)}%@+2.5 rest@+5 {hm}h cd{cd}"] = (
                [dict(M90, part=part, tp2=0.05, hm=hm, cd=cd)], {})
BASES = {"90%": ([M90], {}), "10%": ([M10], {}), "50/50": ([M90, M10], dict(weights=[0.5, 0.5]))}
for b, (sv, kw) in BASES.items():
    for kf in (0.25, 0.5):
        for km in (0, 1):
            D[f"{b} Kelly {kf} {'prop' if km == 0 else 'textbook'}"] = (sv, dict(kw, kf=kf, kmode=km))
    for tv in (0.03, 0.04, 0.05, 0.06):
        D[f"{b} vol-target {tv:.2f}/day"] = (sv, dict(kw, tv=tv))

rows = []
for name, (sv, kw) in D.items():
    rows += L.run_both(name, sv, **kw)
    print(f"done {name}", flush=True)
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "designs.csv"), index=False)
print(f"\n{len(D)} designs x 2 panels = {len(R)} runs")
print(L.table(R))
sc = R.assign(cal=R["21-23 %mo"] / R["21-23 DD"].abs()).groupby("design", sort=False).cal.min()
print("\n2021-23 score (%/mo / |drop|, worse panel), top 12:")
print(sc.sort_values(ascending=False).head(12).to_string(float_format=lambda v: f"{v:.3f}"))
print(f"\nPICK (2021-23 only): {sc.idxmax()}")
