"""m21: how far profit can be pushed with what is already validated. 90% mode and 10% mode, by calendar year, and
with the BNB fee discount (0.075% a side instead of 0.1%). Compounded %/month, worst month, worst drop.
No selection: both modes and their settings are fixed from earlier rounds."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engine as E, m10_velocity_lib as V, m11_fusion_lib as F, m14_judge as J

M90 = F.MODE90
M10 = dict(F.OFFRISK, tiers=[dict(F.TIERA, depth=0.14, tp=0.05, hm=48, cdw=24, cdl=24)])
PER = (("2018-20", True, J.H18, J.H21), ("2021-23", False, V.T21, V.T24), ("2024-26", False, V.T24, None),
       ("since 2025", False, V.T25, None))

def curves(d, **kw):
    out = {}
    for hold in (True, False):
        syms, idx, P = V.market_holdout(False) if hold else V.market(False)
        out[hold] = F.frun(P, idx, d, **kw)[0]
    return out

def cell(e, a, b):
    mult, mo, dd, _ = V.period(e, a, b); m = J.monthly(e, a, b)
    return f"{mo*100:+.1f} / {m.min()*100:+.0f} / {dd*100:.0f}"

print("%/month / worst month / worst drop\n")
print("| setup | " + " | ".join(p[0] for p in PER) + " |\n|---|" + "---|" * len(PER))
res = {}
for fee, flab in ((0.001, "normal fees"), (0.00075, "BNB fee discount")):
    E.FEE = fee
    for name, d in (("90% mode", M90), ("10% mode", M10)):
        c = curves(d); res[(name, flab)] = c
        print(f"| {name}, {flab} | " + " | ".join(cell(c[h], a, b) for _, h, a, b in PER) + " |", flush=True)
    for name, d in (("90% mode", M90), ("10% mode", M10)):
        c = curves(d, pen=0.002, pens=0.002, mtp_now=True, cd_add=1)
        print(f"| {name}, {flab}, bot timing + fills 0.2% worse | " + " | ".join(cell(c[h], a, b) for _, h, a, b in PER) + " |", flush=True)
E.FEE = 0.001
print("\nBy calendar year, compounded %/month (normal fees):\n| year | 90% mode | 10% mode |\n|---|---|---|")
for y in range(2018, 2027):
    h = y < 2021
    a, b = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01")
    row = []
    for name in ("90% mode", "10% mode"):
        e = res[(name, "normal fees")][h]
        if e.index[-1] < b: b2 = None
        else: b2 = b
        mult, mo, dd, months = V.period(e, a, b2)
        m = J.monthly(e, a, b2)
        row.append(f"{mo*100:+.1f} (year x{mult:.2f}, {int((m<0).sum())} losing months, worst {m.min()*100:+.0f}%)")
    print(f"| {y} | " + " | ".join(row) + " |")
