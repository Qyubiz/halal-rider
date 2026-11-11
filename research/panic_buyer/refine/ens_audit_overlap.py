"""Ensemble audit, agent 1 (supplement): how much do 90% mode and 10% mode overlap? Same entry trigger (14% under the
24h high close, 1 coin x 100%, crash guard 35%); they differ in take-profit, time stop and cooldown. Reports the share of
entries on the same coin in the same hour, and the correlation of monthly returns, 2021-01 .. 2026-10 (19 coins).

Usage: PB_DATA=... PB_OUT=... python3 ens_audit_overlap.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_ens/audit"))
os.makedirs(OUT, exist_ok=True)
os.environ["PB_OUT"] = OUT
import w90_wr90_lib as W  # noqa: E402
from ens_audit_modes import MODES  # noqa: E402

syms, idx, P = W.market(False)
res = {}
for m in ("Normal", "90% mode", "10% mode"):
    kw = dict(MODES[m])
    e, lg = W.wrun(P, idx, **kw)
    res[m] = (e, lg)
a, b = res["90% mode"][1], res["10% mode"][1]
ka = set(zip(a.eb.astype(int), a.k.astype(int)))
kb = set(zip(b.eb.astype(int), b.k.astype(int)))
both = ka & kb
print(f"entries: 90% mode {len(ka)}, 10% mode {len(kb)}, same coin+hour {len(both)} "
      f"({len(both)/len(kb)*100:.1f}% of 10% mode entries, {len(both)/len(ka)*100:.1f}% of 90% mode entries)")
mr = pd.DataFrame({m: e.resample("ME").last().pct_change() for m, (e, _) in res.items()}).dropna()
print("monthly-return correlation 2021-02..2026-10:")
print(mr.corr().round(3).to_string())
for p, a0 in (("2024-26", "2024-01-01"), ("since 2025", "2025-01-01")):
    print(f"correlation {p}:"); print(mr[mr.index >= a0].corr().round(3).to_string())
neg = mr[(mr["90% mode"] < 0) & (mr["10% mode"] < 0)]
print(f"months both 90% and 10% mode negative: {len(neg)} of {len(mr)}; months 90% mode <0: {(mr['90% mode']<0).sum()}, "
      f"10% mode <0: {(mr['10% mode']<0).sum()}, Normal <0: {(mr['Normal']<0).sum()}")
print("worst months:"); print((mr.min() * 100).round(2).to_string())
mr.to_csv(os.path.join(OUT, "monthly_returns_modes.csv"))
