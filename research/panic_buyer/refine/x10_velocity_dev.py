"""Development pass for the velocity lever: learn what each switch does using 2021-23 ONLY (this script prints and saves
no 2024-26 number). 19 live coins, volume-rank slippage. Used to choose the switches and value ranges of the
pre-registered grid in x10_velocity_grid.py."""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_velocity_lib as V  # noqa: E402

A, hidx, Ph, SLIP = V.load()
END = V.T24
nd = int(hidx.searchsorted(END))                 # hours before 2024-01-01: the run never sees later data
Ad = {f: v[: nd * 60] for f, v in A.items()}
Pd = {f: v.iloc[:nd] for f, v in Ph.items()}
Sd = SLIP[:nd]
hd = hidx[:nd]
rows = []


def go(tag, **kw):
    t0 = time.time()
    e, lg, inv = V.vrun(Ad, hd, Pd, Sd, K=19, **kw)
    r = V.report(e, lg, hd)
    r = {k: v for k, v in r.items() if k.startswith("d_") or k == "y2022"}
    r.update(tag=tag, inv=float(inv.mean()), **kw)
    rows.append(r)
    print(f"{tag:58s} {V.fmt_d(r)} inv {inv.mean()*100:4.1f}% ({time.time()-t0:.1f}s)", flush=True)
    return r


C2 = dict(N=2, fmax=0.5)
C1 = dict(N=1, fmax=1.0)
print("#### A. reference: hourly cycle, concentrated sizes, fixed take-profit")
for nm, sz in (("live N8", dict()), ("N3", dict(N=3, fmax=0.33)), ("N2", C2), ("N1", C1)):
    for tp in (0.015, 0.03, 0.05):
        go(f"{nm} tp{tp}", tp=tp, **sz)

print("#### B. cycle length, re-arm, cooldown (N2 tp3%, N1 tp3%)")
for nm, sz in (("N2", C2), ("N1", C1)):
    for cyc in (60, 15, 5, 1):
        for rearm in (0, 1):
            for cdm in (-1, 60, 0):
                go(f"{nm} tp3 cyc{cyc} rearm{rearm} cdm{cdm}", tp=0.03, cyc=cyc, rearm=rearm, cdm=cdm, **sz)

print("#### C. near-trigger only bids (fmax 1: all free cash on a lone dip)")
for N in (1, 2, 3, 4):
    for near in (1e9, 0.05, 0.02, 0.01, 0.005):
        go(f"N{N} fmax1 near{near} tp3 cyc1 rearm1 cdm0", N=N, fmax=1.0, near=near, tp=0.03, cyc=1, rearm=1, cdm=0)

print("#### D. take-profit delay with 1-minute cycles (N2 tp3 cyc1 rearm1 cdm0)")
for d in (0, 1, 5, 15, 30):
    go(f"N2 tp3 cyc1 rearm1 cdm0 delay{d}", tp=0.03, cyc=1, rearm=1, cdm=0, delay=d, **C2)

print("#### E. trailing exits (N2 cyc1 rearm1 cdm0, delay 1)")
for arm in (0.015, 0.03, 0.05):
    for trail in (0.0, 0.005, 0.01, 0.02):
        for cap in (0.0, 0.08):
            go(f"N2 trail arm{arm} trail{trail} cap{cap}", xmode=1, arm=arm, trail=trail, cap=cap, delay=1,
               cyc=1, rearm=1, cdm=0, **C2)

print("#### F. depth and time stop (N2 tp3 cyc1 rearm1 cdm0)")
for depth in (0.08, 0.10, 0.12, 0.15, 0.18):
    for hm in (24, 48, 96):
        go(f"N2 tp3 depth{depth} hm{hm}", depth=depth, hm=hm, tp=0.03, cyc=1, rearm=1, cdm=0, **C2)

pd.DataFrame(rows).to_csv(os.path.join(V.OUT, "dev_2021_23.csv"), index=False)
