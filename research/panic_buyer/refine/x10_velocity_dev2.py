"""Round-2 development for the velocity lever, 2021-23 ONLY (prints and saves no 2024-26 number). Round 2 was started
after the round-1 grid's 2024-26 result (no design reached 10%/month), so the choice to look further is hindsight-
informed; the mechanics and value ranges below are judged on 2021-23 only. Mechanics: a second, shorter "flash dip"
trigger (look2 / depth2 / tpf), re-entry after a profitable exit (redip / rewin), shorter time stops."""
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_velocity_lib as V  # noqa: E402

A, hidx, Ph, SLIP = V.load()
nd = int(hidx.searchsorted(V.T24))
Ad = {f: v[: nd * 60] for f, v in A.items()}
Pd = {f: v.iloc[:nd] for f, v in Ph.items()}
Sd = SLIP[:nd]
hd = hidx[:nd]
rows = []


def go(tag, K=19, **kw):
    t0 = time.time()
    e, lg, inv = V.vrun(Ad, hd, Pd, Sd, K=K, **kw)
    r = V.report(e, lg, hd)
    r = {k: v for k, v in r.items() if k.startswith("d_") or k == "y2022"}
    r.update(tag=tag, K=K, inv=float(inv.mean()), **kw)
    rows.append(r)
    print(f"{tag:64s} {V.fmt_d(r)} inv {inv.mean()*100:4.1f}% ({time.time()-t0:.1f}s)", flush=True)
    return r


PICK1 = dict(N=2, fmax=0.5, cyc=1, rearm=1, cdm=-1, cd=24, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1)
N1T = dict(PICK1, N=1, fmax=1.0)
FIX2 = dict(N=2, fmax=0.5, cyc=1, rearm=1, cdm=-1, cd=24, tp=0.03)
for K in (19, 22):
    go("round-1 pick", K=K, **PICK1)
print("#### A. flash-dip trigger on the round-1 pick (trailing exit for all buys)")
for look2 in (1, 3, 6):
    for depth2 in (0.04, 0.06, 0.08):
        go(f"pick + flash look{look2}h depth{depth2}", look2=look2, depth2=depth2, **PICK1)
print("#### B. flash-dip trigger, fixed exits (24h-dip buys tp 3%, flash buys tpf), N2")
for look2 in (1, 3):
    for depth2 in (0.04, 0.06, 0.08):
        for tpf in (0.01, 0.015, 0.03):
            go(f"N2 tp3 + flash look{look2}h depth{depth2} tpf{tpf}", look2=look2, depth2=depth2, tpf=tpf, **FIX2)
print("#### C. re-entry after a profitable exit (cooldown 0 otherwise only for re-entries: cdm 0)")
for redip in (0.01, 0.02, 0.03):
    for rewin in (60, 240, 1440):
        go(f"pick cdm0 + redip{redip} rewin{rewin}", **{**PICK1, "cdm": 0}, redip=redip, rewin=rewin)
print("#### D. time stop")
for hm in (6, 12, 24, 48, 96):
    go(f"pick hm{hm}", **{**PICK1, "hm": hm})
    go(f"N1 trail hm{hm}", **{**N1T, "hm": hm})
print("#### E. promising combos on 22 coins too")
for look2, depth2 in ((1, 0.06), (3, 0.08), (1, 0.08)):
    for K in (19, 22):
        go(f"pick + flash look{look2}h depth{depth2}", K=K, look2=look2, depth2=depth2, **PICK1)
pd.DataFrame(rows).to_csv(os.path.join(V.OUT, "dev2_2021_23.csv"), index=False)
