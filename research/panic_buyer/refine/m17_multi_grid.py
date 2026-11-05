"""m17 "multi": more trades per month, fully funded. Several dip levels (24h high 10-14%, plus optional faster 6h/12h
high 8-12%), 2-3 positions at once (equal split of cash among new bids; nbid=nslots=P, fs=1/P), tp 2.5-6%, re-arm at
once after a win (cdw=0), 96h stop (48h for the fast tier). 135 designs, scored on 2021-23 ONLY.

RULE (written before any run): eligible = 2021-23 normal AND 0.2%-worse fills worst drop >= -70% (all 5 nearby versions
too, via design_scores n/w only for the pick check), and median over grid neighbours (one ordinal step in one axis) of
robust score >= 13.17 and of normal drop >= -70%. Pick = highest robust (mean %/month over 5 nearby versions) among
eligible; then stressed 2021-23 must not lose >=95% (else next one). No win-rate floor. One look afterwards.
"""
import itertools, json, os, sys
import numpy as np, pandas as pd
import m11_fusion_lib as F

D1 = [0.10, 0.12, 0.14]
FAST = [None, (12, .12), (12, .10), (6, .10), (6, .08)]
TP = [0.025, 0.04, 0.06]
PP = [1, 2, 3]
AX = dict(d1=D1, fast=list(range(5)), tp=TP, P=PP)

def design(d1, fast, tp, P):
    tiers = [dict(ref=24, depth=d1, tp=tp, hm=96, cdw=0, cdl=3, fs=1.0 / P)]
    if FAST[fast]:
        r, dp = FAST[fast]
        tiers.append(dict(ref=r, depth=dp, tp=tp, hm=48, cdw=0, cdl=3, fs=1.0 / P))
    return dict(tiers=tiers, nbid=P, nslots=P)

BASE = dict(tiers=[dict(ref=24, depth=.14, tp=.025, hm=96, cdw=3, cdl=3)])

if __name__ == "__main__":
    OUT = F.OUT
    rows = []
    for d1, fa, tp, P in itertools.product(D1, range(5), TP, PP):
        s = F.design_scores(design(d1, fa, tp, P))
        rows.append(dict(d1=d1, fast=fa, tp=tp, P=P, **s))
        print(rows[-1]["d1"], fa, tp, P, round(s["robust"]*100, 2), round(s["n_dd"]*100, 1), round(s["w_dd"]*100, 1), round(s["n_tpm"], 1), flush=True)
    g = pd.DataFrame(rows)
    key = lambda r: (D1.index(r.d1), r.fast, TP.index(r.tp), PP.index(r.P))
    idx = {key(r): i for i, r in g.iterrows()}
    nm, nd = [], []
    for i, r in g.iterrows():
        k = key(r); ms, ds = [], []
        for a in range(4):
            for dl in (-1, 1):
                q = list(k); q[a] += dl
                if tuple(q) in idx:
                    ms.append(g.robust[idx[tuple(q)]]); ds.append(g.n_dd[idx[tuple(q)]])
        nm.append(np.median(ms)); nd.append(np.median(ds))
    g["nb_robust"], g["nb_dd"] = nm, nd
    g["elig"] = (g.n_dd >= -.70) & (g.w_dd >= -.70) & (g.nb_robust >= .1317) & (g.nb_dd >= -.70)
    g.to_csv(os.path.join(OUT, "grid.csv"), index=False)
    print(g[g.elig].sort_values("robust", ascending=False).head(12).to_string())
    print("eligible", g.elig.sum(), "of", len(g))
