"""m14 sleeves grid. RULE (written before any run), 2021-23 only:
 design = (N in 3,5; depth centre 13/14/15%; depth step .5/1 pt; tp step 0/.5 pt around 2.5%; cooldown spread 0/1h;
 rebalance never/monthly). Sleeve j: depth = centre + o_j*step, tp = 2.5% + o_(j+1)*step, cooldown 3h + spread.
 Pass bar 1: for ALL 5 versions (n, cm, cp, bt, ls) win rate >= 92.5% and drop > -64.4%; normal %/month > 13.17;
 worse fills (w) wr>=92.5, drop>-64.4, %/month > 9.79. Score = mean 2021-23 %/month over the 5 versions. Neighbours =
 one step on one axis; their median n_mo, n_wr, n_dd must pass bar 1. PICK = best score among passers; none -> none.
 Design count 96."""
import itertools, os, sys
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V, m14_sleeves_lib as L

AX = dict(N=[3, 5], dc=[.13, .14, .15], ds=[.005, .01], ts=[0, .005], cs=[0, 1], rb=[0, 1])
NM = list(AX)
if __name__ == "__main__":
    syms, idx, P = V.market(False)
    # engine proof: one sleeve (90% mode) vs fk base
    b = L.scores([dict(depth=.14, tp=.025, cd=3)], 0, P, idx)
    print("90% mode 2021-23 (n):", b["n_wr"], b["n_mo"], b["n_dd"], "(expect 93.8/13.17/-64.4)")
    rows = []
    for ix in itertools.product(*[range(len(AX[k])) for k in NM]):
        p = {k: AX[k][i] for k, i in zip(NM, ix)}
        sl = L.sleeve_params(p["N"], p["dc"], p["ds"], p["ts"], p["cs"])
        s = L.scores(sl, p["rb"], P, idx); s.update(p); s["ix"] = ix
        rows.append(s)
    g = pd.DataFrame(rows); g.to_csv(os.path.join(F.OUT, "grid.csv"), index=False)
    V_ = ["n", "cm", "cp", "bt", "ls"]
    g["p1"] = g.apply(lambda s: all(s[t + "_wr"] >= .925 and s[t + "_dd"] > -.644 for t in V_) and s.n_mo > .1317
                      and s.w_wr >= .925 and s.w_dd > -.644 and s.w_mo > .0979, axis=1)
    key = {tuple(r.ix): i for i, r in g.iterrows()}
    nm = []
    for i, r in g.iterrows():
        nb = []
        for j in range(len(NM)):
            for dl in (-1, 1):
                k = list(r.ix); k[j] += dl
                if tuple(k) in key: nb.append(key[tuple(k)])
        x = g.loc[nb]
        nm.append(bool(np.median(x.n_wr) >= .925 and np.median(x.n_dd) > -.644 and np.median(x.n_mo) > .1317))
    g["nbok"] = nm
    g["final"] = g.p1 & g.nbok
    g.drop(columns="ix").to_csv(os.path.join(F.OUT, "grid_flags.csv"), index=False)
    print("pass bar1:", g.p1.sum(), "pass nb:", g.final.sum(), "of", len(g))
    c = ["N", "dc", "ds", "ts", "cs", "rb", "n_wr", "n_mo", "n_dd", "w_mo", "robust", "p1", "nbok"]
    print(g.sort_values("robust", ascending=False)[c].head(15).to_string())
    print(g[["n_wr", "n_mo", "n_dd", "robust"]].describe().to_string())
