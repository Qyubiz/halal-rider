"""m14 bars4h grid. RULE (written before any run; 2021-23 data only):
 designs: bar B in {2,4,8,12} h x depth {12,14,16,18}% x tp {2,2.5,3,4}% x time stop {96,168} h = 128 designs.
   trigger = highest bar close of the last 24h (24/B bars) x (1-depth); bids re-placed once per bar; cooldown 1 bar;
   crash guard 35% vs highest close of 168h on bar closes.
 bar 1: all 5 versions (n,cm,cp,bt,ls) win rate>=92.5% and drop>-64.4%; worse fills wr>=92.5%, drop>-64.4%;
        n_mo>13.17 and w_mo>9.79.
 score: robust = mean %/month over the 5 versions. Neighbours = one grid step in B, depth, tp or hm; median of
        neighbours' normal 2021-23 wr, mo, dd must pass bar 1 (wr>=92.5, dd>-64.4, mo>13.17).
 pick:  highest robust among those passing bar 1 and neighbours; none if none.
"""
import json, os, itertools
import numpy as np, pandas as pd
import m14_bars4h_lib as L
WR, DD, MON, MOW = 0.925, -0.644, 0.1317, 0.0979
BS, DEPS, TPS, HMS = [2, 4, 8, 12], [.12, .14, .16, .18], [.02, .025, .03, .04], [96, 168]
AX = (BS, DEPS, TPS, HMS)
def dz(p): return dict(Wh=24, depth=p[1], tp=p[2], hm=p[3], cd=1, guard=0.35)
cache = {}
def score(p):
    if p not in cache: cache[p] = L.design_scores(dz(p), p[0])
    return cache[p]
def ok(s):
    return all(s[t + "_wr"] >= WR and s[t + "_dd"] > DD for t in ("n", "cm", "cp", "bt", "ls", "w")) and s["n_mo"] > MON and s["w_mo"] > MOW
def nbrs(p):
    out = []
    for a, vals in enumerate(AX):
        i = vals.index(p[a])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals): q = list(p); q[a] = vals[j]; out.append(tuple(q))
    return out
if __name__ == "__main__":
    rows = []
    for p in itertools.product(*AX):
        s = score(p); rows.append(dict(B=p[0], dep=p[1], tp=p[2], hm=p[3], **s))
    df = pd.DataFrame(rows); df["bar1"] = df.apply(ok, axis=1)
    df.to_csv(os.path.join(L.OUT, "grid.csv"), index=False)
    print("designs", len(df), "bar1 pass", int(df.bar1.sum()))
    cols = ["B", "dep", "tp", "hm", "n_wr", "n_mo", "n_dd", "w_wr", "w_mo", "w_dd", "robust", "rmin", "n_tpm", "bar1"]
    print(df.sort_values("robust", ascending=False).head(15)[cols].round(4).to_string())
    print(df.groupby("B")[["n_wr", "n_mo", "n_dd", "robust"]].mean().round(4))
    pick = None
    for _, r in df[df.bar1].sort_values("robust", ascending=False).iterrows():
        p = (int(r.B), r.dep, r.tp, int(r.hm)); nb = [score(q) for q in nbrs(p)]
        med = lambda k: float(np.median([x[k] for x in nb]))
        npass = med("n_wr") >= WR and med("n_dd") > DD and med("n_mo") > MON
        print("cand", p, "robust %.4f" % r.robust, "nbr med wr %.3f mo %.4f dd %.3f" % (med("n_wr"), med("n_mo"), med("n_dd")), npass)
        if npass: pick = p; break
    print("PICK", pick)
    if pick: json.dump(dict(B=pick[0], depth=pick[1], tp=pick[2], hm=pick[3]), open(os.path.join(L.OUT, "pick_params.json"), "w"))
