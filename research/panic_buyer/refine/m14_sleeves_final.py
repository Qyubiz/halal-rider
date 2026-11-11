"""m14 sleeves final look: the pick from the pre-written rule (best robust score among bar1+neighbour passers). One look."""
import json, os
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V, m14_sleeves_lib as L, m14_sleeves_grid as G
g = pd.read_csv(os.path.join(F.OUT, "grid_flags.csv"))
b = g[g.final].sort_values("robust", ascending=False).iloc[0]
pk = dict(N=int(b.N), dc=float(b.dc), ds=float(b.ds), ts=float(b.ts), cs=int(b.cs), rb=int(b.rb))
print("PICK", pk, "score", b.robust)
sl = L.sleeve_params(pk["N"], pk["dc"], pk["ds"], pk["ts"], pk["cs"]); print(sl)
json.dump(dict(pk, sleeves=sl), open(os.path.join(F.OUT, "pick_params.json"), "w"))
MK = {}
def mk(panel, dead):
    if (panel, dead) not in MK:
        MK[(panel, dead)] = V.market_holdout(dead) if panel == "hold" else V.market(dead)
    return MK[(panel, dead)]
MODES = {"n": ({}, False), "w": (dict(pen=F.PEN, pens=F.PEN), False),
         "st": (dict(pen=F.PEN, pens=F.PEN, mtp_now=True, cd_add=1, liquidate=True), True)}
def look(sls, rb, panel, mode, extra={}):
    kw, dead = MODES[mode]; kw = dict(kw, **extra)
    _, idx, P = mk(panel, dead)
    e, lg, inv = L.run(P, idx, f"{panel}{dead}", sls, rb, kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if panel == "hold" else V.MAIN), e
def base(panel, mode):
    return look([dict(depth=.14, tp=.025, cd=3)], 0, panel, mode)[0]
def row(r, t): return f"{r[t+'_wr']*100:5.1f} / {r[t+'_mo']*100:+6.2f} / {r[t+'_dd']*100:6.1f} / {r[t+'_tpm']:4.1f}"
print("columns: win% / %/mo / worst drop / trades per month")
for nm, panel, mode, tags in [("normal main", "main", "n", "dts"), ("normal hold", "hold", "n", "h"), ("worse main", "main", "w", "dts"),
                              ("worse hold", "hold", "w", "h"), ("stress main", "main", "st", "dt")]:
    r, e = look(sl, pk["rb"], panel, mode); rb_ = base(panel, mode)
    for t in tags: print(f"{nm:12s} {t}  pick {row(r,t)} | 90% {row(rb_,t)}")
    if nm == "normal main": e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_main.csv"))
    if nm == "normal hold": e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_hold.csv"))
    if nm == "stress main": print("stressed 2021-23 mult", r["d_mult"], "(fail if <=0.05)")
def items(sls, rb, extra={}):
    n, _ = look(sls, rb, "main", "n", extra); h, _ = look(sls, rb, "hold", "n", extra)
    w, _ = look(sls, rb, "main", "w", extra); wh, _ = look(sls, rb, "hold", "w", extra)
    return (n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35,
            h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663,
            w["t_mo"] > .0531 and wh["h_mo"] > .0741 and w["t_wr"] >= .92 and wh["h_wr"] >= .92)
print("pick items 1-3:", items(sl, pk["rb"]))
res = []
for j, k in enumerate(G.NM):
    for dl in (-1, 1):
        ix = G.AX[k].index(pk[k]) + dl
        if 0 <= ix < len(G.AX[k]):
            q = dict(pk); q[k] = G.AX[k][ix]
            res.append(items(L.sleeve_params(q["N"], q["dc"], q["ds"], q["ts"], q["cs"]), q["rb"]))
vr = [items(sl, pk["rb"], kw) for kw in (dict(cd_add=-1), dict(cd_add=1), dict(mtp_now=True, cd_add=1))]
allr = res + vr
print("grid neighbours passing 1-3:", sum(all(r) for r in res), "of", len(res), res)
print("nearby versions passing 1-3:", sum(all(r) for r in vr), "of", len(vr), vr)
print("combined share:", sum(all(r) for r in allr), "/", len(allr))
