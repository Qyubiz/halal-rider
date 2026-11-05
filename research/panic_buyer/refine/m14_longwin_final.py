"""m14 longwin final look: pick from m14_longwin_grid.py (pick_params.json). One look, no re-picking."""
import json, os
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V
import m14_longwin_grid as G
pk = json.load(open(os.path.join(F.OUT, "pick_params.json"))); print("PICK", pk)
D = G.design(pk["mode"], pk["W"], pk["dep"], pk["tp"], pk["hm"])
row = lambda r, t: f"{r[t+'_wr']*100:5.1f} / {r[t+'_mo']*100:+6.2f} / {r[t+'_dd']*100:6.1f} / {r[t+'_tpm']:4.1f}"
R = {}
for nm, kw in [("normal", {}), ("worse", dict(worse=True)), ("stress", dict(stress=True))]:
    for ho in (False, True):
        if nm == "stress" and ho: continue
        for lab, d in (("pick", D), ("90%", F.MODE90)):
            r, e, lg = F.evaluate(d, holdout=ho, **kw); R[(nm, ho, lab)] = r
            if lab == "pick" and nm == "normal":
                e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_hold.csv" if ho else "pick_eq_main.csv"))
for k, r in R.items():
    tags = "h" if k[1] else "dts"
    for t in tags: print(f"{k[0]:7s} {'hold' if k[1] else 'main'} {t} {k[2]:5s} {row(r,t)}")
print("stress 2021-23 mult pick", R[("stress", False, "pick")]["d_mult"], "90%", R[("stress", False, "90%")]["d_mult"])
def items(d):
    n = F.evaluate(d)[0]; h = F.evaluate(d, holdout=True)[0]; w = F.evaluate(d, worse=True)[0]; wh = F.evaluate(d, worse=True, holdout=True)[0]
    return (n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35, h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663,
            w["t_mo"] > .0531 and wh["h_mo"] > .0741 and w["t_wr"] >= .92 and wh["h_wr"] >= .92)
print("pick items 1-3", items(D))
s = R[("stress", False, "pick")]; print("item4: stress 2024-26 mo", s["t_mo"], ">0.0549:", s["t_mo"] > .0549, "d_mult>0.05:", s["d_mult"] > .05)
p = (pk["mode"], pk["W"], pk["dep"], pk["tp"], pk["hm"]); res = []
for ai, vals in ((1, G.WS), (2, G.DEPS), (3, G.TPS), (4, G.HMS)):
    i = vals.index(p[ai])
    for j in (i - 1, i + 1):
        if 0 <= j < len(vals):
            q = list(p); q[ai] = vals[j]; res.append((tuple(q), items(G.design(*q))))
for q, r in res: print("nbr", q, r)
print("neighbours passing 1-3:", sum(all(r) for _, r in res), "of", len(res))
vr = []
for kw in (dict(cd_add=-1), dict(cd_add=1), dict(mtp_now=True, cd_add=1)):
    out = []
    for ho in (False, True):
        _, idx, P = V.market_holdout(False) if ho else V.market(False)
        e, lg, inv = F.frun(P, idx, D, **kw); r = V.summarize(e, lg, inv, idx, V.HOLD if ho else V.MAIN); out.append(r)
    n, h = out
    vr.append((n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35, h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663))
print("versions item1/item2:", vr)
# crash guard blocking: share of hours (coin-hours with a bid price within reach) blocked, and effect of guard off
for ho in (False, True):
    _, idx, P = V.market_holdout(False) if ho else V.market(False)
    g = F.guard(P, .35); Rw = F.ref(P, pk["W"]); o = P["o"].values
    near = np.isfinite(Rw) & np.isfinite(o) & (o <= Rw * (1 - pk["dep"]) * 1.0)
    print("hold" if ho else "main", "guard blocks", f"{(near & (g == 0)).sum()} of {near.sum()} coin-hours at/below the bid ({(near&(g==0)).sum()/max(near.sum(),1)*100:.1f}%)")
    for gg in (.35, 0.0):
        d = dict(D, guard=gg); e, lg, inv = F.frun(P, idx, d); r = V.summarize(e, lg, inv, idx, V.HOLD if ho else V.MAIN)
        t = "h" if ho else "d"; print("  guard", gg, row(r, t))
