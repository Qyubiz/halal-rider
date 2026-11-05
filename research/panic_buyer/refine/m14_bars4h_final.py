"""m14 bars4h final look: pick from m14_bars4h_grid.py (pick_params.json). One look, no re-picking."""
import json, os
import numpy as np, pandas as pd
import m11_fusion_lib as F, m10_velocity_lib as V, m14_bars4h_lib as L, m14_bars4h_grid as G
pk = json.load(open(os.path.join(L.OUT, "pick_params.json"))); print("PICK", pk)
B = pk["B"]; D = G.dz((B, pk["depth"], pk["tp"], pk["hm"]))
row = lambda r, t: f"{r[t+'_wr']*100:5.1f} / {r[t+'_mo']*100:+6.2f} / {r[t+'_dd']*100:6.1f} / {r[t+'_tpm']:4.1f}"
R = {}
for nm, kw in [("normal", {}), ("worse", dict(worse=True)), ("stress", dict(stress=True))]:
    for ho in (False, True):
        if nm == "stress" and ho: continue
        for lab, (d, b) in (("pick", (D, B)), ("90%", (L.BASE, 1))):
            r, e, lg = L.evaluate(d, b, holdout=ho, **kw); R[(nm, ho, lab)] = r
            if lab == "pick" and nm == "normal":
                e.rename("equity").rename_axis("time").to_csv(os.path.join(L.OUT, "pick_eq_hold.csv" if ho else "pick_eq_main.csv"))
for k, r in R.items():
    for t in ("h" if k[1] else "dts"): print(f"{k[0]:7s} {'hold' if k[1] else 'main'} {t} {k[2]:5s} {row(r,t)}")
print("stress 2021-23 mult pick", R[("stress", False, "pick")]["d_mult"], "90%", R[("stress", False, "90%")]["d_mult"])
def items(d, b):
    n = L.evaluate(d, b)[0]; h = L.evaluate(d, b, holdout=True)[0]; w = L.evaluate(d, b, worse=True)[0]; wh = L.evaluate(d, b, worse=True, holdout=True)[0]
    return (n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35, h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663,
            w["t_mo"] > .0531 and wh["h_mo"] > .0741 and w["t_wr"] >= .92 and wh["h_wr"] >= .92)
print("pick items 1-3", items(D, B))
s = R[("stress", False, "pick")]; print("item4: stress 2024-26 mo", s["t_mo"], ">0.0549:", s["t_mo"] > .0549, "d_mult>0.05:", s["d_mult"] > .05)
p = (B, pk["depth"], pk["tp"], pk["hm"]); res = []
for q in G.nbrs(p): res.append((q, items(G.dz(q), q[0])))
for q, r in res: print("nbr", q, r)
print("grid neighbours passing 1-3:", sum(all(r) for _, r in res), "of", len(res))
vr = []
for kw in (dict(cd_add=-1), dict(cd_add=1), dict(mtp_now=True, cd_add=1)):
    out = []
    for ho in (False, True):
        _, idx, P = V.market_holdout(False) if ho else V.market(False)
        e, lg, inv = L.run(P, idx, D, B, **kw); out.append(V.summarize(e, lg, inv, idx, V.HOLD if ho else V.MAIN))
    n, h = out
    vr.append((n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35, h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663))
print("versions item1/item2 (cd-1, cd+1, bot timing):", vr)
# fill quality: share of limit fills vs market (gap-down) fills, avg worst trade
for ho in (False, True):
    r, e, lg = L.evaluate(D, B, holdout=ho); r0, e0, lg0 = L.evaluate(L.BASE, 1, holdout=ho)
    for nm, x in (("pick", lg), ("90%", lg0)):
        print("hold" if ho else "main", nm, "trades", len(x), "time-stop share %.3f" % (x.why == 1).mean(), "mean ret %.4f" % x.ret.mean(), "worst %.3f" % x.ret.min())
