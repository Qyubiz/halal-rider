"""m17 multi final look: pick = best robust among eligible (rule in m17_multi_grid.py), stressed 2021-23 check first."""
import json, os
import pandas as pd
import m11_fusion_lib as F, m17_multi_grid as G
g = pd.read_csv(os.path.join(F.OUT, "grid.csv")); g = g[g.elig].sort_values("robust", ascending=False)
pk = None
for _, b in g.iterrows():
    d = G.design(b.d1, int(b.fast), b.tp, int(b.P))
    r, _, _ = F.evaluate(d, stress=True)
    print("stress2123 mult", b.d1, b.fast, b.tp, b.P, round(r["d_mult"], 3))
    if r["d_mult"] > 0.05: pk = b; break
d = G.design(pk.d1, int(pk.fast), pk.tp, int(pk.P)); print("PICK", d)
json.dump(dict(d1=pk.d1, fast=int(pk.fast), tp=pk.tp, P=int(pk.P), design=d), open(os.path.join(F.OUT, "pick_params.json"), "w"))
def row(r, t): return f"{r[t+'_wr']*100:5.1f} / {r[t+'_mo']*100:+6.2f} / {r[t+'_dd']*100:6.1f} / {r[t+'_tpm']:4.1f}"
for nm, kw, tags in [("normal main", {}, "dts"), ("normal hold", dict(holdout=True), "h"), ("worse main", dict(worse=True), "dts"),
                     ("worse hold", dict(worse=True, holdout=True), "h"), ("stress main", dict(stress=True), "dt")]:
    r, e, _ = F.evaluate(d, **kw); b, _, _ = F.evaluate(F.MODE90, **kw)
    for t in tags: print(f"{nm:12s} {t}  pick {row(r,t)} | 90% {row(b,t)}")
    if nm == "normal main": e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_main.csv"))
    if nm == "normal hold": e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_hold.csv"))
    if nm == "stress main": print("stress mult 21-23", r["d_mult"])
def items(dd, **extra):
    n, _, _ = F.evaluate(dd); h, _, _ = F.evaluate(dd, holdout=True); w, _, _ = F.evaluate(dd, worse=True); wh, _, _ = F.evaluate(dd, worse=True, holdout=True)
    return (n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35, h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663,
            w["t_mo"] > .0531 and wh["h_mo"] > .0741 and w["t_wr"] >= .92 and wh["h_wr"] >= .92)
print("pick items 1-3", items(d))
res = []
ks = ["d1", "fast", "tp", "P"]
for k in ks:
    for dl in (-1, 1):
        ax = G.AX[k]; cur = pk[k] if k in ("d1", "tp") else int(pk[k]); i = ax.index(cur) + dl
        if 0 <= i < len(ax):
            q = dict(d1=pk.d1, fast=int(pk.fast), tp=pk.tp, P=int(pk.P)); q[k] = ax[i]
            res.append(items(G.design(**q)))
print("neighbours passing 1-3:", sum(all(r) for r in res), "of", len(res), res)
# relaxed look (hindsight, labelled): the same items with the 20%/month bar
