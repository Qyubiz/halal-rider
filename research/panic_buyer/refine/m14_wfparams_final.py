"""m14 wfparams final look (labelled FALLBACK: the pre-written rule gave no pick, 0/64 pass bar 1, all by the -64.4% drop bar).
Fallback = best robust score among designs with win rate>=92.5%, %/month bars (normal & worse) all met on all versions, ignoring only the drop bar.
One look; no re-picking."""
import json, os, itertools
import numpy as np, pandas as pd
import m14_wfparams_lib as L, m10_velocity_lib as V, m11_fusion_lib as F
import m14_wfparams_grid as G
g = pd.read_csv(os.path.join(F.OUT, "grid.csv"))
ok = g[g.apply(lambda s: all(s[t + "_wr"] >= G.WR for t in list(G.VERS) + ["w"]) and s.n_mo > G.MON and s.w_mo > G.MOW, axis=1)]
print("designs passing everything but drop:", len(ok), "of", len(g))
b = ok.sort_values("robust", ascending=False).iloc[0]
pk = {k: (int(b[k]) if k in ("W", "R") else b[k]) for k in G.NM}
print("FALLBACK", pk, "score", b.robust); json.dump(pk, open(os.path.join(F.OUT, "pick_params.json"), "w"))

def look(des, panel, mode, **kw):
    (e, lg, inv), dep, tp, idx = L.wf_run(des, panel, mode, **kw)
    per = V.HOLD if panel == "hold" else V.MAIN
    return V.summarize(e, lg, inv, idx, per), e
def row(r, t): return f"{r[t+'_wr']*100:5.1f} / {r[t+'_mo']*100:+6.2f} / {r[t+'_dd']*100:6.1f} / {r[t+'_tpm']:4.1f}"
B = dict(W=12, R=6, cs="depth", sc="wr92")  # unused
def base(panel, mode):
    kw = dict(L.MODES[mode]); dead = mode == "st"
    _, idx, P = V.market_holdout(dead) if panel == "hold" else V.market(dead)
    e, lg, inv = L.const(P, idx, .14, .025, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if panel == "hold" else V.MAIN)
def table(des):
    rows = {}
    for nm, panel, mode, tags in [("normal main", "main", "n", "dts"), ("normal hold", "hold", "n", "h"), ("worse main", "main", "w", "dts"),
                                  ("worse hold", "hold", "w", "h"), ("stress main", "main", "st", "dt")]:
        r, e = look(des, panel, mode); rb = base(panel, mode)
        for t in tags:
            rows[(nm, t)] = (r, rb)
            print(f"{nm:12s} {t}  pick {row(r,t)} | 90% {row(rb,t)}")
        if nm == "normal main": e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_main.csv"))
        if nm == "normal hold": e.rename("equity").rename_axis("time").to_csv(os.path.join(F.OUT, "pick_eq_hold.csv"))
    return rows
rows = table(pk)
def items(des, full=False):
    n, _ = look(des, "main", "n"); h, _ = look(des, "hold", "n"); w, _ = look(des, "main", "w"); wh, _ = look(des, "hold", "w")
    i1 = n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35
    i2 = h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663
    i3 = w["t_mo"] > .0531 and wh["h_mo"] > .0741 and w["t_wr"] >= .92 and wh["h_wr"] >= .92
    return i1, i2, i3
print("pick items 1-3 (stress item 4 below):", items(pk))
s, _ = look(pk, "main", "st"); print("stress 2024-26 mo", s["t_mo"], "2021-23 mult", s["d_mult"], "(fail if <=0.05)")
# neighbours: design neighbours + nearby versions (main only, versions of pick)
ix = tuple(G.AX[k].index(pk[k]) for k in G.NM)
res = []
for j in range(4):
    for dl in (-1, 1):
        k = list(ix); k[j] += dl
        if 0 <= k[j] < len(G.AX[G.NM[j]]): d = G.des_of(k); res.append(items(d))
print("neighbours", len(res), "pass 1-3:", sum(all(r) for r in res), res)
# nearby versions of pick: cd-1, cd+1, bot timing on main (item1) and hold (item2)
vr = []
for kw in (dict(cd_add=-1), dict(cd_add=1), dict(mtp_now=True, cd_add=1)):
    n, _ = look(pk, "main", "n", **kw); h, _ = look(pk, "hold", "n", **kw)
    vr.append((n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35, h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663))
print("versions item1/item2:", vr)
(e, lg, inv), dep, tp, idx = L.wf_run(pk, "main", "n")
print(pd.Series(list(zip(dep, tp))).groupby(idx.to_period("Q").astype(str).values).first().to_string())
