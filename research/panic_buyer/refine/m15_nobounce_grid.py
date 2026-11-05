"""m15 nobounce grid, 2021-23 ONLY. RULE (written before any grid run):
Designs: mode 1 (market sell at open after N hours if best high since fill < entry*(1+y)) with N in {4,6,8,12,16,24},
y in {0.5,1,1.5,2}% (24 designs); mode 2 (move take-profit to entry*(1+tp2), keep 96h stop) same N,y and
tp2 in {0.3,0.8,1.5}% (72 designs). 96 designs.
Bars: normal and every-fill-0.2%-worse 2021-23 win rate >= 92.5%, drop > -64.4%, %/month > 13.17 normal and > 9.79 worse;
all 5 nearby versions (base, cd-1, cd+1, bot timing, late start) must pass wr>=92.5% and drop > -64.4%.
Score = mean of the 5 versions' %/month. Neighbours = one grid step in N, y or tp2 (same mode); the median of the
neighbours' normal 2021-23 %/month, win rate, drop must pass the normal-fill bars. PICK = highest score among designs
passing all of that; none if none pass.
NOTE (added after the first grid run, no design changed): the first run used drop > -0.644, which wrongly rejected the
90%-mode drop itself (-0.64434 rounds to -64.4%); bar corrected to >= -0.6445 (m11 lib uses -0.645)."""
import os, sys, json, itertools
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m15_nobounce_lib as L
NS = [4, 6, 8, 12, 16, 24]; YS = [.005, .01, .015, .02]; TPS = [.003, .008, .015]
designs = [dict(L.OFF, mode=1, N=n, y=y) for n in NS for y in YS] + \
          [dict(L.OFF, mode=2, N=n, y=y, tp2=t) for n in NS for y in YS for t in TPS]
rows = []
for i, d in enumerate(designs):
    s = L.scores(d); s.update(mode=d["mode"], N=d["N"], y=d["y"], tp2=d["tp2"] if d["mode"] == 2 else 0)
    rows.append(s)
    print(i, d["mode"], d["N"], d["y"], d["tp2"], round(s["n_wr"], 4), round(s["n_mo"]*100, 2), round(s["w_mo"]*100, 2), round(s["n_dd"], 3), flush=True)
df = pd.DataFrame(rows)
df.to_csv(os.path.join(L.F.OUT, "grid.csv"), index=False)
def ok(r):
    return (r.n_wr >= .925 and r.n_dd > -.6445 and r.n_mo > .1317 and r.w_wr >= .925 and r.w_dd > -.6445 and r.w_mo > .0979
            and all(r[f"{t}_wr"] >= .925 and r[f"{t}_dd"] > -.6445 for t in ("cm", "cp", "bt", "ls")))
df["ok"] = df.apply(ok, axis=1)
def nbrs(r):
    m = df[df["mode"] == r["mode"]]
    out = []
    for col, vals in (("N", NS), ("y", YS), ("tp2", TPS if r["mode"] == 2 else [0])):
        if len(vals) < 2: continue
        j = vals.index(r[col])
        for jj in (j-1, j+1):
            if 0 <= jj < len(vals):
                q = m[(m[col] == vals[jj]) & all_other_eq(m, r, col)]
                out.append(q.iloc[0])
    return pd.DataFrame(out)
def all_other_eq(m, r, col):
    c = pd.Series(True, index=m.index)
    for o in ("N", "y", "tp2"):
        if o != col: c &= (m[o] == r[o])
    return c
npass = []
for i, r in df.iterrows():
    nb = nbrs(r)
    good = (nb.n_wr.median() >= .925) and (nb.n_dd.median() > -.6445) and (nb.n_mo.median() > .1317)
    df.loc[i, "nbok"] = bool(good)
    df.loc[i, "nbfrac"] = float(((nb.n_wr >= .925) & (nb.n_dd > -.6445) & (nb.n_mo > .1317)).mean())
df.to_csv(os.path.join(L.F.OUT, "grid.csv"), index=False)
fin = df[df.ok & (df.nbok == True)].sort_values("robust", ascending=False)
print("passing bars:", int(df.ok.sum()), "with neighbours:", len(fin))
print(fin[["mode", "N", "y", "tp2", "n_wr", "n_mo", "w_mo", "n_dd", "robust", "rmin", "nbfrac"]].head(15).to_string())
print(df.sort_values("robust", ascending=False)[["mode", "N", "y", "tp2", "n_wr", "n_mo", "w_mo", "n_dd", "robust", "ok", "nbok"]].head(10).to_string())
if len(fin):
    json.dump({k: (float(v) if k != "mode" and k != "N" else int(v)) for k, v in fin.iloc[0][["mode", "N", "y", "tp2"]].items()},
              open(os.path.join(L.F.OUT, "pick.json"), "w"))
