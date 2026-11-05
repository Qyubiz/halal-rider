"""m14 confirm grid. RULE (written before any run; 2021-23 data only):
Designs (144): confirmation g0 (first green close) | g2 (two rising closes) | x1 (close >= dip low +1%) | x3 (+3%)
 x dip level 12/14/16% x cancel after N=12/24/48 h x cancel on a further fall y = none/6% under the dip level
 x take-profit over the fill 2.5%/3.5%. Exits 96h, cooldown 3h, guard 35%, one coin x 100%.
Pass bar 1 (normal and 0.2%-worse fills): win rate >= 92.5%, drop > -64.4%, %/month > 13.17 / 9.79.
Score = mean 2021-23 %/month over 5 versions (base, cd-1, cd+1, bot timing, start 2021-01-15), all 5 must pass the
win-rate and drop bars. Neighbours (one grid step in one parameter, within the grid): median normal-fill win rate,
drop and %/month must pass bar 1. Pick = best score among those; none if none. Nothing after 2023 is computed here.
"""
import itertools, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_confirm_lib as C

AX = dict(conf=["g0", "g2", "x1", "x3"], dep=[0.12, 0.14, 0.16], N=[12, 24, 48], y=[0.0, 0.06], tp=[0.025, 0.035])
NAMES = list(AX)
WR, DD, MO_N, MO_W = 0.925, -0.644, 0.1317, 0.0979


def design(ix):
    p = {k: AX[k][i] for k, i in zip(NAMES, ix)}
    cm, x = {"g0": (0, 0.0), "g2": (2, 0.0), "x1": (1, 0.01), "x3": (1, 0.03)}[p["conf"]]
    return dict(cmode=cm, x=x, dep=p["dep"], N=p["N"], y=p["y"], tp=p["tp"], hm=96, cd=3, guard=0.35)


if __name__ == "__main__":
    out = os.environ["PB_OUT"]
    ixs = list(itertools.product(*[range(len(v)) for v in AX.values()]))
    rows = []
    for ix in ixs:
        s = C.design_scores(design(ix)); s["ix"] = list(ix); rows.append(s)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/grid_all.csv", index=False)
    ok1 = lambda r: (r.n_wr >= WR and r.n_dd > DD and r.n_mo > MO_N and r.w_wr >= WR and r.w_dd > DD and r.w_mo > MO_W)
    ok5 = lambda r: all(r[f"{t}_wr"] >= WR and r[f"{t}_dd"] > DD for t in ("n", "w", "cm", "cp", "bt", "ls"))
    key = {tuple(r.ix): i for i, r in df.iterrows()}
    def neigh(ix):
        out_ = []
        for a in range(len(ix)):
            for dlt in (-1, 1):
                j = list(ix); j[a] += dlt
                if 0 <= j[a] < len(AX[NAMES[a]]): out_.append(key[tuple(j)])
        return out_
    res = []
    for i, r in df.iterrows():
        nb = df.loc[neigh(tuple(r.ix))]
        nbok = (nb.n_wr.median() >= WR and nb.n_dd.median() > DD and nb.n_mo.median() > MO_N)
        res.append(dict(i=i, ix=r.ix, bar1=bool(ok1(r)), v5=bool(ok5(r)), nbok=bool(nbok), robust=r.robust,
                        n_mo=r.n_mo, n_wr=r.n_wr, n_dd=r.n_dd, w_mo=r.w_mo, nbmed=nb.n_mo.median()))
    R = pd.DataFrame(res)
    R.to_csv(f"{out}/grid_sel.csv", index=False)
    fin = R[R.bar1 & R.v5 & R.nbok].sort_values("robust", ascending=False)
    print("designs", len(R), "bar1", R.bar1.sum(), "bar1+5v", (R.bar1 & R.v5).sum(), "final", len(fin))
    print(R.sort_values("n_mo", ascending=False).head(8).to_string())
    print(fin.head(5).to_string())
    pick = None
    if len(fin):
        ix = fin.iloc[0].ix; pick = dict(ix=ix, design=design(tuple(ix)), robust=float(fin.iloc[0].robust))
    json.dump(pick, open(f"{out}/pick.json", "w"), indent=1)
    print(pick)
