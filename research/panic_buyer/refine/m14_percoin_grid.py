"""m14 percoin grid: per-coin walk-forward depth (12-17%) / take-profit (2/2.5/3%). python3 m14_percoin_grid.py

RULE (written before any run). Each coin's candidates (depth 12..17% x tp 2/2.5/3%) are replayed standalone on its own
history (96h, 3h cooldown, 35% guard); at each refit date D (Jan/Jul or Jan only) the coin uses only its own trades
that exited before D (window: expanding or trailing 24 months; 2018-20 history allowed as past data, never scored).
A candidate replaces the default (14%, +2.5%) only if its shrunk edge over the default is >0:
  delta = (score_cand - score_def) * ne/(ne+m), ne = min(n_cand, n_def), both >= minT trades, score = mean net return
  per trade ('mean') or summed net return per month of window ('sum'); the best positive delta wins.
Design axes: win {exp,24}, R {6,12}, minT {5,10}, m {5,20}, metric {mean,sum}, cset {both,depth-only} = 64 designs.
Bars (M14_PREREG) on 2021-23, normal AND every fill 0.2% worse: win rate >=92.5%, drop > -64.4%, %/month >13.17 / 9.79;
all 5 nearby versions (base, cd-1, cd+1, bot timing, start 2021-01-15) pass win rate and drop; score = mean %/month of
the 5 versions; neighbours (one step on one axis) median normal win rate / drop / %/month must pass bar 1.
(Drop bar -64.4% is applied as > -64.45%, i.e. rounded: 90% mode itself is -64.43%; fixed after run 1 showed all 64 designs failed the strict float bar.)
Pick = highest score among those passing; none -> no pick. Nothing after 2023 is summarised here.
"""
import itertools, json, os, sys
import numpy as np, pandas as pd
import m14_percoin_lib as L, m10_velocity_lib as V, m12_entry_lib as M

WR, DD, MON, MOW = 0.925, -0.6445, 0.1317, 0.0979
AX = dict(win=[0, 24], R=[6, 12], minT=[5, 10], m=[5, 20], metric=["mean", "sum"], cset=["both", "depth"])
NM = list(AX)
VERS = {"n": {}, "cm": dict(cd_add=-1), "cp": dict(cd_add=1), "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=M.LATE)}
PEN = 0.002
_C = {}


def env():
    if not _C:
        sy, idx, P = V.market(False)
        sh, ih, Ph = V.market_holdout(False)
        stm = L.trade_store(P, idx); sth = L.trade_store(Ph, ih)
        _C["main"] = (idx, P)
        _C["hold"] = (ih, Ph)
        _C["store"] = L.merge_stores([(list(Ph["o"].columns), sth), (list(P["o"].columns), stm)])
        _C["first"] = {**L.first_bar(Ph), **{s: v for s, v in L.first_bar(P).items() if s not in L.first_bar(Ph)}}
    return _C


def des_of(ix): return {k: AX[k][i] for k, i in zip(NM, ix)}


def tables(des, which="main"):
    C = env(); idx, P = C[which]
    return L.schedule(P, idx, C["store"], des, C["first"])


def score(des):
    C = env(); idx, P = C["main"]
    DEP, TP = tables(des)
    out = {}
    for tag, kw in list(VERS.items()) + [("w", dict(pen=PEN, pens=PEN))]:
        e, lg, inv = L.prun(P, idx, DEP, TP, **kw)
        per = (("d", M.LATE, V.T24),) if tag == "ls" else M.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[tag + "_wr"], out[tag + "_mo"], out[tag + "_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
    out["robust"] = float(np.mean([out[t + "_mo"] for t in VERS]))
    out["ok_versions"] = all(out[t + "_wr"] >= WR and out[t + "_dd"] > DD for t in VERS)
    out["pass1"] = bool(out["n_wr"] >= WR and out["n_dd"] > DD and out["n_mo"] > MON and out["w_wr"] >= WR
                        and out["w_dd"] > DD and out["w_mo"] > MOW)
    return out


if __name__ == "__main__":
    ixs = list(itertools.product(*[range(len(v)) for v in AX.values()]))
    res = {}
    for ix in ixs:
        res[ix] = score(des_of(ix))
        r = res[ix]
        print(ix, f"wr {r['n_wr']:.3f} mo {r['n_mo']:.4f} dd {r['n_dd']:.3f} | worse mo {r['w_mo']:.4f} | rob {r['robust']:.4f} p1 {r['pass1']} v {r['ok_versions']}", flush=True)
    rows = []
    for ix in ixs:
        r = res[ix]
        nb = []
        for a in range(len(NM)):
            for dlt in (-1, 1):
                j = ix[a] + dlt
                if 0 <= j < len(AX[NM[a]]):
                    nb.append(res[ix[:a] + (j,) + ix[a + 1:]])
        med = lambda k: float(np.median([x[k] for x in nb]))
        nbok = med("n_wr") >= WR and med("n_dd") > DD and med("n_mo") > MON
        rows.append(dict(ix=ix, des=des_of(ix), robust=r["robust"], pass1=r["pass1"], ok_versions=r["ok_versions"], nbok=nbok,
                         npass=float(np.mean([x["pass1"] for x in nb]))))
    el = [x for x in rows if x["pass1"] and x["ok_versions"] and x["nbok"]]
    el.sort(key=lambda x: -x["robust"])
    print("designs", len(rows), "pass1", sum(x["pass1"] for x in rows), "eligible", len(el))
    for x in el[:10]: print(x)
    best = sorted(rows, key=lambda x: -x["robust"])[:5]
    print("top by robust regardless:"); [print(x) for x in best]
    json.dump(dict(pick=el[0] if el else None, rows=rows), open(os.path.join(M.OUT, "grid.json"), "w"), default=str)
