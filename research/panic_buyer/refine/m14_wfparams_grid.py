"""m14 wfparams grid: walk-forward re-tuning rule, selected on 2021-23 only.  python3 m14_wfparams_grid.py

RULE (written before any run). Design = (W trailing months 12/24, R refit every 3/6 months, candidate set
full15 / depth-only / tp-only / coarse6, score wr92 / wr94 / mo / wr). Refit at the first of month; uses only
completed months before it (2019-20 holdout months allowed as past data for 2021 refits; never scored).
Bars on 2021-23 (M14_PREREG): normal AND every-fill-0.2%-worse: win rate >= 92.5%, worst drop > -64.4%, %/month >
13.17 normal / 9.79 worse; all 5 nearby versions (base, cd-1, cd+1, bot timing, start 2021-01-15) pass win rate and
drop; score = mean %/month of the 5 versions; neighbours (one step in one parameter; sets/scores adjacent in the
listed order) median of normal win rate / drop / %/month must pass the bars. Pick = highest score. None -> no pick.
Nothing after 2023 is summarised here.
"""
import itertools, json, sys, os
import numpy as np, pandas as pd
import m14_wfparams_lib as L, m10_velocity_lib as V, m11_fusion_lib as F

WR, DD, MON, MOW = 0.925, -0.644, 0.1317, 0.0979
AX = dict(W=[12, 24], R=[3, 6], cs=["full", "depth", "tp", "coarse"], sc=["wr92", "wr94", "mo", "wr"])
NM = list(AX)
VERS = {"n": {}, "cm": dict(cd_add=-1), "cp": dict(cd_add=1), "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}


def des_of(ix): return {k: AX[k][i] for k, i in zip(NM, ix)}


def score(des):
    _, idx, P = V.market(False)
    out = {}
    for tag, kw in list(VERS.items()) + [("w", dict(pen=F.PEN, pens=F.PEN))]:
        st = L.cand_stats("w" if tag == "w" else "n")
        dep, tp = L.schedule(idx, st, des)
        e, lg, inv = L.run(P, idx, dep, tp, **kw)
        per = (("d", F.LATE, F.T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[tag + "_wr"], out[tag + "_mo"], out[tag + "_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
    out["robust"] = float(np.mean([out[t + "_mo"] for t in VERS]))
    return out


if __name__ == "__main__":
    ixs = list(itertools.product(*[range(len(v)) for v in AX.values()]))
    rows = []
    for ix in ixs:
        d = des_of(ix); s = score(d); s.update(d); s["ix"] = ix; rows.append(s)
    df = pd.DataFrame(rows)
    ok_all = lambda s: all(s[t + "_wr"] >= WR and s[t + "_dd"] > DD for t in list(VERS) + ["w"]) and s["n_mo"] > MON and s["w_mo"] > MOW
    df["p1"] = df.apply(ok_all, axis=1)
    bymap = {r.ix: r for r in df.itertuples()}
    def nbr(ix):
        for j in range(len(ix)):
            for dlt in (-1, 1):
                k = list(ix); k[j] += dlt
                if 0 <= k[j] < len(AX[NM[j]]): yield tuple(k)
    med = []
    for r in df.itertuples():
        ns = [bymap[k] for k in nbr(r.ix)]
        m = (np.median([x.n_wr for x in ns]) >= WR and np.median([x.n_dd for x in ns]) > DD and np.median([x.n_mo for x in ns]) > MON)
        med.append(m)
    df["p3"] = med
    df.to_csv(os.path.join(F.OUT, "grid.csv"), index=False)
    ok = df[df.p1 & df.p3].sort_values("robust", ascending=False)
    pd.set_option("display.width", 250)
    print(df.sort_values("robust", ascending=False).head(12)[NM + ["n_wr", "n_mo", "n_dd", "w_mo", "robust", "p1", "p3"]])
    print("pass p1:", int(df.p1.sum()), "pass p1&p3:", len(ok), "of", len(df))
    if len(ok):
        b = ok.iloc[0]; pick = {k: b[k] for k in NM}; pick["W"] = int(pick["W"]); pick["R"] = int(pick["R"])
        json.dump(dict(pick=pick, ix=list(map(int, b.ix))), open(os.path.join(F.OUT, "pick.json"), "w")); print("PICK", pick)
    else:
        print("NO PICK")
