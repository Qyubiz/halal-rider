"""m14 grid: about 9,000 settings of the m11 fusion engine, selected on 2021-23 only (rule: M14_PREREG.md).

  python3 m14_grid.py stage1 [workers]   every setting, normal and 0.2%-worse fills, 2021-23 numbers only
  python3 m14_grid.py stage2 [workers]   settings passing bar 1: 5 nearby versions + neighbour medians -> finalists

stage1 and stage2 never compute or store anything after 2023 (frun runs 2021-26 but only 2021-23 is summarized).
Outputs in $PB_OUT: stage1.csv, stage2.csv, finalists.json.
"""
import itertools
import json
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402
import m11_fusion_lib as F  # noqa: E402

OUT = F.OUT
RNG = ("rng", 0.03, 0.5, 0.01, 0.06)
AX = dict(depth=[0.12, 0.13, 0.14, 0.15, 0.16], tp=[0.02, 0.025, 0.03, RNG], hm=[72, 96, 144], cdw=[0, 3],
          vol=[0.0, 0.07], bon=[0, 1], bref=[6, 12], bdep=[0.10, 0.13], btp=[0.02, 0.03], guard=[0.30, 0.35],
          size=[1, 2])
NAMES = list(AX)
WR, DD, MO_N, MO_W = 0.925, -0.644, 0.1317, 0.0979


def grid():
    out = []
    for ix in itertools.product(*[range(len(v)) for v in AX.values()]):
        g = dict(zip(NAMES, ix))
        if g["bon"] == 0 and (g["bref"] or g["bdep"] or g["btp"]):
            continue                                     # second-trigger settings are irrelevant when it is off
        out.append(ix)
    return out


def design(ix):
    p = {k: AX[k][i] for k, i in zip(NAMES, ix)}
    fs = 1.0 if p["size"] == 1 else 0.5
    tiers = [dict(ref=24, depth=p["depth"], tp=p["tp"], hm=p["hm"], cdw=p["cdw"], cdl=3, fs=fs, vol=p["vol"])]
    if p["bon"]:
        tiers.append(dict(ref=p["bref"], depth=p["bdep"], tp=p["btp"], hm=48, cdw=p["cdw"], cdl=3, fs=fs, vol=0.0))
    return dict(F.OFFRISK, guard=p["guard"], nbid=p["size"], nslots=99 if p["size"] == 1 else 2, tiers=tiers)


_M = {}


def _s1(ix):
    if "m" not in _M:
        _M["m"] = V.market(False)
    syms, idx, P = _M["m"]
    d = design(ix)
    row = {"ix": json.dumps(ix)}
    for tag, kw in (("n", {}), ("w", dict(pen=F.PEN, pens=F.PEN))):
        e, lg, inv = F.frun(P, idx, d, **kw)
        r = V.summarize(e, lg, inv, idx, F.DPER)
        row.update({f"{tag}_wr": r["d_wr"], f"{tag}_mo": r["d_mo"], f"{tag}_dd": r["d_dd"]})
    return row


def bar1(r):
    return (r["n_wr"] >= WR and r["w_wr"] >= WR and r["n_dd"] >= DD and r["w_dd"] >= DD
            and r["n_mo"] > MO_N and r["w_mo"] > MO_W)


def neighbours(ix, have):
    out = []
    for j, k in enumerate(NAMES):
        for step in (-1, 1):
            y = list(ix)
            y[j] += step
            if 0 <= y[j] < len(AX[k]) and tuple(y) in have:
                out.append(tuple(y))
    return out


def _s2(ix):
    s = F.design_scores(design(ix))
    s["ix"] = json.dumps(ix)
    return s


def main():
    stage = sys.argv[1]
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    if stage == "stage1":
        G = grid()
        print(f"{len(G)} settings", flush=True)
        with Pool(w) as pool:
            rows = pool.map(_s1, G, chunksize=50)
        df = pd.DataFrame(rows)
        df.to_csv(os.path.join(OUT, "stage1.csv"), index=False)
        df["pass1"] = df.apply(bar1, axis=1)
        print(f"pass bar 1: {int(df.pass1.sum())} of {len(df)}")
        b = df[df.ix == json.dumps(next(ix for ix in G if design(ix)["tiers"][0]["depth"] == 0.14
                                        and design(ix)["tiers"][0]["tp"] == 0.025 and design(ix)["tiers"][0]["hm"] == 96
                                        and design(ix)["tiers"][0]["cdw"] == 3 and design(ix)["tiers"][0]["vol"] == 0
                                        and len(design(ix)["tiers"]) == 1 and design(ix)["guard"] == 0.35
                                        and design(ix)["nbid"] == 1))]
        print("90% mode row:\n" + b.to_string(index=False))
    else:
        df = pd.read_csv(os.path.join(OUT, "stage1.csv"))
        df["pass1"] = df.apply(bar1, axis=1)
        tab = {tuple(json.loads(x)): r for x, r in zip(df.ix, df.to_dict("records"))}
        cand = [tuple(json.loads(x)) for x in df[df.pass1].ix]
        keep = []
        for ix in cand:
            nb = [tab[y] for y in neighbours(ix, tab)]
            if not nb:
                continue
            med = {k: float(np.median([r[k] for r in nb])) for k in ("n_wr", "n_mo", "n_dd")}
            if med["n_wr"] >= WR and med["n_dd"] >= DD and med["n_mo"] > MO_N:
                keep.append(ix)
        print(f"{len(cand)} pass bar 1, {len(keep)} also pass the neighbour median", flush=True)
        with Pool(w) as pool:
            rows = pool.map(_s2, keep, chunksize=5)
        s2 = pd.DataFrame(rows)
        ok = np.ones(len(s2), bool)
        for t in ("n", "w", "cm", "cp", "bt", "ls"):
            ok &= (s2[f"{t}_wr"] >= WR).values & (s2[f"{t}_dd"] >= DD).values
        s2["ok"] = ok
        s2 = s2.sort_values("robust", ascending=False)
        s2.to_csv(os.path.join(OUT, "stage2.csv"), index=False)
        fin = [json.loads(x) for x in s2[s2.ok].ix.head(20)]
        json.dump(fin, open(os.path.join(OUT, "finalists.json"), "w"))
        print(f"{int(ok.sum())} pass all 5 versions; finalists: {len(fin)}")
        for x in fin:
            print(F.describe(design(tuple(x))), "robust", round(float(s2[s2.ix == json.dumps(x)].robust.iloc[0]) * 100, 2))


if __name__ == "__main__":
    main()
