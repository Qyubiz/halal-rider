"""m12 sizing (agent A) grid search on 2021-23 ONLY (data cut at 2024-01-01 inside every run).

RANKING RULE (written before any run; follows M12_PREREG.md):
  Bars (2021-23, normal fills AND every fill 0.2% worse): win rate >= 92.5%, worst drop no deeper than -64.4%,
       %/month above 90% mode's (13.17% normal, 9.79% worse; exact values from this engine), and the robust score
       (below) above 90% mode's robust score.
  Robust score: mean 2021-23 %/month over 5 nearby versions: as designed, cooldown 2h, cooldown 4h, bot timing
       (take-profit at once after a market buy + cooldown +1h), start 2021-01-15 (m12_sizing_lib.design_scores).
  Drop proxy for the -35% test limit: among designs passing the bars, rank by robust score those whose normal-fill
       2021-23 worst drop is no deeper than -50%; if none, those no deeper than -57%; if none, all passers.
  Neighbours: every switched-on numeric setting moved one notch up and down on the NB grids (one at a time); the
       MEDIAN over neighbours of normal & worse win rate, worst drop and %/month must pass the same bars (excluding
       the robust-score bar). Walk down the ranked list (at most 8 tries); first design whose neighbours pass = THE PICK.
  If nothing passes: no pick, the answer is "nothing beats 90% mode" (the best failing design is still reported, and
       m12_sizing_final.py may show it labelled as NOT a pick).
SEARCH (<= ~300 designs): stage 1 one sizing family at a time (base fraction / second slot, losing streak, account
  drawdown, volatility targeting, crash-guard depth, trailing Kelly, deep fill); stage 2 pairs of the best design of
  the 4 best families (by the same ordering: passers by score, then fewest failed bars, then score).
Usage: source m12env.sh; python3 m12_sizing_grid.py   -> $PB_OUT/grid.csv, grid.log, pick.json
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_sizing_lib as S  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

LOG = open(os.path.join(S.OUT, "grid.log"), "a")
ROWS, SEEN = [], {}


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def key(d):
    return json.dumps(d, sort_keys=True)


BASE = S.design_scores({})
B_N, B_W, B_R = BASE["n_mo"], BASE["w_mo"], BASE["robust"]
WR, DD = 0.925, -0.644


def fails(s, robust=True):
    f = int(s["n_wr"] < WR) + int(s["w_wr"] < WR) + int(s["n_dd"] < DD - 5e-5) + int(s["w_dd"] < DD - 5e-5) + \
        int(s["n_mo"] <= B_N) + int(s["w_mo"] <= B_W)
    if robust:
        f += int(s["robust"] <= B_R)
    return f


def score(d, stage, fam):
    k = key(d)
    if k in SEEN:
        return SEEN[k]
    s = S.design_scores(d)
    s.update(fails=fails(s), stage=stage, fam=fam, design=k, name=S.describe(d))
    SEEN[k] = s; ROWS.append(s)
    say(f"[{stage}/{fam}] {s['name']:60s} n {s['n_wr']*100:5.1f}/{s['n_mo']*100:+6.2f}/{s['n_dd']*100:5.1f} "
        f"w {s['w_wr']*100:5.1f}/{s['w_mo']*100:+6.2f}/{s['w_dd']*100:5.1f} rob {s['robust']*100:+6.2f} "
        f"size {s['n_size']:.2f} fails {s['fails']}")
    return s


def order(rows):
    return sorted(rows, key=lambda s: (s["fails"], -s["robust"]))


NB = dict(fs=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], ln=[1, 2, 3, 4], lmul=[0.33, 0.5, 0.75, 0.9], lmin=[0.1, 0.25, 0.5, 0.75],
          wstep=[0.1, 0.25, 0.5, 1.0], d0=[0.0, 0.1, 0.2, 0.3, 0.4], dk=[0.5, 1.0, 2.0, 4.0, 8.0],
          dmin=[0.1, 0.25, 0.5, 0.75], pw=[0, 336, 720, 2160], vmin=[0.1, 0.25, 0.5, 0.75],
          g1=[0.1, 0.15, 0.2, 0.25, 0.3], gf=[0.25, 0.33, 0.5, 0.75, 0.9], kN=[5, 10, 20, 40, 80],
          kc=[0.02, 0.05, 0.1, 0.2, 0.4], kmin=[0.1, 0.25, 0.5, 0.75], dx=[0.01, 0.02, 0.05, 0.1, 0.15],
          dm=[1.25, 1.5, 2.0, 3.0], nslots=[1, 2, 3])


def neighbours(d):
    out = []
    for p, v in d.items():
        if p == "vt":
            for f in (0.8, 1.25):
                out.append(dict(d, vt=v * f))
            continue
        if p not in NB or p == "nbid":
            continue
        g = NB[p]
        if v not in g:
            continue
        i = g.index(v)
        for j in (i - 1, i + 1):
            if 0 <= j < len(g):
                nd = dict(d, **{p: g[j]})
                if nd.get("nbid", 1) > nd.get("nslots", 1):
                    nd["nbid"] = nd["nslots"]
                out.append(nd)
    return out


def main():
    t00 = time.time()
    say(f"=== m12 sizing grid {time.ctime()}  90% mode: n {B_N*100:.2f} w {B_W*100:.2f} robust {B_R*100:.2f}")
    syms, idx, P = V.market(False)
    ne = int(np.searchsorted(idx.values, np.datetime64(S.T24)))
    vols, *_ = S.feats(P)
    q = {}
    for src, i in S.VSRC.items():
        x = vols[i][:ne]; x = x[x > 0]
        q[src] = [float(np.quantile(x, p)) for p in (0.5, 0.7, 0.9)]
    say("vol quantiles 2021-23 (50/70/90%):", {k: [round(a, 5) for a in v] for k, v in q.items()})
    fam = {}
    # A: base fraction / second slot
    A = [dict(fs=f, nslots=ns, nbid=nb) for f in (0.5, 0.6, 0.7, 0.8, 0.9) for ns, nb in ((1, 1), (2, 1), (2, 2))]
    A += [dict(fs=0.4, nslots=3, nbid=1), dict(fs=0.5, nslots=3, nbid=1)]
    # B: losing streak
    B = [dict(ln=a, lmul=b, lmin=c, wstep=w, nslots=ns) for a in (1, 2, 3) for b in (0.5, 0.75) for c in (0.25, 0.5)
         for w in (0.25, 1.0) for ns in (1, 2)]
    # C: account drawdown
    C = [dict(d0=a, dk=b, dmin=c, pw=p, nslots=ns) for a in (0.1, 0.2, 0.3) for b in (1.0, 2.0, 4.0)
         for c in (0.25, 0.5) for p in (0, 720) for ns in (1, 2)]
    # D: volatility targeting
    D = [dict(vt=round(q[src][j], 6), vsrc=src, vmin=m, nslots=ns) for src in S.VSRC for j in range(3)
         for m in (0.25, 0.5) for ns in (1, 2)]
    # E: crash-guard depth
    Eg = [dict(g1=a, gf=b, nslots=ns) for a in (0.15, 0.2, 0.25, 0.3) for b in (0.33, 0.5, 0.75) for ns in (1, 2)]
    # F: trailing Kelly
    Fk = [dict(kN=a, kc=b, kmin=c, nslots=2) for a in (10, 20, 40) for b in (0.05, 0.1, 0.2) for c in (0.25, 0.5)]
    # G: deep fill
    G = [dict(fs=f, dx=a, dm=b, nslots=2) for f in (0.5, 0.7) for a in (0.02, 0.05, 0.1) for b in (1.5, 2.0)]
    for name, L in (("A", A), ("B", B), ("C", C), ("D", D), ("E", Eg), ("F", Fk), ("G", G)):
        fam[name] = order([score(d, 1, name) for d in L])
        say(f"  family {name} best: {fam[name][0]['name']} fails {fam[name][0]['fails']} "
            f"rob {fam[name][0]['robust']*100:.2f}   ({time.time()-t00:.0f}s)")
    top = sorted(fam, key=lambda f: (fam[f][0]["fails"], -fam[f][0]["robust"]))[:4]
    say("stage 2 families:", top)
    for a, b in itertools.combinations(top, 2):
        da, db = json.loads(fam[a][0]["design"]), json.loads(fam[b][0]["design"])
        for ns in sorted({da.get("nslots", 1), db.get("nslots", 1)}):
            d = dict(da); d.update(db); d["nslots"] = ns
            d["nbid"] = min(d.get("nbid", 1), ns)
            if "fs" in da and "fs" in db:
                d["fs"] = min(da["fs"], db["fs"])
            score(d, 2, a + b)
    df = pd.DataFrame(ROWS); df.to_csv(os.path.join(S.OUT, "grid.csv"), index=False)
    say(f"designs scored: {len(ROWS)}  passing all bars: {(df.fails == 0).sum()}")
    passers = [s for s in ROWS if s["fails"] == 0]
    for lim in (-0.50, -0.57, -1.0):
        ranked = sorted([s for s in passers if s["n_dd"] >= lim], key=lambda s: -s["robust"])
        if ranked:
            say(f"ranking tier: normal 2021-23 drop >= {lim}: {len(ranked)} designs")
            break
    pick = None
    for s in (ranked if passers else [])[:8]:
        d = json.loads(s["design"])
        nb = [S.design_scores(x) for x in neighbours(d)]
        med = {k: float(np.median([x[k] for x in nb])) for k in ("n_wr", "w_wr", "n_dd", "w_dd", "n_mo", "w_mo", "robust")}
        f = fails(med, robust=False)
        say(f"  try {s['name']}: {len(nb)} neighbours, median n {med['n_wr']*100:.1f}/{med['n_mo']*100:+.2f}/"
            f"{med['n_dd']*100:.1f} w {med['w_wr']*100:.1f}/{med['w_mo']*100:+.2f}/{med['w_dd']*100:.1f} fails {f}")
        if f == 0:
            pick = dict(design=d, name=s["name"], scores=s, nb_median=med); break
    if pick is None:
        best = order(ROWS)[0]
        say(f"NO PICK. Best failing design: {best['name']} fails {best['fails']}")
        json.dump(dict(design=None, best_failing=json.loads(best["design"]), name=best["name"]),
                  open(os.path.join(S.OUT, "pick.json"), "w"), indent=1)
    else:
        say("PICK:", pick["name"], json.dumps(pick["design"]))
        json.dump(pick, open(os.path.join(S.OUT, "pick.json"), "w"), indent=1)
    say(f"done in {time.time()-t00:.0f}s")


if __name__ == "__main__":
    main()
