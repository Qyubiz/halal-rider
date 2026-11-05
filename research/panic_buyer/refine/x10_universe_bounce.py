"""x10 lever "universe", round 5: universe = the coins whose own recent dips bounced best.

Written after rounds 1-3 were seen (so not independent of 2024-26; numbers are semi-honest at best).
At each month's first hour, among the top M coins by trailing-365-day volume, rank coins by their standalone dip
record (sum of the returns of their own dip trades with the same depth/tp/hm/cd that EXITED in the trailing
180 days; no trades = 0) and trade only the best k. Only past prices are used.

GRID (192 variants): universe {B30k10, B50k10, B50k20} x mode {bid, watch} x sizing {N1 x 100%, N2 x 50%}
  x depth {12, 15%} x tp {+3, +5%} x cooldown {12, 24h} x time stop {48, 96h}.
Rule A (2022 >= 0%, 2021-23 drop >= -50%, best 2021-23 %/month) is applied and reported with the hindsight best.
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_lib as X  # noqa: E402
import x10_universe_round2 as G2  # noqa: E402

UNIVS5 = [("B30k10", 30, 10), ("B50k10", 50, 10), ("B50k20", 50, 20)]
MODES = ["bid", "watch"]
SIZING = [(1, 1.0), (2, 0.5)]
DEPTH = [0.12, 0.15]
TP = [0.03, 0.05]
CD = [12, 24]
HM = [48, 96]


def bmask(U_, M, k, scores, force=()):
    R = G2.rank365(U_)
    first = np.where(~U_["idx"].to_period("M").duplicated())[0]
    m = np.zeros(R.shape)
    for j, t0 in enumerate(first):
        t1 = first[j + 1] if j + 1 < len(first) else len(U_["idx"])
        elig = R[t0] <= M
        sc = np.where(np.isnan(scores[t0]), 0.0, scores[t0])
        cand = np.where(elig)[0]
        best = cand[np.argsort(-sc[cand], kind="stable")[:k]]
        m[t0:t1, best] = 1.0
    for s in force:
        if s in U_["syms"]:
            m[:, U_["syms"].index(s)] = 1.0
    return np.ascontiguousarray(m)


if __name__ == "__main__":
    out = os.path.join(X.OUT, "grid5.csv")
    U_ = G2.build2()
    G2.rank365(U_)
    rows = []
    t0 = time.time()
    for depth, tp, cd, hm in itertools.product(DEPTH, TP, CD, HM):
        sc = X.standalone_scores(U_, depth, tp, hm, cd)
        for (un, M, k), mode, (N, fmax) in itertools.product(UNIVS5, MODES, SIZING):
            m = bmask(U_, M, k, sc)
            P, sl, mm, i19, cols = X.sub(U_, m)
            e, lg = X.urun(P, U_["idx"], sl, smult=mm, N=N, fmax=fmax, depth=depth, tp=tp, hm=hm, cd=cd,
                           watch=(mode == "watch"))
            r = X.report(e, lg, U_["idx"], i19)
            r.update(univ=un, mode=mode, N=N, fmax=fmax, depth=depth, tp=tp, cd=cd, hm=hm)
            rows.append(r)
        del sc
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"depth {depth} tp {tp} cd {cd} hm {hm}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    G5 = pd.DataFrame(rows)
    c = ["univ", "mode", "N", "depth", "tp", "cd", "hm", "24-26_mo", "24-26_dd", "25-26_mo", "21-23_mo", "21-23_dd", "y2022"]
    ok = G5[(G5.y2022 >= 0) & (G5["21-23_dd"] >= -0.5)]
    print("RULE A (2021-23 only):")
    print(ok.sort_values(["21-23_mo", "21-23_dd"], ascending=[False, False])[c].head(3).to_string())
    print("HINDSIGHT best 2024-26:")
    print(G5.sort_values("24-26_mo", ascending=False)[c].head(5).to_string())
