"""x10 lever "universe", round 4: HINDSIGHT fine search around the best round-3 region (watch mode, one coin at
a time, tiered universes). Everything here is hindsight: it was placed after seeing 2024-26 results.

Universes: L{M}T[x][s][c] as in round 3, plus "U19L{M}T": the 19 live coins as the core (normal depth, full size)
and every other coin in the monthly top M by 365-day volume as the tier (depth + 6 points, half size).
Optional crash guard "g{G}": a coin is skipped for the hour when its last close is more than G% below its highest
close of the previous 7 days (known at the hour's start).

GRID (864 variants): universe {L50T, L40T, L50Tx8, L50Tc15, U19L30T, U19L50T} x guard {none, 35%}
  x depth {14, 15, 17%} x tp {+4, +5, +6, +8%} x cooldown {6, 12, 24h} x time stop {48, 96h}; watch mode, N1 x 100%.
"""
import itertools
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_grid as G  # noqa: E402
import x10_universe_lib as X  # noqa: E402
import x10_universe_push as G3  # noqa: E402
import x10_universe_round2 as G2  # noqa: E402

UNIVS4 = ["L50T", "L40T", "L50Tx8", "L50Tc15", "U19L30T", "U19L50T"]
GUARD = [0.0, 0.35]
DEPTH = [0.14, 0.15, 0.17]
TP = [0.04, 0.05, 0.06, 0.08]
CD = [6, 12, 24]
HM = [48, 96]
_guard = {}


def guard_mask(U_, g):
    """(n, K) 1 = allowed: last close no more than g below the highest close of the previous 168 hours."""
    key = (id(U_), g)
    if key not in _guard:
        c = U_["P"]["cf"]
        hi = c.rolling(168, min_periods=1).max()
        ok = ((c / hi - 1) >= -g).shift(1, fill_value=True)
        _guard[key] = ok.values.astype(float)
    return _guard[key]


def setup4(U_, univ, depth, force=()):
    m_ = re.fullmatch(r"U19L(\d+)T", univ)
    if m_:
        R = G2.rank365(U_)
        core = np.repeat(U_["is19"][None, :], len(U_["idx"]), axis=0)
        tier = (R <= int(m_.group(1))) & ~core
        m = core * 1.0 + tier * 0.5
        dm = np.where(tier, depth + 0.06, depth)
        for s in force:
            if s in U_["syms"]:
                j = U_["syms"].index(s)
                m[:, j] = 1.0; dm[:, j] = depth
        return np.ascontiguousarray(m), dm
    return G3.tier_setup(U_, univ, depth, force)


def run_one4(U_, univ, mode, N, fmax, depth, tp, cd, force=(), pen=0.0, wslip=0.0, worder=0, hm=48, guard=0.0,
             start=None):
    """start: run from this timestamp only (fresh account), else from 2021."""
    if univ.startswith("U19L") or ("T" in univ and univ.startswith("L")):
        m, dm = setup4(U_, univ, depth, force)
    elif univ.startswith("L"):
        m, dm = G2.setup2(U_, univ, depth, force)
    else:
        m = G.umask(U_, univ, None, force); dm = np.full(m.shape, depth)
    if guard > 0:
        m = np.ascontiguousarray(m * guard_mask(U_, guard))
    P, sl, mm, i19, cols = X.sub(U_, m)
    idx = U_["idx"]
    dm = np.ascontiguousarray(dm[:, cols])
    if start is not None:
        i0 = int(np.searchsorted(idx, pd.Timestamp(start)))
        P = {f: v.iloc[i0:] for f, v in P.items()}
        sl, mm, dm, idx = sl[i0:], mm[i0:], dm[i0:], idx[i0:]
    e, lg = X.urun(P, idx, sl, smult=mm, N=N, fmax=fmax, depth=dm, tp=tp, hm=hm, cd=cd, pen=pen,
                   watch=(mode == "watch"), wslip=wslip, worder=worder)
    return e, lg, i19, cols


if __name__ == "__main__":
    out = os.path.join(X.OUT, "grid4.csv")
    U_ = G2.build2()
    G2.rank365(U_)
    rows = []
    t0 = time.time()
    for univ in UNIVS4:
        for g, depth, tp, cd, hm in itertools.product(GUARD, DEPTH, TP, CD, HM):
            e, lg, i19, _ = run_one4(U_, univ, "watch", 1, 1.0, depth, tp, cd, hm=hm, guard=g)
            r = X.report(e, lg, U_["idx"], i19)
            r.update(univ=univ + (f"+g{int(g * 100)}" if g else ""), mode="watch", N=1, fmax=1.0, depth=depth,
                     tp=tp, cd=cd, hm=hm)
            rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{univ}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    print("grid4 done", len(rows))
