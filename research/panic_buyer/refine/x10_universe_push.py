"""x10 lever "universe", round 3: HINDSIGHT push toward 10%/month.

Written after rounds 1-2 were seen; every number from this round is hindsight. Purpose: find the highest
2024-26 %/month any variant reaches with the universe options that did best in rounds 1-2 (the 19 coins, the
top 10 by 365-day volume, and the tiered 365-day universes), widening the other settings into aggressive corners.
The pre-registered rule A (2022 >= 0%, 2021-23 drop >= -50%, best 2021-23 %/month) is also applied to this grid
and reported, but it is NOT an honest pick because the grid itself was placed with hindsight.

Tiered universe "L{M}T[x{extra}][s{size}][c{core}]": the top `core` coins by 365-day volume at the normal depth
and full size; ranks core+1..M at depth + extra and `size` x the order (defaults: core 10, extra 6 points,
size 0.5).

GRID (1,536 variants): universe {U19, L10, L20, L20T, L30T, L50T, L30Tx8, L30Ts1} x mode {bid, watch}
  x sizing {N1 x 100%, N2 x 50%} x depth {12, 15%} x tp {+3, +5, +8%} x cooldown {0, 3, 12, 24h}
  x time stop {48, 96h}.
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
import x10_universe_round2 as G2  # noqa: E402

UNIVS3 = ["U19", "L10", "L20", "L20T", "L30T", "L50T", "L30Tx8", "L30Ts1"]
MODES = ["bid", "watch"]
SIZING = [(1, 1.0), (2, 0.5)]
DEPTH = [0.12, 0.15]
TP = [0.03, 0.05, 0.08]
CD = [0, 3, 12, 24]
HM = [48, 96]


def tier_setup(U_, univ, depth, force=()):
    m_ = re.fullmatch(r"L(\d+)T(?:x(\d+))?(?:s([\d.]+))?(?:c(\d+))?", univ)
    M = int(m_.group(1)); extra = int(m_.group(2) or 6) / 100; size = float(m_.group(3) or 0.5)
    core = int(m_.group(4) or 10)
    R = G2.rank365(U_)
    tier = (R > core) & (R <= M)
    m = (R <= core) * 1.0 + tier * size
    dm = np.where(tier, depth + extra, depth)
    for s in force:
        if s in U_["syms"]:
            j = U_["syms"].index(s)
            m[:, j] = 1.0; dm[:, j] = depth
    return np.ascontiguousarray(m), dm


def run_one3(U_, univ, mode, N, fmax, depth, tp, cd, force=(), pen=0.0, wslip=0.0, worder=0, hm=48):
    """Any universe of rounds 1-3. Returns e, lg, i19, cols."""
    if "T" in univ and univ.startswith("L"):
        m, dm = tier_setup(U_, univ, depth, force)
    elif univ.startswith("L"):
        m, dm = G2.setup2(U_, univ, depth, force)
    else:
        m = G.umask(U_, univ, None, force); dm = np.full(m.shape, depth)
    P, sl, mm, i19, cols = X.sub(U_, m)
    e, lg = X.urun(P, U_["idx"], sl, smult=mm, N=N, fmax=fmax, depth=np.ascontiguousarray(dm[:, cols]), tp=tp,
                   hm=hm, cd=cd, pen=pen, watch=(mode == "watch"), wslip=wslip, worder=worder)
    return e, lg, i19, cols


if __name__ == "__main__":
    out = os.path.join(X.OUT, "grid3.csv")
    U_ = G2.build2()
    G2.rank365(U_)
    rows = []
    t0 = time.time()
    for univ in UNIVS3:
        for mode, (N, fmax), depth, tp, cd, hm in itertools.product(MODES, SIZING, DEPTH, TP, CD, HM):
            e, lg, i19, _ = run_one3(U_, univ, mode, N, fmax, depth, tp, cd, hm=hm)
            r = X.report(e, lg, U_["idx"], i19)
            r.update(univ=univ, mode=mode, N=N, fmax=fmax, depth=depth, tp=tp, cd=cd, hm=hm)
            rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"{univ}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    print("grid3 done", len(rows))
