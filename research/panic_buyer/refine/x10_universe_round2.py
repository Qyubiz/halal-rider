"""x10 lever "universe", round 2: established-coin universes ranked by LONG-window volume, and a tiered universe.

PRE-REGISTRATION (written 22 Oct 2026 before this round was run)
----------------------------------------------------------------
Designed AFTER seeing part of round 1 (x10_universe_grid.py), including 2024-26 numbers for depth 10% / tp 1.5%
and a diagnosis of the losses: top-M by 30-day volume brings in "hot" small coins whose 12% dips keep falling
(48h time-stop losses of -40 to -60%, LUNA -100%, a few positions stuck at delisting); with 1-2 large orders
that wipes out the account. So this round is NOT independent of 2024-26: treat its picks as semi-honest.

Idea: a point-in-time stand-in for "the big established coins": rank screened coins at each month's first hour
by quote volume over the trailing 365 days (or as much history as exists; 30d+ history; trading in the previous
hour). A coin listed recently has little 365-day volume and ranks low, so new hot coins stay out.

GRID (1,080 variants; hm 48h; same costs as round 1):
  universe (5): L10, L15, L20, L30   monthly top 10/15/20/30 by trailing-365-day volume
                L30T  tiered: top 10 by 365d volume at the normal depth and full size; ranks 11-30 at
                      depth + 6 points and half size
  mode (2): bid, watch      sizing (4): N1 x 100%, N2 x 50%, N3 x 33%, N5 x 20%
  depth (3): 10%, 12%, 15%  take-profit (3): +1.5%, +3%, +5%  cooldown (3): 6h, 12h, 24h

SELECTION (2021-23 columns only, normal fills), same rules as round 1:
  Pick A2: keep 2022 >= 0% and 2021-23 worst drop >= -50%; highest 2021-23 %/month (tie: shallower drop).
  Pick B2: same with 2021-23 worst drop >= -35%.
Reports as in round 1 (x10_universe_pick.py handles both rounds).
"""
import hashlib
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_universe_lib as X  # noqa: E402

UNIVS2 = ["L10", "L15", "L20", "L30", "L30T"]
MODES = ["bid", "watch"]
SIZING = [(1, 1.0), (2, 0.5), (3, 1 / 3), (5, 0.2)]
DEPTH = [0.10, 0.12, 0.15]
TP = [0.015, 0.03, 0.05]
CD = [6, 12, 24]
HM = 48
TIER_EXTRA, TIER_SIZE, TIER_CORE = 0.06, 0.5, 10
_r365 = {}


def rank365_all(stress):
    """Monthly rank by trailing-365-day volume over ALL screened coins: (DataFrame (n, coins), coins ever <= 30)."""
    cp = os.path.join(X.OUT, f"rank365_{'stress' if stress else 'base'}.pkl")
    if os.path.exists(cp):
        return pd.read_pickle(cp)
    Dq = X.load_univ(stress, cols=("c", "qv"))
    s_all = list(Dq)
    idx = pd.date_range(min(d.index[0] for d in Dq.values()), max(d.index[-1] for d in Dq.values()), freq="1h")
    Pq = {f: pd.DataFrame({s: Dq[s][f] for s in s_all}).reindex(idx) for f in ("c", "qv")}
    del Dq
    R = pd.DataFrame(X.month_rank(Pq, look_h=365 * 24, min_hist_h=720), idx, s_all)
    del Pq
    ever = [s for s in s_all if R[s].min() <= 30]
    pd.to_pickle((R, ever), cp)
    return R, ever


def build2(stress=False):
    """Round-1 panel plus every coin that was ever in the 365-day-volume top 30."""
    _, ever = rank365_all(stress)
    return X.build(stress, extra=tuple(sorted(ever)))


def rank365(U_):
    """(n, K) monthly rank by trailing-365-day volume, mapped to U_'s columns."""
    key = bool(U_["stress"])
    if key not in _r365:
        R, ever = rank365_all(key)
        assert R.index.equals(U_["idx"])
        missing = [s for s in ever if s not in U_["syms"]]
        assert not missing, f"coins ranked <= 30 missing from the panel: {missing}"
        _r365[key] = np.ascontiguousarray(R[U_["syms"]].values)
    return _r365[key]


def setup2(U_, univ, depth, force=()):
    """(smult, depth matrix) for a round-2 universe."""
    R = rank365(U_)
    n, K = R.shape
    dm = np.full((n, K), depth)
    if univ == "L30T":
        core = R <= TIER_CORE
        tier = (R > TIER_CORE) & (R <= 30)
        m = core * 1.0 + tier * TIER_SIZE
        dm = np.where(tier, depth + TIER_EXTRA, depth)
    else:
        m = (R <= int(univ[1:])).astype(float)
    for s in force:
        if s in U_["syms"]:
            m[:, U_["syms"].index(s)] = 1.0
            dm[:, U_["syms"].index(s)] = depth
    return np.ascontiguousarray(m), dm


def run_one2(U_, univ, mode, N, fmax, depth, tp, cd, force=(), pen=0.0, wslip=0.0, worder=0, hm=HM):
    m, dm = setup2(U_, univ, depth, force)
    P, sl, mm, i19, cols = X.sub(U_, m)
    e, lg = X.urun(P, U_["idx"], sl, smult=mm, N=N, fmax=fmax, depth=np.ascontiguousarray(dm[:, cols]), tp=tp,
                   hm=hm, cd=cd, pen=pen, watch=(mode == "watch"), wslip=wslip, worder=worder)
    return e, lg, i19, cols


if __name__ == "__main__":
    with open(os.path.join(X.OUT, "prereg2_sha.txt"), "w") as f:
        f.write(hashlib.sha256(__doc__.encode()).hexdigest() + "\n")
    out = os.path.join(X.OUT, "grid2.csv")
    U_ = build2()
    R = rank365(U_)
    first = ~U_["idx"].to_period("M").duplicated()
    for M in (10, 20, 30):
        sel = (R <= M)[first]
        print(f"L{M}: on average {sel[:, U_['is19']].sum(1).mean():.1f} of the {M} are live coins; "
              f"coins ever included {int(sel.any(0).sum())}")
    rows = []
    t0 = time.time()
    for depth, tp, cd in itertools.product(DEPTH, TP, CD):
        for univ, mode, (N, fmax) in itertools.product(UNIVS2, MODES, SIZING):
            e, lg, i19, _ = run_one2(U_, univ, mode, N, fmax, depth, tp, cd)
            r = X.report(e, lg, U_["idx"], i19)
            r.update(univ=univ, mode=mode, N=N, fmax=fmax, depth=depth, tp=tp, cd=cd, hm=HM)
            rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"depth {depth} tp {tp} cd {cd}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    print("grid2 done", len(rows))
