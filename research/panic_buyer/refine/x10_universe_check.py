"""Proof that x10_universe_lib.ukernel (new switches off) reproduces engine.run exactly, and a description of the
universe panel. Run: PB_DATA=... PB_OUT=... python x10_universe_check.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import study as S  # noqa: E402
import x10_universe_lib as X  # noqa: E402

SETTINGS = [
    dict(),                                                       # live
    dict(N=2, fmax=0.5, tp=0.03, cd=12, depth=0.10),
    dict(N=1, fmax=1.0, tp=0.05, cd=6, depth=0.15, hm=96),
    dict(N=3, fmax=0.33, tp=0.015, cd=24, mtp_now=True),
    dict(N=5, fmax=0.2, part=0.5, tp2=0.04, cd=12),
    dict(N=4, fmax=0.25, part=0.5, trail=0.03, tpwait=2, tprest=False, pen=0.002),
]


def same(a_eq, a_lg, b_eq, b_lg):
    closed = b_lg[b_lg.xb >= 0][E.LOG_COLS].values
    return np.array_equal(a_eq.values, b_eq.values) and np.array_equal(a_lg[E.LOG_COLS].values, closed)


if __name__ == "__main__":
    ok = True
    for dead in (False, True):
        syms, idx, P = S.market(dead)
        for kw in SETTINGS:
            ea, la = E.run(P, idx, **kw)
            eb, lb = X.urun(P, idx, E.SLIP_MAJOR, **{"N": 8, "fmax": 0.2, **kw})
            r = same(ea, la, eb, lb)
            ok &= r
            print(f"19 coins dead={dead} {kw}: {'IDENTICAL' if r else 'DIFFERENT'} "
                  f"(x{ea.iloc[-1]:.4f} vs x{eb.iloc[-1]:.4f}, {len(la)} trades)")
    U_ = X.build()
    P, idx = U_["P"], U_["idx"]
    print(f"\nuniverse panel: {U_['n_screened']} screened coins with 30+ days, {len(U_['syms'])} ever in the monthly "
          f"top 50 or live; {len(idx)} hours {idx[0]} .. {idx[-1]}")
    vec = np.where(np.nanmin(U_["slip"], axis=0) < 0.0005, E.SLIP_MAJOR, E.SLIP_OTHER)   # a constant per-coin vector
    for M in (20, 50):
        sm = X.mask(U_, "top", M)
        for kw in SETTINGS[:3]:
            ea, la = E.run(P, idx, smult=sm, slip=vec, **kw)
            eb, lb = X.urun(P, idx, vec, smult=sm, **{"N": 8, "fmax": 0.2, **kw})
            r = same(ea, la, eb, lb)
            ok &= r
            print(f"universe top {M} {kw}: {'IDENTICAL' if r else 'DIFFERENT'} (x{ea.iloc[-1]:.4f} vs x{eb.iloc[-1]:.4f})")
    print("\nALL IDENTICAL" if ok else "\nMISMATCH")
    n_major = (U_["slip"] == E.SLIP_MAJOR).sum(axis=1)
    print(f"coins in the 0.02% slippage tier per hour: median {np.median(n_major):.0f}")
    for M in (20, 30, 50):
        m = X.mask(U_, "top", M)
        in19 = (m[:, U_["is19"]].sum(axis=1))
        print(f"top {M}: of the {M} monthly picks, on average {in19.mean():.1f} are live coins")
