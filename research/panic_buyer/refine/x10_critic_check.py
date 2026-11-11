"""x10 critic, code check (not a result).
1. xkernel with lat=mlat=watch=0 must reproduce x10_velocity_lib.vkernel exactly (same trade log, same equity) for the
   reference, the velocity picks and the hindsight designs, on 19 and 22 coins, normal and stressed costs.
2. Minute-level watch mode with hourly cycles and positions that never compete for cash (N 19, fmax 0.05) must give
   the same trades as the universe agent's hourly watch kernel (x10_universe_lib.ukernel watch=True) on the hourly
   candles built from the same minutes: with no competition, the in-hour crossing order cannot matter.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_critic_lib as X  # noqa: E402
import x10_velocity_lib as V  # noqa: E402
import x10_universe_lib as U  # noqa: E402

R1 = dict(N=2, fmax=0.5, cyc=1, rearm=1, cdm=-1, cd=24, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1)
R2 = dict(N=2, fmax=0.5, cyc=15, rearm=1, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1, hm=48, look2=3, depth2=0.08,
          cdm=-1, cd=24)
HB = dict(cyc=1, rearm=1, N=1, fmax=1.0, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1, hm=24, look2=3, depth2=0.08,
          cdm=0, redip=0.03, rewin=60, cdl=1440)
SETS = [("live", {}), ("R1 pick", R1), ("R2 pick", R2), ("hindsight best", HB),
        ("hindsight best, fills 0.2% worse", dict(HB, pen=0.002, spen=0.002)),
        ("N1 tp5 depth14 hourly", dict(N=1, fmax=1.0, depth=0.14, tp=0.05)),
        ("N1 1min tp3 d30 cd0", dict(N=1, fmax=1.0, cyc=1, rearm=1, cdm=0, tp=0.03, delay=30))]


def main():
    t0 = time.time()
    A, hidx, Ph, SLIP = V.load()
    print(f"loaded {A['c'].shape} in {time.time()-t0:.0f}s")
    cols = ["eb", "xb", "k", "why", "ret", "amt", "entry", "xm", "fm", "eqb"]
    for K in (19, 22):
        for name, kw in SETS:
            ve, vl, vi = V.vrun(A, hidx, Ph, SLIP, K=K, **kw)
            xe, xl, xi = X.xrun(A, hidx, Ph, SLIP, K=K, **kw)
            same_log = vl.shape[0] == xl.shape[0] and np.array_equal(vl[cols].values, xl[cols].values)
            rel = float(np.nanmax(np.abs(xe.values / ve.values - 1)))
            print(f"  K={K} {name:36s} trades {len(vl):5d} vs {len(xl):5d} log identical {same_log} max rel eq diff "
                  f"{rel:.1e} final x{ve.iloc[-1]:.6g} vs x{xe.iloc[-1]:.6g}", flush=True)
            assert same_log and rel == 0.0, name

    print("\nminute watch (cyc 60) vs universe hourly watch kernel, no cash competition (N 19, fmax 0.05)")
    K = 19
    P19 = {f: v.iloc[:, :K] for f, v in Ph.items()}
    n = len(hidx)
    for g in (0.0, 0.35):
        ref = X.guard_ref(Ph, g, K) if g > 0 else None
        sm = np.ones((n, K)) if g == 0 else (~np.isnan(ref)).astype(float)
        for kw in (dict(depth=0.14, tp=0.05, hm=48, cd=24), dict(depth=0.12, tp=0.015, hm=48, cd=24)):
            ue, ul = U.urun(P19, hidx, SLIP[:, :K], N=19, fmax=0.05, smult=sm, watch=True, **kw)
            ul = ul[ul.xb >= 0]
            xe, xl, _ = X.xrun(A, hidx, Ph, SLIP, K=K, ref=ref, N=19, fmax=0.05, watch=1, cyc=60, **kw)
            a = ul[["eb", "xb", "k", "entry"]].values; b = xl[["eb", "xb", "k", "entry"]].values
            sa = a[np.lexsort(a.T[::-1])]; sb = b[np.lexsort(b.T[::-1])]
            ok = a.shape == b.shape and np.allclose(sa, sb, rtol=1e-12, atol=0)
            rel = float(np.nanmax(np.abs(xe.values / ue.values - 1)))
            print(f"  guard {g} {kw}: trades {len(ul)} vs {len(xl)}; same trades {ok}; max rel eq diff {rel:.1e}; "
                  f"final x{ue.iloc[-1]:.6g} vs x{xe.iloc[-1]:.6g}", flush=True)
            assert ok and rel < 1e-9
    print("CRITIC KERNEL CHECK PASSED")


if __name__ == "__main__":
    main()
