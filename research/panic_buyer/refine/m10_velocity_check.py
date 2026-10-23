"""Check: m10_velocity_lib.vrun with one tier (24h reference) == w90_wr90_lib.wrun exactly (equity and trade log),
on the 19 coins and on the 22-coin panel with liquidation, for the 90% mode and other settings (incl. N=2/4 bids per
hour, real bot timing, fills 0.2% worse on buys and sells). Also prints the 90% mode baseline through vrun."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

CASES = [dict(N=1, fmax=1.0, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.35),
         dict(N=1, fmax=1.0, depth=0.13, tp=0.0125, hm=96, cd=3, guard=0.35),
         dict(N=2, fmax=0.5, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.25),
         dict(N=8, fmax=0.2, depth=0.12, tp=0.015, hm=48, cd=24, guard=0.0),
         dict(N=4, fmax=0.5, depth=0.16, tp=0.02, hm=96, cd=24, guard=0.35)]


def to_v(c):
    return dict(tiers=[dict(ref=24, depth=c["depth"], tp=c["tp"], hm=c["hm"], cdw=c["cd"], cdl=c["cd"], fs=c["fmax"])],
                nslots=99, nbid=c["N"], excl=1, dyn=0, guard=c["guard"])


def main():
    ok_all = True
    for dead in (False, True):
        syms, idx, P = V.market(dead)
        for c in CASES:
            for extra in ({}, dict(mtp_now=True, pen=0.002, pens=0.002)):
                lw = dict(liquidate=True) if dead else {}
                e1, l1 = W.wrun(P, idx, **c, **extra, **lw)
                e2, l2, _ = V.vrun(P, idx, **to_v(c), **extra, **lw)
                cols = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]
                ok = np.array_equal(e1.values, e2.values) and np.array_equal(l1[cols].values, l2[cols].values)
                ok_all &= ok
                print(f"{'22 coins' if dead else '19 coins'} {c} {extra}: wrun x{e1.iloc[-1]:.6f} vrun x{e2.iloc[-1]:.6f} "
                      f"trades {len(l1)}/{len(l2)} {'IDENTICAL' if ok else 'DIFFERENT'}", flush=True)
    print("ALL CHECKS PASS" if ok_all else "SOME CHECK FAILED")
    r, e, lg, inv = V.evaluate(V.MODE90)
    print(V.fmt("90% mode through vrun", r))


if __name__ == "__main__":
    main()
