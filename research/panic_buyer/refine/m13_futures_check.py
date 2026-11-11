"""m13 futures: (1) frun with the signal off == m10_velocity_lib.vrun 90% mode, every run type; (2) 90% mode table;
(3) futures data coverage per coin (first hour a signal value is known).
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... PB_FUT=... NUMBA_NUM_THREADS=1 python3 m13_futures_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m13_futures_lib as F  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

VKW = dict(tiers=[dict(ref=24, depth=0.14, tp=0.025, hm=96, cdw=3, cdl=3)], nslots=1, excl=1, dyn=0, guard=0.35)


def same(a, b):
    return np.allclose(a[0].values, b[0].values, rtol=1e-12, atol=0) and len(a[1]) == len(b[1])


def main():
    for lab, extra, hold, dead in (("normal", {}, False, False), ("worse", dict(pen=F.PEN, pens=F.PEN), False, False),
                                   ("stressed", None, False, True), ("holdout", {}, True, False),
                                   ("holdout worse", dict(pen=F.PEN, pens=F.PEN), True, False)):
        syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
        k = V.stress_kw(VKW) if extra is None else dict(VKW, **extra)
        a = V.vrun(P, idx, **k)
        b = F.frun(P, idx, {}, **{kk: k[kk] for kk in ("pen", "pens", "mtp_now", "cd_add", "liquidate") if kk in k})
        print(f"{lab:14s} vrun x{a[0].iloc[-1]:.6g} frun x{b[0].iloc[-1]:.6g} {'IDENTICAL' if same(a, b) else 'DIFFERENT'}")
    print("\n90% mode (win / %/month / worst drop / worst trade % acct / trades per month)")
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, _, _ = F.evaluate({}, **kw)
        if lab != "stressed":
            r.update(F.evaluate({}, holdout=True, **kw)[0])
        for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
            if f"{p}_wr" in r:
                print(f"  {lab:9s} {pl:10s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} "
                      f"{r[f'{p}_wt']*100:6.1f} {r[f'{p}_tpm']:5.1f}")
    if os.path.isdir(F.FUT):
        print("\ncoverage 2021-26 (first hour known)\n", F.coverage().to_string(index=False))
        print("\ncoverage 2018-20 holdout\n", F.coverage(holdout=True).to_string(index=False))


if __name__ == "__main__":
    main()
