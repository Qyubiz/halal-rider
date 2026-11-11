"""m12 entry: check that ek (m12_entry_lib) reproduces m10_velocity_lib.vk / m11_fusion_lib.fk with the new switches
off (anchor 'c', scalar depth, rank 'dist'), then print 90% mode's table from ek.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... NUMBA_NUM_THREADS=1 python3 m12_entry_check.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_entry_lib as L  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m11_fusion_lib as F  # noqa: E402


def same(a, b):
    (e1, l1, _), (e2, l2, _) = a, b
    return np.allclose(e1.values, e2.values, rtol=1e-12, atol=0) and len(l1) == len(l2) and \
        np.allclose(l1.values[:, :8], l2.values[:, :8], rtol=1e-12, atol=0)


def main():
    other = dict(depth=0.155, tp=0.03, hm=72, cd=1)
    for name, d in (("90% mode", {}), ("d15.5 tp3 72h cd1", other)):
        dd = dict(L.BASE, **d)
        vkw = dict(tiers=[dict(ref=24, depth=dd["depth"], tp=dd["tp"], hm=dd["hm"], cdw=dd["cd"], cdl=dd["cd"])],
                   nslots=1, excl=1, dyn=0, guard=0.35)
        for lab, extra, hold, dead in (("normal", {}, False, False), ("worse", dict(pen=L.PEN, pens=L.PEN), False, False),
                                       ("stressed", None, False, True), ("holdout", {}, True, False),
                                       ("holdout worse", dict(pen=L.PEN, pens=L.PEN), True, False)):
            syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
            k = V.stress_kw(vkw) if extra is None else dict(vkw, **extra)
            a = V.vrun(P, idx, **k)
            ekw = {kk: k[kk] for kk in ("pen", "pens", "mtp_now", "cd_add", "liquidate") if kk in k}
            t0 = time.time(); b = L.erun(P, idx, d, **ekw); dt = time.time() - t0
            print(f"{name:18s} {lab:14s} vk x{a[0].iloc[-1]:.6g} ek x{b[0].iloc[-1]:.6g} trades {len(a[1])}/{len(b[1])} "
                  f"{'IDENTICAL' if same(a, b) else 'DIFFERENT'} ({dt:.2f}s)", flush=True)
    syms, idx, P = V.market(False)
    fd = dict(F.MODE90)
    a = F.frun(P, idx, fd, t0=L.LATE); b = L.erun(P, idx, {}, t0=L.LATE)
    print(f"90% mode start 2021-01-15 fk vs ek: {'IDENTICAL' if same(a, b) else 'DIFFERENT'}")
    print("\n90% mode through ek (win / %/month / worst drop / worst trade % acct / trades per month)")
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, _, _ = L.evaluate({}, **kw)
        rh, _, _ = L.evaluate({}, holdout=True, worse=kw.get("worse", False), stress=kw.get("stress", False))
        r.update(rh)
        for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
            print(f"  {lab:9s} {pl:10s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} "
                  f"{r[f'{p}_wt']*100:6.1f} {r[f'{p}_tpm']:5.1f}")
    t0 = time.time(); s = L.design_scores({}); print(f"\ndesign_scores(90% mode) {time.time()-t0:.1f}s:",
                                                      {k: round(v, 4) for k, v in s.items()})


if __name__ == "__main__":
    main()
