"""m11 fusion: check that fk reproduces m10_velocity_lib.vk (and so 90% mode) with every new switch off.

Compares equity curves and trade logs for 90% mode and the m10 velocity pick (two tiers, re-arm after a win) under
normal fills, worse fills, the stressed run (+LUNA/FTT/SRM) and the 2018-20 holdout, then prints 90% mode's table.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... NUMBA_NUM_THREADS=1 python3 m11_fusion_check.py
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

U1 = dict(tiers=[dict(ref=24, depth=0.14, tp=0.025, hm=96, cdw=0, cdl=3),
                 dict(ref=12, depth=0.13, tp=0.03, hm=48, cdw=0, cdl=3)], nslots=1, excl=1, dyn=0, guard=0.35)


def to_f(kw):
    return dict(F.OFFRISK, tiers=[dict(F.TIERA, **{k: v for k, v in t.items() if k in F.TIERA}) for t in kw["tiers"]],
                guard=kw["guard"])          # vk nslots=1/nbid=99 == fk nbid=1/nslots=99 when fs=1


def main():
    for name, vkw in (("90% mode", V.MODE90), ("velocity U1", U1)):
        fd = to_f(vkw)
        for lab, extra, hold, dead in (("normal", {}, False, False), ("worse", dict(pen=F.PEN, pens=F.PEN), False, False),
                                       ("stressed", None, False, True), ("holdout", {}, True, False),
                                       ("holdout worse", dict(pen=F.PEN, pens=F.PEN), True, False)):
            syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
            k = V.stress_kw(vkw) if extra is None else dict(vkw, **extra)
            e1, l1, _ = V.vrun(P, idx, **k)
            fk = {kk: k[kk] for kk in ("pen", "pens", "mtp_now", "cd_add", "liquidate") if kk in k}
            t0 = time.time()
            e2, l2, _ = F.frun(P, idx, fd, **fk)
            dt = time.time() - t0
            same = np.allclose(e1.values, e2.values, rtol=1e-12, atol=0) and len(l1) == len(l2) and \
                np.allclose(l1.values, l2.values, rtol=1e-12, atol=0)
            print(f"{name:12s} {lab:14s} vk x{e1.iloc[-1]:.6g} fk x{e2.iloc[-1]:.6g} trades {len(l1)}/{len(l2)} "
                  f"{'IDENTICAL' if same else 'DIFFERENT'} ({dt:.2f}s)", flush=True)
    print("\n90% mode through fk (win / %/month / worst drop / worst trade % acct / trades per month)")
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, _, _ = F.evaluate(F.MODE90, **kw)
        rh, _, _ = F.evaluate(F.MODE90, holdout=True, worse=kw.get("worse", False), stress=kw.get("stress", False))
        r.update(rh)
        for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025"), ("h", "2018-20")):
            print(f"  {lab:9s} {pl:10s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} "
                  f"{r[f'{p}_wt']*100:6.1f} {r[f'{p}_tpm']:5.1f}")


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def guard_check():
    """fk's guardrails vs ens_lib.ekernel (hsmode 0 = hourly close stop, as guardrails.py) on 90% mode."""
    import ens_lib as EN
    syms, idx, P = V.market(False)
    for kw in (dict(hs=0.2), dict(killm=0.3), dict(cln=2, clT=48), dict(hs=0.15, zcap=0.05),
               dict(hs=0.2, killm=0.3, cln=2, clT=48)):
        e1, l1, _, _ = EN.erun(P, idx, [EN.MODE90], hsmode=0, **kw)
        e2, l2, _ = F.frun(P, idx, dict(F.MODE90, **kw))
        same = np.allclose(e1.values, e2.values, rtol=1e-9, atol=0) and len(l1) == len(l2)
        print(f"guardrails {kw}: ens x{e1.iloc[-1]:.6g} fk x{e2.iloc[-1]:.6g} trades {len(l1)}/{len(l2)} "
              f"{'IDENTICAL' if same else 'DIFFERENT'}", flush=True)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "guard":
    guard_check()
