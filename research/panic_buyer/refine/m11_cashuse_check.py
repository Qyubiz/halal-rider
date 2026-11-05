"""Check: m11_cashuse_lib.ck with the second engine off == m10_velocity_lib 90% mode (equity and trades), on the
19-coin panel, the 22-coin collapse panel and the 2018-20 holdout, with normal fills, 0.2% worse fills and the
stressed settings. Then print 90% mode through ck in the report format, and a sanity run with S2 on but never
allowed to bid (armx=1e8: every hour armed, pm=1) which must also equal 90% mode."""
import numpy as np
import m10_velocity_lib as V
import m11_cashuse_lib as C

for hold in (False, True):
    for dead in (False, True):
        syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
        for opt in ({}, dict(pen=0.002, pens=0.002), dict(mtp_now=True, cd_add=1, pen=0.002, pens=0.002)):
            e1, l1, _ = V.vrun(P, idx, **V.MODE90, liquidate=dead, **opt)
            e2, l2, _ = C.crun(P, idx, None, liquidate=dead, **opt)
            e3, l3, _ = C.crun(P, idx, dict(d2=0.05), armx=1e8, pm=1, liquidate=dead, **opt)
            same = np.allclose(e1.values, e2.values, rtol=0, atol=1e-12) and len(l1) == len(l2) and \
                np.allclose(l1.values[:, :8], l2.values[:, :8])
            same3 = np.allclose(e1.values, e3.values, rtol=0, atol=1e-12) and len(l1) == len(l3)
            print(f"{'holdout' if hold else '2021-26'} {'22' if dead else '19'} coins {opt}: vk x{e1.iloc[-1]:.6f} "
                  f"ck x{e2.iloc[-1]:.6f} S2-blocked x{e3.iloc[-1]:.6f} trades {len(l1)}/{len(l2)}/{len(l3)} "
                  f"{'IDENTICAL' if same and same3 else 'DIFFERENT'}")
print("\n90% mode through m11_cashuse_lib (second engine off)")
for tag, opt in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
    r, *_ = C.evaluate({}, **opt)
    for p in ("d", "t", "s"):
        print(C.fmt(tag, p, r))
    r, *_ = C.evaluate({}, holdout=True, **opt)
    print(C.fmt(tag, "h", r))
