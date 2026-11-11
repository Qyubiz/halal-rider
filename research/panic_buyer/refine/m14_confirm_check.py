import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import m14_confirm_lib as C, m11_fusion_lib as F, m10_velocity_lib as V
for name, kw in (("normal", {}), ("worse", dict(pen=.002, pens=.002)), ("stress", dict(mtp_now=True, cd_add=1, pen=.002, pens=.002, liquidate=True))):
    dead = name == "stress"
    for hold in (False, True):
        syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
        e1, l1, _ = F.frun(P, idx, F.MODE90, **kw)
        e2, l2, _ = C.crun(P, idx, C.BASE, **kw)
        print(name, "hold" if hold else "main", "maxdiff", float(np.abs(e1.values - e2.values).max()), len(l1), len(l2))
