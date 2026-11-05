"""m16 riskmodel, step 1: (a) sk with SZ = 1 reproduces m12 ek (== 90% mode) exactly on every panel and fill
setting, and the M12_PREREG baseline table; (b) builds the model panel (cached)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m16_riskmodel_lib as R  # noqa: E402
import m12_entry_lib as L  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

CASES = {"normal": {}, "worse": dict(pen=R.PEN, pens=R.PEN),
         "stress": dict(pen=R.PEN, pens=R.PEN, mtp_now=True, cd_add=1, liquidate=True)}
for hold in (False, True):
    for name, kw in CASES.items():
        dead = name == "stress"
        syms, idx, P = V.market_holdout(dead) if hold else V.market(dead)
        e1, l1, _ = R.srun(P, idx, **kw)
        e2, l2, _ = L.erun(P, idx, {}, **kw)
        r = V.summarize(e1, l1, _, idx, V.HOLD if hold else V.MAIN)
        same = np.array_equal(e1.values, e2.values) and np.array_equal(l1.values, l2.values)
        print(("2018-20 " if hold else "2021-26 ") + f"{name:7s} identical={same}  " + "  ".join(
            f"{p}: {r[p+'_wr']*100:.1f}/{r[p+'_mo']*100:+.2f}/{r[p+'_dd']*100:.1f}" for p in (("h",) if hold else ("d", "t", "s"))),
            flush=True)
syms, idx, P, Z = R.build()
print("built", len(idx), len(syms), Z["X"].shape, int((Z["cand"] & Z["guard"]).sum()), "guarded candidates")
