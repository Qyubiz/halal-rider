"""Engine proof: sx=0 reproduces 90% mode (M12_PREREG baseline) and equals m11 fk bit for bit."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import m14_second_lib as L, m11_fusion_lib as F, m10_velocity_lib as V
d = F.MODE90
for hold in (False, True):
    for kw in ({}, dict(worse=True)):
        r, e, lg = L.evaluate_sc(d, holdout=hold, **kw)
        r0, e0, lg0 = F.evaluate(d, holdout=hold, **kw)
        keys = ("h",) if hold else ("d", "t", "s")
        print("hold" if hold else "main", kw, "same" if np.array_equal(e.values, e0.values) else "DIFF",
              " ".join(f"{k}: {r[k+'_wr']*100:.1f}/{r[k+'_mo']:.2f}/{r[k+'_dd']*100:.1f}" for k in keys))
# with SC on: sanity
r, e, lg = L.evaluate_sc(dict(d, sx=0.08, stp=0.03, shm=72, sN=48))
print("SC on", {k: round(v, 3) for k, v in r.items() if k.startswith("d_") and k[2:] in ("wr", "mo", "dd", "tpm")}, "sc trades", int((lg.tier == 1).sum()), "of", len(lg))
