"""HINDSIGHT ONLY (no pick was made by the pre-registered rule): look at the best-scoring grid design vs 90% mode."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_second_lib as L, m11_fusion_lib as F
D = dict(F.MODE90, sx=.07, stp=.04, shm=96, sN=72, sblk=1)
def line(r, ks): return " | ".join(f"{k}: {r[k+'_wr']*100:.1f}/{r[k+'_mo']*100:.2f}/{r[k+'_dd']*100:.1f}/{r[k+'_tpm']:.1f}" for k in ks)
for name, d in (("90%", F.MODE90), ("SC", D)):
    for lab, kw, ks in (("normal", {}, ("d", "t", "s")), ("worse", dict(worse=True), ("d", "t", "s")),
                        ("hold", dict(holdout=True), ("h",)), ("hold worse", dict(holdout=True, worse=True), ("h",)),
                        ("stress", dict(stress=True), ("d", "t"))):
        r, e, lg = L.evaluate_sc(d, **kw)
        print(name, lab, line(r, ks), "sc trades", int((lg.tier == 1).sum()))
