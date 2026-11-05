"""m15 earlycut: prove ek with fam=0 == 90% mode (m11 fk) in every run used (normal/worse/stressed, main/holdout)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import m15_earlycut_lib as L
import m11_fusion_lib as F
for kw in (dict(), dict(worse=True), dict(stress=True), dict(holdout=True), dict(holdout=True, worse=True)):
    r1, e1, g1 = L.evaluate(L.OFF, **kw); r2, e2, g2 = F.evaluate(F.MODE90, **kw)
    print(kw, "max eq diff", float(np.max(np.abs(e1.values - e2.values))), "trades", len(g1), len(g2),
          " ".join(f"{p}:{r1[p+'_wr']*100:.1f}/{r1[p+'_mo']*100:+.2f}/{r1[p+'_dd']*100:.1f}" for p in "dtsh" if p+"_wr" in r1))
