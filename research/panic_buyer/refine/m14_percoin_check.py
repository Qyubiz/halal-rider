"""m14 percoin: ek2 with default tables == 90% mode (baseline table of M12_PREREG)."""
import numpy as np, m14_percoin_lib as L, m10_velocity_lib as V
for hold in (False, True):
    syms, idx, P = V.market_holdout(False) if hold else V.market(False)
    n, K = P["o"].shape
    DEP = np.full((n, K), 0.14); TP = np.full((n, K), 0.025)
    for w, kw in (("normal", {}), ("worse", dict(pen=0.002, pens=0.002))):
        e, lg, inv = L.prun(P, idx, DEP, TP, **kw)
        r = V.summarize(e, lg, inv, idx, V.HOLD if hold else V.MAIN)
        print(hold, w, {k: round(v, 4) for k, v in r.items() if k.endswith(("wr", "mo", "dd"))})
