"""Proof: B=1 engine == 90% mode (m11 fusion, MODE90), all runs bit-for-bit and vs M12_PREREG baseline table."""
import numpy as np
import m11_fusion_lib as F, m10_velocity_lib as V, m14_bars4h_lib as L
for lab, kw, fk in [("normal", {}, {}), ("worse", dict(pen=F.PEN, pens=F.PEN), {}), ("bot", dict(mtp_now=True, cd_add=1), {})]:
    for ho in (False, True):
        syms, idx, P = V.market_holdout(False) if ho else V.market(False)
        e0, lg0, _ = F.frun(P, idx, F.MODE90, **kw); e1, lg1, _ = L.run(P, idx, L.BASE, 1, **kw)
        print(lab, "hold" if ho else "main", "equity max abs diff", float(np.abs(e0.values - e1.values).max()), "trades", len(lg0), len(lg1))
syms, idx, P = V.market(True); e0, lg0, _ = F.frun(P, idx, F.MODE90, mtp_now=True, cd_add=1, pen=F.PEN, pens=F.PEN, liquidate=True)
e1, lg1, _ = L.run(P, idx, L.BASE, 1, mtp_now=True, cd_add=1, pen=F.PEN, pens=F.PEN, liquidate=True)
print("stress equity max abs diff", float(np.abs(e0.values - e1.values).max()))
row = lambda r, t: f"{r[t+'_wr']*100:5.1f} / {r[t+'_mo']*100:+6.2f} / {r[t+'_dd']*100:6.1f} / {r[t+'_tpm']:4.1f}"
for nm, kw in [("normal", {}), ("worse", dict(worse=True)), ("stress", dict(stress=True))]:
    r = L.evaluate(L.BASE, 1, **kw)[0]; print(nm, "d", row(r, "d"), "| t", row(r, "t"), "| s", row(r, "s"))
    if nm != "stress": h = L.evaluate(L.BASE, 1, holdout=True, **kw)[0]; print(nm, "h", row(h, "h"))
