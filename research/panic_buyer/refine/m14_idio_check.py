import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_idio_lib as L, m11_fusion_lib as F
for lab, kw in (("n", {}), ("w", dict(worse=True)), ("s", dict(stress=True))):
    r, e, lg = L.evaluate(L.OFF, **kw); r2, _, _ = F.evaluate(F.MODE90, **kw)
    for p in "dts":
        if f"{p}_mo" in r: print(lab, p, round(r[f"{p}_wr"]*100,1), round(r[f"{p}_mo"]*100,2), round(r[f"{p}_dd"]*100,1), "| fusion", round(r2[f"{p}_mo"]*100,2), round(r2[f"{p}_dd"]*100,1))
r, _, _ = L.evaluate(L.OFF, holdout=True); print("hold", r["h_wr"], r["h_mo"], r["h_dd"])
