import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m17_bigtp_lib as L
b = L.D()
for nm, d in (("fk 90%", b), ("ekernel 90%", L.D(part=0.9999, tp2=0.0))):
    for kw in ({}, {"worse": True}):
        r, e, lg = L.full(d, **kw)
        print(nm, kw, *(f"{r[p+'_wr']*100:.1f}/{r[p+'_mo']*100:+.2f}/{r[p+'_dd']*100:.1f}" for p in "dts"))
for kw in ({}, {"worse": True}):
    r, e, lg = L.full(b, holdout=True, **kw); print("hold", kw, f"{r['h_wr']*100:.1f}/{r['h_mo']*100:+.2f}/{r['h_dd']*100:.1f}")
r, e, lg = L.full(L.D(0.14,0.05,48,24)); print("10% mode", *(f"{r[p+'_wr']*100:.1f}/{r[p+'_mo']*100:+.2f}/{r[p+'_dd']*100:.1f}" for p in "dts"))
r, e, lg = L.full(L.D(0.14,0.05,48,24,part=0.9999)); print("10% mode ek", *(f"{r[p+'_wr']*100:.1f}/{r[p+'_mo']*100:+.2f}/{r[p+'_dd']*100:.1f}" for p in "dts"))
