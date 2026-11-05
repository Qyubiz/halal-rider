"""m15 nobounce: NO PICK under the written rule (0 of 96 designs pass). HINDSIGHT look (labelled) at the design with the
best 2021-23 robust score among those that only fail the worse-fill/near-version drop bars: mode 2, N=12, y=1%, tp2=0.3%.
Compared with 90% mode from the same engine."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m15_nobounce_lib as L, m10_velocity_lib as V
HS = dict(L.OFF, mode=2, N=12, y=0.01, tp2=0.003)
def run(d, **k):
    r, e, lg = L.evaluate(d, **k)
    syms, idx, P = V.market_holdout(k.get("stress", False)) if k.get("holdout") else V.market(k.get("stress", False))
    tb = idx[lg.eb.astype(int).values]; acct = lg.ret * lg.amt / lg.eq0
    per = V.HOLD if k.get("holdout") else V.MAIN
    out = {}
    for tag, a, b in per:
        sel = (tb >= a) & ((tb < b) if b is not None else True)
        x = acct[sel]; l = x[lg.ret[sel] <= 0]
        out[tag] = (r[f"{tag}_wr"]*100, r[f"{tag}_mo"]*100, r[f"{tag}_dd"]*100, r[f"{tag}_tpm"], l.mean()*100 if len(l) else 0, x.min()*100, int((lg.why[sel]==6).sum()))
    return out, e
for nm, d in (("90% mode", L.OFF), ("HINDSIGHT nobounce m2 N12 y1% tp2 0.3%", HS)):
    print(nm, " [win% / %mo / worst drop / trades/mo / avg loser % acct / worst trade % acct]")
    for lab, k in (("normal", {}), ("worse", dict(worse=True)), ("stress", dict(stress=True)), ("hold normal", dict(holdout=True)), ("hold worse", dict(holdout=True, worse=True))):
        o, e = run(d, **k)
        for tag, v in o.items():
            print(f"  {lab:12s} {tag}: " + " / ".join(f"{x:.2f}" for x in v))
        if nm.startswith("HIND") and lab == "normal": e.rename("equity").rename_axis("time").to_csv(os.path.join(L.F.OUT, "hindsight_eq_main.csv"))
        if nm.startswith("HIND") and lab == "hold normal": e.rename("equity").rename_axis("time").to_csv(os.path.join(L.F.OUT, "hindsight_eq_hold.csv"))
