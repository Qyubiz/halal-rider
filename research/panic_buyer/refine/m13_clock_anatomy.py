"""m13 clock anatomy: 90% mode trades 2021-23 only (normal fills) by entry-hour bucket (6h, UTC) and weekday.
Also engine proof: clock off must equal m12 ek 90% mode (baseline table of M12_PREREG.md)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m13/clock"))
import numpy as np, pandas as pd
import m13_clock_lib as C, m12_entry_lib as L, m10_velocity_lib as V
syms, idx, P = V.market(False)
e, lg, inv = C.erun(P, idx, {}, {"typ": "none"})
tb = idx[lg.eb.astype(int).values]
sel = (tb >= L.T21) & (tb < L.T24)
x = lg[sel].copy(); tb = tb[sel]
x["h"] = tb.hour; x["wd"] = tb.dayofweek; x["win"] = x.ret > 0
print("n", len(x), "wr", x.win.mean(), "avg", x.ret.mean())
x["hb"] = (x.h // 4) * 4
print(x.groupby("hb").agg(n=("ret", "size"), wr=("win", "mean"), avg=("ret", "mean"), tm=("why", "mean")).round(4))
print(x.groupby("wd").agg(n=("ret", "size"), wr=("win", "mean"), avg=("ret", "mean"), tm=("why", "mean")).round(4))
print(x.groupby("h").agg(n=("ret", "size"), wr=("win", "mean"), avg=("ret", "mean")).round(3).T.to_string())
