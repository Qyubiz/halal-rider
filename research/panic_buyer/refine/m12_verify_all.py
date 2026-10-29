"""m12 verify (agent C): tasks 1-3 and 5. Usage: python3 m12_verify_all.py base|A|B|nearA"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m12_sizing_lib as S, m12_entry_lib as L, m10_velocity_lib as V

def row(r, ps):
    return "  ".join(f"{p}:{r[p+'_wr']*100:5.1f}/{r[p+'_mo']*100:+6.2f}/{r[p+'_dd']*100:6.1f}" for p in ps)

def tab(name, f):
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r = f(lab, kw, False)
        ps = ("d", "t", "s") if lab == "stressed" else ("d", "t", "s")
        s = row(r, ps)
        if lab != "stressed":
            s += "  " + row(f(lab, kw, True), ("h",))
        print(f"{name:8s} {lab:8s} {s}", flush=True)

mode = sys.argv[1]
if mode == "base":
    tab("m10", lambda l, kw, h: V.evaluate(V.MODE90, holdout=h, **kw)[0])
    tab("sizing", lambda l, kw, h: S.evaluate({}, holdout=h, **kw)[0])
    tab("entry", lambda l, kw, h: L.evaluate({}, holdout=h, **kw)[0])
elif mode == "A":
    d = {"g1": 0.15, "gf": 0.5, "nbid": 1, "nslots": 2, "vmin": 0.25, "vsrc": "m168", "vt": 0.016653}
    tab("A", lambda l, kw, h: S.evaluate(d, holdout=h, **kw)[0])
    r, e, lg = S.evaluate(d, stress=True); x = e[e.index < S.T24]
    print("A stressed 2021-23 drop", (x / x.cummax() - 1).min())
elif mode == "B":
    d = {"anchor": "c", "cd": 0, "depth": 0.15, "dmaj": 0.14, "guard": 0.35, "hm": 96, "near": 0.01, "rank": "d7deep", "tp": 0.03, "vhi": 0.25, "vk": 0.0, "vlo": 0.1}
    tab("B", lambda l, kw, h: L.evaluate(d, holdout=h, **kw)[0])
    r, e, lg = L.evaluate(d, stress=True)
    print(r["d_wr"], r["d_mo"], r["d_dd"], r["d_mult"])
    x = e[e.index < L.T24]; dd = x / x.cummax() - 1
    print("min dd date", dd.idxmin(), dd.min(), "equity min", x.min())
    print(lg.columns.tolist())
    syms, idx, P = V.market(True)
    lg = lg.copy()
    lg["t_in"] = idx[lg.eb.astype(int)] if "eb" in lg else None
    cols = [c for c in lg.columns]
    print(lg[(lg.eb < len(idx)) & (idx[lg.eb.clip(0, len(idx)-1).astype(int)] < pd.Timestamp("2024-01-01"))].assign(t=lambda q: idx[q.eb.astype(int)]).nsmallest(5, "ret" if "ret" in lg else cols[-1]).to_string())
    print("symbols sample", list(P["o"].columns)[:3])
elif mode == "nearA":
    d = {"g1": 0.15, "gf": 0.5, "nbid": 1, "nslots": 2, "vmin": 0.25, "vsrc": "m168", "vt": 0.016653}
    vars_ = {"base": ({}, {}), "cd-1": ({}, dict(cd_add=-1)), "cd+1": ({}, dict(cd_add=1)), "mtp_now": ({}, dict(mtp_now=True)),
             "bot(mtp+cd1)": ({}, dict(mtp_now=True, cd_add=1)), "start+2w": ({}, "late"),
             "vt0.8": (dict(vt=0.016653*0.8), {}), "vt1.2": (dict(vt=0.016653*1.2), {})}
    pan = {False: V.market(False), True: V.market_holdout(False)}
    for nm, (dd_, kw) in vars_.items():
        dz = dict(d, **dd_)
        res = {}
        for hold in (False, True):
            syms, idx, P = pan[hold]
            for w in (False, True):
                k = {} if kw == "late" else dict(kw)
                if kw == "late": k["t0"] = idx[0] + pd.Timedelta(days=14)
                if w: k.update(pen=S.PEN, pens=S.PEN)
                e, lg, inv = S.srun(P, idx, dz, **k)
                res[(hold, w)] = V.summarize(e, lg, inv, idx, V.HOLD if hold else V.MAIN)
        n, h, wn = res[(False, False)], res[(True, False)], res[(False, True)]; wh = res[(True, True)]
        i1 = n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35
        i2 = h["h_mo"] > .1082 and h["h_wr"] >= .925 and h["h_dd"] >= -.663
        i3 = wn["t_mo"] > .0531 and wh["h_mo"] > .0741 and wn["t_wr"] >= .92 and wh["h_wr"] >= .92
        print(f"{nm:13s} t {n['t_wr']*100:.1f}/{n['t_mo']*100:+.2f}/{n['t_dd']*100:.1f}  h {h['h_wr']*100:.1f}/{h['h_mo']*100:+.2f}/{h['h_dd']*100:.1f}  worse t {wn['t_mo']*100:+.2f} h {wh['h_mo']*100:+.2f} wr {wn['t_wr']*100:.1f}/{wh['h_wr']*100:.1f}  items1-3 {i1,i2,i3}", flush=True)
