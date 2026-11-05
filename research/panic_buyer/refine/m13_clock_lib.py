"""m13 "clock" (arena contestant 2): UTC-clock rule on top of 90% mode (m12_entry_lib.ek engine, unchanged).
Clock design c = dict(typ, start, len, wk):
  typ  'none' | 'off' (no bid during the window) | 'shal' (bid depth 2pt shallower in the window: 12% instead of 14%)
       | 'deep' (depth 2pt deeper: 16%)
  window = UTC hours start, start+1, ... start+len-1 (mod 24)  (hour of the bar's open, known in advance)
  wk   'none' | 'off' | 'shal' | 'deep' applied to Saturday+Sunday bars (typ='none' only; weekend designs)
The mask is applied to the bid table LIM[t,k] only (NaN = no bid / different depth); fills, costs, exits as 90% mode.
Not tested: time stop by entry hour (engine takes one scalar time stop).
"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m12_entry_lib as L  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
E = L.E
DEL = {"shal": -0.02, "deep": 0.02}


def hours_in(h, start, ln):
    return ((h - start) % 24) < ln


def clock_lim(P, idx, d, c):
    dd = dict(L.BASE, **d)
    A = L.anchor(P, dd["anchor"])
    G = L.guard(P, dd["guard"])
    base = dd["depth"]
    D = np.full(A.shape, base)
    off = np.zeros(len(idx), bool)
    hr = idx.hour.values; wd = idx.dayofweek.values
    def apply(mask, typ):
        if typ == "off":
            off[mask] = True
        elif typ in DEL:
            D[mask, :] = base + DEL[typ]
    if c.get("typ", "none") != "none":
        apply(hours_in(hr, c["start"], c["len"]), c["typ"])
    if c.get("wk", "none") != "none":
        apply(wd >= 5, c["wk"])
    Lm = A * (1.0 - D)
    Lm = np.where(G > 0, Lm, np.nan)
    Lm[off, :] = np.nan
    return np.ascontiguousarray(Lm)


def erun(P, idx, d, c, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    dd = dict(L.BASE, **d)
    O = P["o"].values; K = O.shape[1]
    Lm = clock_lim(P, idx, d, c)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = L.ek(O, P["h"].values, P["l"].values, P["cf"].values, Lm, np.zeros((1, 1)), 0, 0.0, lastb,
                       float(dd["tp"]), int(dd["hm"]), max(int(dd["cd"]) + cd_add, 0), float(E.EPS), float(E.FEE), sl,
                       float(pen), float(pens), bool(mtp_now), i0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


def shifted(c, s):
    c = dict(c)
    if c.get("typ", "none") != "none":
        c["start"] = (c["start"] + s) % 24
    return c


def design_scores(c, shift=0):
    """2021-23 only. 5 versions x (normal, 0.2% worse). Clock window shifted by `shift` hours."""
    syms, idx, P = V.market(False)
    cc = shifted(c, shift); out = {}
    for tag, kw in L.VERS.items():
        for fl, fk in (("", {}), ("w", dict(pen=L.PEN, pens=L.PEN))):
            e, lg, inv = erun(P, idx, {}, cc, **kw, **fk)
            per = (("d", L.LATE, L.T24),) if tag == "ls" else L.DPER
            r = V.summarize(e, lg, inv, idx, per)
            t = fl + tag
            out[f"{t}_wr"], out[f"{t}_mo"], out[f"{t}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
            if t == "n": out["n_tpm"] = r["d_tpm"]
    vs = list(L.VERS)
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in vs]))
    out["wrobust"] = float(np.mean([out[f"w{t}_mo"] for t in vs]))
    out["rdd_med"] = float(np.median([out[f"{t}_dd"] for t in vs]))
    out["w_wr"], out["w_mo"], out["w_dd"] = out["wn_wr"], out["wn_mo"], out["wn_dd"]
    return out


def evaluate(c, worse=False, stress=False, holdout=False):
    syms, idx, P = V.market_holdout(stress) if holdout else V.market(stress)
    kw = {}
    if worse or stress: kw.update(pen=L.PEN, pens=L.PEN)
    if stress: kw.update(mtp_now=True, cd_add=1, liquidate=True)
    e, lg, inv = erun(P, idx, {}, c, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
