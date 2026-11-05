"""m14 bars4h: the 90%-mode logic on B-hour candles built from the hourly data (aligned to 00:00 UTC).
bar o = first open, h = max high, l = min low, c = last close. The m11 fusion kernel (fk) runs unchanged on the bar
panel: trigger = highest close of the last Wb bars x (1-depth), bids re-placed once per bar, a limit fills when the bar's
LOW trades 0.1% through, take-profit/time stop/cooldown counted in bars, crash guard (35% under the highest close of the
last 168h = 168/B bars) measured on bar closes. B=1, Wb=24, cooldown 3 == 90% mode exactly (m14_bars4h_check.py).
Equity is expressed hourly (value at the bar's last hour, held through the bar) so summarize() works unchanged."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m11_fusion_lib as F, m10_velocity_lib as V, engine as E

OUT = F.OUT
_bp = {}

def bar_panel(P, B):
    k = (id(P["o"]), B)
    if k not in _bp:
        if B == 1:
            Q = {f: P[f] for f in ("o", "h", "l", "c", "cf")}
        else:
            r = lambda x: x.resample(f"{B}h")
            Q = {"o": r(P["o"]).first(), "h": r(P["h"]).max(), "l": r(P["l"]).min(), "c": r(P["c"]).last()}
            Q["cf"] = Q["c"].ffill()
        _bp[k] = Q
    return _bp[k]

def run(P, idx, d, B=1, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    """d: dict(Wh=24, depth, tp, hm (hours), cd (bars), guard=0.35). Returns hourly equity, log (hourly eb/xb), hourly inv."""
    Q = bar_panel(P, B)
    bidx = Q["o"].index
    O = Q["o"].values; n, K = O.shape
    Wb = max(int(round(d.get("Wh", 24) / B)), 1)
    R = np.ascontiguousarray(V.ref_panel({"c": Q["c"], "ref": Q["c"].rolling(24, min_periods=1).max().shift(1)}, Wb)[None]) \
        if Wb != 24 else np.ascontiguousarray(Q["c"].rolling(24, min_periods=1).max().shift(1).values[None])
    rid = np.zeros(1, np.int64)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars({"o": Q["o"]}) if liquidate else np.full(K, -1, np.int64)
    g = V.guard_mask({"o": Q["o"], "cf": Q["cf"]}, d.get("guard", 0.35), L=max(168 // B, 1))
    gmS = np.ascontiguousarray(g[None])
    tpS = np.full((1,) + O.shape, float(d["tp"]))
    cd = max(int(d["cd"]) + cd_add, 0)
    i0 = 0 if t0 is None else int(np.searchsorted(bidx.values, np.datetime64(t0)))
    a = lambda x, dt=np.float64: np.array([x], dt)
    eq, lg, inv = F.fk(O, Q["h"].values, Q["l"].values, Q["cf"].values, gmS, R, rid, lastb, a(d["depth"]), tpS,
                       a(int(round(d["hm"] / B)), np.int64), a(cd, np.int64), a(cd, np.int64), a(1.0), 99, 1, float(E.EPS),
                       float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0, 0.0, 0.0, 0.0, 720, 168, 0, 0)
    # to hourly
    endt = np.minimum(bidx.values + np.timedelta64(B - 1, "h"), idx.values[-1])
    eh = pd.Series(eq, pd.DatetimeIndex(endt)).reindex(idx, method="ffill").fillna(1.0)
    ih = pd.Series(inv, pd.DatetimeIndex(endt)).reindex(idx, method="ffill").fillna(0.0)
    lg = pd.DataFrame(lg, columns=V.LOGC)
    if len(lg):
        lg["eb"] = np.searchsorted(idx.values, bidx.values[lg.eb.astype(int).values])
        lg["xb"] = np.searchsorted(idx.values, bidx.values[lg.xb.astype(int).values])
    return eh, lg, ih

def evaluate(d, B=1, worse=False, stress=False, holdout=False, **extra):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress: kw.update(pen=F.PEN, pens=F.PEN)
    if stress: kw.update(mtp_now=True, cd_add=1)
    if dead: kw["liquidate"] = True
    kw.update(extra)
    e, lg, inv = run(P, idx, d, B, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg

def design_scores(d, B):
    """2021-23 only; same 6 versions as m11_fusion_lib.design_scores (cooldown +-1 BAR for B>1, +-1h for B=1)."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = run(P, idx, d, B, **kw)
        per = (("d", F.LATE, V.T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n": out["n_tpm"] = r["d_tpm"]
    ks = ("n", "cm", "cp", "bt", "ls")
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ks])); out["rmin"] = float(np.min([out[f"{t}_mo"] for t in ks]))
    return out

BASE = dict(Wh=24, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.35)
