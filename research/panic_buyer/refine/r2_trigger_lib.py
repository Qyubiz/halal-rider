"""Round 2, lens "trigger": engine wrapper for trigger-mechanics variants.

gkernel_t is engine.gkernel copied unchanged except for ONE switch: the cooldown after a sale depends on how the
trade ended. After a time-stop exit whose return is <= tsloss the coin is blocked for `cdts` hours instead of
`cd`. With cdts == cd (the default) the kernel is the live engine (r2_trigger_check.py proves it reproduces
engine.run exactly).

trun(P, idx, ref=None, cdts=None, tsloss=..., **engine.run kwargs) is engine.run with two extras:
  ref     (n,K) array used as the trigger reference instead of P["ref"] (24h highest close, shifted 1)
  cdts    cooldown after a time stop (hours); None = same as cd
Install it with use_trun() so study.study() (which calls engine.run) runs the protocol unchanged.
"""
import numpy as np
import pandas as pd
from numba import njit

import engine as E

_ORIG_RUN = E.run


@njit(cache=True)
def gkernel_t(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
              tpwait, tprest, cdts, tsloss):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 7)); nl = 0
    for t in range(n):
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or smult[t, k] <= 0.0:
                continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0
        order = np.argsort(dist)
        nsel = 0
        for q in range(K):
            if dist[order[q]] < 1e8 and nsel < nord[t]:
                nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax[t] * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth[t, k])
                fill = -1.0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per * smult[t, k], cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if t - ebar[k] < tpwait:
                        pass
                    elif o[t, k] >= tpp[k]:
                        px = o[t, k]
                    elif tprest and h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                    if px > 0.0 and part < 1.0:
                        q = units[k] * part
                        p1 = q * px * (1.0 - fee)
                        cash += p1; got[k] += p1; units[k] -= q
                        stage[k] = 1; sbar[k] = t; peak[k] = max(px, h[t, k])
                        px = -1.0
                elif sbar[k] < t:
                    if tp2 > 0.0:
                        p2 = entry[k] * (1.0 + tp2)
                        if o[t, k] >= p2:
                            px = o[t, k]; why = 3.0
                        elif h[t, k] >= p2 * (1.0 + eps):
                            px = p2; why = 3.0
                    if px < 0.0 and trail > 0.0:
                        if h[t, k] > peak[k]:
                            peak[k] = h[t, k]
                        if c[t, k] <= peak[k] * (1.0 - trail):
                            px = c[t, k] * (1.0 - slip[k]); why = 2.0
                if px < 0.0 and t - ebar[k] >= hm[t]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    r = got[k] / amt0[k] - 1.0
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = r; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; nl += 1
                    units[k] = 0.0
                    # the only change vs engine.gkernel: a longer block after a (losing) time stop
                    if why == 1.0 and r <= tsloss:
                        okbar[k] = t + cdts
                    else:
                        okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def trun(P, idx, depth=E.LIVE["depth"], N=E.LIVE["N"], fmax=E.LIVE["fmax"], smult=1.0, tp=E.LIVE["tp"], part=1.0,
         tp2=0.0, trail=0.0, hm=E.LIVE["hm"], cd=E.LIVE["cd"], eps=E.EPS, fee=E.FEE, slip=E.SLIP_MAJOR, pen=0.0,
         tpwait=1, tprest=True, ref=None, cdts=None, tsloss=1e9):
    full = E.full
    O = P["o"].values; n, K = O.shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    R = P["ref"].values if ref is None else np.ascontiguousarray(np.asarray(ref, np.float64))
    assert R.shape == (n, K)
    eq, lg = gkernel_t(O, P["h"].values, P["l"].values, P["cf"].values, R,
                       full(n, K, depth), full(n, 0, N, np.int64), full(n, 0, fmax), full(n, K, smult), full(n, K, tp),
                       float(part), float(tp2), float(trail), full(n, 0, hm, np.int64), int(cd), eps, fee, sl, pen,
                       int(tpwait), bool(tprest), int(cd if cdts is None else cdts), float(tsloss))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=E.LOG_COLS)


def use_trun():
    """Make engine.run (and so study.evaluate / study.study) use trun."""
    E.run = trun


# ------------------------------------------------------------------ trigger references (known at the hour's start)
def ref_close(P, w):
    """Highest close of the last w closed hourly candles (live: w=24)."""
    return P["c"].rolling(w, min_periods=1).max().shift(1)


def ref_high(P, w):
    """Highest hourly high of the last w closed hourly candles."""
    return P["h"].rolling(w, min_periods=1).max().shift(1)
