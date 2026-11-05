"""m14 "confirm": wait for a bounce instead of resting a limit at the dip level. Engine fc = single-account 90%-mode
kernel (m11_fusion_lib.fk with one tier, one position, no risk switches) plus a per-coin "dip armed" state machine.

cmode -1: baseline resting buy at ref24 x (1-dep) (must equal 90% mode exactly; m14_confirm_check.py).
cmode >= 0: the dip is "hit" when an hourly low trades 0.1% through ref24 x (1-dep) (crash guard mask applied to the
 hit hour). From the next hourly open on, the coin is armed, with dlow = lowest low since the hit bar. At each open t,
 using only bars up to t-1:
   cancel if t - hit_bar > N hours, or dlow <= dip level x (1 - y)   (y <= 0: no further-fall cancel)
   confirm (cmode 0) bar t-1 closed green; (1) bar t-1 close >= dlow x (1+x); (2) closes of t-3<t-2<t-1 rising twice
          in a row with t-2 >= hit bar
 then buy at the open of t at market: o x (1+slip) x (1+pen); several confirmed coins: the one whose open is the
 lowest vs its ref24. Exits as 90% mode: tp over the fill, hm hours, cooldown cdw/cdl after win/loss (counted from exit
 bar), mtp_now timing as fk. One position at a time, 100% of the money.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import engine as E  # noqa: E402

OUT = F.OUT
PEN = F.PEN


@njit(cache=True)
def fc(o, h, l, c, gm, R, dep, tp, hm, cd, eps, fee, slip, pen, pens, mtp_now, t0, cmode, cN, cy, cx, lastb):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    eq0 = np.zeros(K); ebar = np.full(K, -1, np.int64); mkt = np.zeros(K, np.int64)
    okc = np.zeros(K, np.int64)
    arm = np.full(K, -1, np.int64); dlow = np.zeros(K); lim0 = np.zeros(K)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0
    lg = np.zeros((400000, 9)); nl = 0
    held = -1
    for t in range(t0):
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
    for t in range(t0, n):
        # ---- entry
        bk = -1; blim = 0.0
        if held < 0 and cash > 1e-9:
            best = 1e18
            for k in range(K):
                if t < okc[k] or np.isnan(o[t, k]):
                    continue
                if cmode < 0:
                    if gm[t, k] <= 0.0:
                        continue
                    v = R[t, k]
                    if np.isnan(v):
                        continue
                    lim = v * (1.0 - dep)
                    d = o[t, k] / lim - 1.0
                    if d < best:
                        best = d; bk = k; blim = lim
                else:
                    if arm[k] < 0 or t - 1 < arm[k]:
                        continue
                    if t - arm[k] > cN or (cy > 0.0 and dlow[k] <= lim0[k] * (1.0 - cy)):
                        arm[k] = -1
                        continue
                    if gm[t, k] <= 0.0:
                        continue
                    ok = False
                    if cmode == 0:
                        ok = c[t - 1, k] > o[t - 1, k]
                    elif cmode == 1:
                        ok = c[t - 1, k] >= dlow[k] * (1.0 + cx)
                    else:
                        ok = (t >= 3) and (t - 2 >= arm[k]) and c[t - 1, k] > c[t - 2, k] and c[t - 2, k] > c[t - 3, k]
                    if ok:
                        r = R[t, k]
                        if np.isnan(r):
                            continue
                        d = o[t, k] / r
                        if d < best:
                            best = d; bk = k
        if bk >= 0:
            k = bk
            fill = -1.0; ismkt = 0
            if cmode >= 0:
                fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
            elif o[t, k] <= blim:
                fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
            elif l[t, k] <= blim * (1.0 - eps):
                fill = blim * (1.0 + pen)
            if fill > 0.0:
                mkt[k] = ismkt if mtp_now else 0
                a = min(eq_prev, cash)
                cash -= a; units[k] = a * (1.0 - fee) / fill
                entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev
                tpp[k] = fill * (1.0 + tp); held = k; arm[k] = -1
        # ---- exits
        for k in range(K):
            if units[k] > 0.0 and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if ebar[k] < t or mkt[k] == 1:
                    if ebar[k] == t:
                        if h[t, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    elif o[t, k] >= tpp[k]:
                        px = o[t, k]
                    elif h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                    age = t - ebar[k] + mkt[k]
                    if px < 0.0 and age >= hm:
                        px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; nl += 1
                    units[k] = 0.0; held = -1
                    okc[k] = t + cd
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; nl += 1
                units[k] = 0.0; held = -1; okc[k] = t + cd
        # ---- dip bookkeeping with bar t (used from t+1)
        if cmode >= 0:
            for k in range(K):
                if np.isnan(o[t, k]):
                    continue
                if arm[k] >= 0:
                    if l[t, k] < dlow[k]:
                        dlow[k] = l[t, k]
                elif units[k] <= 0.0 and gm[t, k] > 0.0:
                    r = R[t, k]
                    if not np.isnan(r):
                        lim = r * (1.0 - dep)
                        if l[t, k] <= lim * (1.0 - eps):
                            arm[k] = t; dlow[k] = l[t, k]; lim0[k] = lim
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for k in range(K):
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
    for k in range(K):
        if units[k] > 0.0 and nl < 400000:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; nl += 1
    return eq, lg[:nl].copy(), inv


def crun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    """d: dict(cmode, dep, tp, hm, cd, N, y, x, guard)."""
    O = P["o"].values
    n, K = O.shape
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    g = np.ascontiguousarray(F.guard(P, d.get("guard", 0.35)))
    R = np.ascontiguousarray(F.ref(P, 24))
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = fc(O, P["h"].values, P["l"].values, P["cf"].values, g, R, float(d["dep"]), float(d["tp"]),
                     int(d["hm"]), max(int(d.get("cd", 3)) + cd_add, 0), float(E.EPS), float(E.FEE), sl, float(pen),
                     float(pens), bool(mtp_now), i0, int(d["cmode"]), int(d.get("N", 24)), float(d.get("y", 0.0)),
                     float(d.get("x", 0.0)), lastb)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


BASE = dict(cmode=-1, dep=0.14, tp=0.025, hm=96, cd=3, guard=0.35)
LATE = F.LATE


def design_scores(d):
    """2021-23 only: n, w, cm, cp, bt, ls versions as m11_fusion_lib.design_scores."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=PEN, pens=PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=LATE)}
    for tag, kw in runs.items():
        e, lg, inv = crun(P, idx, d, **kw)
        per = (("d", LATE, V.T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"] = r["d_tpm"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


def evaluate(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress:
        kw.update(pen=PEN, pens=PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1)
    if dead:
        kw["liquidate"] = True
    e, lg, inv = crun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
