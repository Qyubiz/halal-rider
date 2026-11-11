"""m15 "earlycut": early warning exits inside a 90% mode trade, using only what happens after the buy.

ek = m11_fusion_lib.fk with one tier (24h high x 0.86, +2.5%, 96h, cooldown 3h, guard 35%, 1 coin x 100%), the
guardrails switches removed (all off in 90% mode), plus three conditional exits, each evaluated at an hourly close
(known at that close, sold at that close - slippage, or a resting limit from the next hour on):
  A "continuation" (fam=1): age <= aN hours and coin close <= entry x (1 - ax) AND market close <= market close of
     the entry hour x (1 - ay); market = BTC close (mi=0) or the equal-weight 19-coin index (mi=1: cumulated
     cross-coin mean of hourly log returns, live coins only). Market sell at the close.  why = 11
  B "failed bounce" (fam=2): age <= bN, the highest close since the buy reached entry x (1 + bb), and the coin now
     closes under the fill hour's low x (1 - bz) (a new lower low after a failed bounce). Market sell. why = 12
  C "limit at a small loss" (fam=3): first close with age >= cN that is under entry arms a resting limit sell at
     entry x cl (cl < 1); from the next hour it fills at the open if the open is at/above it, else when the high
     trades 0.1% through it; the +2.5% take-profit and the 96h stop stay. why = 13
fam=0: 90% mode exactly (m15_earlycut_check.py). Costs/timing as fk (fee 0.1%/side, 0.02% slippage on market
orders, 0.1% for LUNA/FTT/SRM when liquidate, limit fills need 0.1% trade-through, pen/pens, mtp_now, t0).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m15/earlycut"))
OUT = os.environ["PB_OUT"]
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m11_fusion_lib as F  # noqa: E402

LOGC = V.LOGC
T21, T24, T25 = V.T21, V.T24, V.T25
OFF = dict(fam=0, mi=0, aN=0, ax=0.0, ay=0.0, bN=0, bb=0.0, bz=0.0, cN=0, cl=1.0)
PEN = 0.002


@njit(cache=True)
def ek(o, h, l, c, gm, R, lastb, dep, tp, hm, cdw, cdl, eps, fee, slip, pen, pens, mtp_now, t0,
       mk, fam, aN, ax, ay, bN, bb, bz, cN, cl):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    eq0 = np.zeros(K); ebar = np.full(K, -1, np.int64); mkt = np.zeros(K, np.int64)
    maxc = np.zeros(K); armed = np.full(K, -1, np.int64); lowf = np.zeros(K)
    okc = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 9)); nl = 0
    cdist = np.empty(K); ck = np.empty(K, np.int64); clim = np.empty(K)
    for t in range(t0):
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
    for t in range(t0, n):
        m = 0
        for k in range(K):
            if units[k] > 0.0 or t < okc[k] or np.isnan(o[t, k]):
                continue
            if gm[t, k] <= 0.0:
                continue
            v = R[t, k]
            if np.isnan(v):
                continue
            lim = v * (1.0 - dep)
            cdist[m] = o[t, k] / lim - 1.0; ck[m] = k; clim[m] = lim; m += 1
        if m > 0 and cash > 1e-9:
            j = np.argsort(cdist[:m])[0]
            k = ck[j]; lim = clim[j]
            per = min(eq_prev, cash)
            fill = -1.0; ismkt = 0
            if o[t, k] <= lim:
                fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
            elif l[t, k] <= lim * (1.0 - eps):
                fill = lim * (1.0 + pen)
            if fill > 0.0:
                mkt[k] = ismkt if mtp_now else 0
                a = min(per, cash)
                cash -= a; units[k] = a * (1.0 - fee) / fill
                entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev
                tpp[k] = fill * (1.0 + tp); maxc[k] = 0.0; armed[k] = -1; lowf[k] = l[t, k]
        # ---------------- exits
        for k in range(K):
            if units[k] > 0.0 and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if ebar[k] < t or mkt[k] == 1:
                    tgt = tpp[k]
                    if armed[k] >= 0 and armed[k] < t:
                        tgt = min(tgt, entry[k] * cl)
                    if ebar[k] == t:
                        if h[t, k] >= tgt * (1.0 + eps):
                            px = tgt
                    elif o[t, k] >= tgt:
                        px = o[t, k]
                    elif h[t, k] >= tgt * (1.0 + eps):
                        px = tgt
                    if px > 0.0 and tgt < tpp[k]:
                        why = 13.0
                    age = t - ebar[k] + mkt[k]
                    if px < 0.0 and age >= hm:
                        px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px < 0.0 and fam > 0:
                    age = t - ebar[k]
                    if c[t, k] > maxc[k]:
                        maxc[k] = c[t, k]
                    if fam == 1:
                        if age <= aN and c[t, k] <= entry[k] * (1.0 - ax) and \
                                np.exp(mk[t] - mk[ebar[k]]) - 1.0 <= -ay:
                            px = c[t, k] * (1.0 - slip[k]); why = 11.0
                    elif fam == 2:
                        if age <= bN and maxc[k] >= entry[k] * (1.0 + bb) and c[t, k] < lowf[k] * (1.0 - bz):
                            px = c[t, k] * (1.0 - slip[k]); why = 12.0
                    elif fam == 3:
                        if armed[k] < 0 and age >= cN and c[t, k] < entry[k]:
                            armed[k] = t
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; nl += 1
                    units[k] = 0.0
                    win = proceeds > amt0[k]
                    okc[k] = t + (cdw if win else cdl)
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; nl += 1
                units[k] = 0.0
                okc[k] = t + cdl
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for k in range(K):
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
    for k in range(K):
        if units[k] > 0.0 and nl < maxlog:
            proceeds = units[k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            lg[nl, 0] = ebar[k]; lg[nl, 1] = n - 1; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; lg[nl, 8] = 0; nl += 1
    return eq, lg[:nl].copy(), inv


_mkc = {}


def mkt_index(P, mi):
    """Log market level per hour (known at that hour's close): mi=0 BTC log close (ffilled); mi=1 cumulated
    cross-coin mean of hourly log returns of the live 19 (or holdout 16) coins (dead coins excluded)."""
    key = (id(P["o"]), mi)
    if key not in _mkc:
        if mi == 0:
            x = np.log(P["cf"]["BTCUSDT"].values)
        else:
            cols = [s for s in P["c"].columns if s not in E.DEAD]
            lr = np.log(P["c"][cols]).diff()
            with np.errstate(all="ignore"):
                x = np.nancumsum(np.nanmean(lr.values, axis=1))
        _mkc[key] = np.ascontiguousarray(np.nan_to_num(x), dtype=np.float64)
    return _mkc[key]


def erun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    d = dict(OFF, **d)
    O = P["o"].values
    K = O.shape[1]
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    g = np.ascontiguousarray(F.guard(P, 0.35))
    R = np.ascontiguousarray(F.ref(P, 24))
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = ek(O, P["h"].values, P["l"].values, P["cf"].values, g, R, lastb, 0.14, 0.025, 96,
                     max(3 + cd_add, 0), max(3 + cd_add, 0), float(E.EPS), float(E.FEE), sl, float(pen), float(pens),
                     bool(mtp_now), i0, mkt_index(P, d["mi"]), int(d["fam"]), int(d["aN"]), float(d["ax"]),
                     float(d["ay"]), int(d["bN"]), float(d["bb"]), float(d["bz"]), int(d["cN"]), float(d["cl"]))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)


def design_scores(d):
    """2021-23 ONLY: normal, worse fills, and the 5 nearby versions (as m11_fusion_lib.design_scores)."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=PEN, pens=PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = erun(P, idx, d, **kw)
        per = (("d", F.LATE, T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"], out["n_wt"], out["lg"] = r["d_tpm"], r["d_wt"], lg
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
    e, lg, inv = erun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg


def loss_stats(lg, idx, a, b):
    """Average losing trade and worst trade, both as % of the account the trade was sized on."""
    tb = idx[lg.eb.astype(int).values]
    sel = np.asarray((tb >= a) & ((tb < b) if b is not None else True), bool)
    x = lg[sel]
    acct = (x.ret * x.amt / x.eq0).values
    los = acct[acct <= 0]
    return (float(los.mean()) if len(los) else 0.0), (float(acct.min()) if len(acct) else 0.0), int(len(los))
