"""m16 "riskmodel": 90% mode with the BET SIZE (never the decision to buy) set by a walk-forward model of the trade's
downside.

Trading rules = 90% mode exactly (1 coin, resting bid at the 24h highest hourly close x 0.86, +2.5% take-profit, 96h
time stop, 3h cooldown, 35% crash guard, closest coin to its bid). The only change: when a bid fills on coin k in hour
t, the bot spends SZ[t, k] x equity (SZ in [floor, 1]); the rest stays in USDT until the trade closes. Because the
coin, the hours and the exits are untouched, the set of trades is IDENTICAL to 90% mode (same win rate).

sk = m12_entry_lib.ek (== 90% mode) with one switch: the size table SZ (n, K). With SZ = 1 it is ek (proved in
m16_riskmodel_check.py).

Model (one panel Dec 2017 - Oct 2026 from m10_model_lib.load_all(dead=True): 16 coins before 2021, 19 + LUNA/FTT/SRM
after; LUNA/FTT/SRM are in so the stressed runs get scores too):
  rows      every coin-hour whose 14% bid would fill and that passes the crash guard (dd7 >= -35%), m10 definition.
  targets   'mae'  worst hourly close in hours t..t+96 vs the fill price (maximum adverse excursion; regression,
                   risk score = minus the predicted value)
            'big'  1 if the trade's realized return (take-profit or 96h close, after fees) < -10% (classifier,
                   risk score = predicted probability)
  features  'base' the 39 m10 features (m10_model_lib.FEATS)
            'fut'  + 10 Binance USD-M perpetual features (FUTS), NaN where the coin had no futures data yet
                   (funding / premium from 2020, open interest from Dec 2021)
  training  LightGBM, retrained at each month start from 2018-06 on rows bought >= 97 hours before the month starts
            (outcome known), expanding window (m10_model_lib.walk_forward2).
Sizing: thr = q-quantile of the out-of-sample risk scores of all guard-passing candidate hours in the 365 days
before the month (known at the month start); SZ = floor where score > thr, else 1. No score yet / no threshold yet
(before ~mid-2019) -> SZ = 1.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m16/riskmodel"))
OUT = os.environ["PB_OUT"]
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m12_entry_lib as L  # noqa: E402
import m10_model_lib as M  # noqa: E402
import m13_futures_lib as FU  # noqa: E402

PEN = L.PEN
T21, T24, T25 = V.T21, V.T24, V.T25
VERS = L.VERS
FUTS = ["fund_last", "fund_mean72", "fund_chg72", "prem1", "prem24", "prem_chg", "oi_r1", "oi_r12", "oi_r72",
        "oi_rel7d"]


@njit(cache=True)
def sk(o, h, l, c, LIM, SZ, lastb, tpa, hm, cd, eps, fee, slip, pen, pens, mtp_now, t0):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); mkt = np.zeros(K, np.int64); okc = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.ones(n); inv = np.zeros(n); eq_prev = 1.0
    maxlog = 200000; lg = np.zeros((maxlog, 9)); nl = 0
    key = np.empty(K); ck = np.empty(K, np.int64); clim = np.empty(K)
    for t in range(t0):
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
    for t in range(t0, n):
        nheld = 0
        for k in range(K):
            if units[k] > 0.0:
                nheld += 1
        m = 0
        if nheld == 0:
            for k in range(K):
                if t < okc[k] or np.isnan(o[t, k]) or np.isnan(LIM[t, k]):
                    continue
                key[m] = o[t, k] / LIM[t, k] - 1.0
                ck[m] = k; clim[m] = LIM[t, k]; m += 1
        if m > 0 and cash > 1e-9:
            j = np.argsort(key[:m])[0]
            k = ck[j]; lim = clim[j]
            per = min(eq_prev, cash)
            fill = -1.0; ismkt = 0
            if o[t, k] <= lim:
                fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
            elif l[t, k] <= lim * (1.0 - eps):
                fill = lim * (1.0 + pen)
            if fill > 0.0:
                mkt[k] = ismkt if mtp_now else 0
                a = min(per * SZ[t, k], cash)
                cash -= a; units[k] = a * (1.0 - fee) / fill
                entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev
                tpp[k] = fill * (1.0 + tpa)
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
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
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; nl += 1
                    units[k] = 0.0
                    okc[k] = t + cd
        for k in range(K):
            if units[k] > 0.0 and lastb[k] == t:
                proceeds = units[k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 5.0
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; nl += 1
                units[k] = 0.0
                okc[k] = t + cd
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
            lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; lg[nl, 7] = eq0[k]; nl += 1
    return eq, lg[:nl].copy(), inv


def srun(P, idx, SZ=None, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    O = P["o"].values
    n, K = O.shape
    Lm = L.lim_table(P, L.BASE)
    SZ = np.ones((n, K)) if SZ is None else np.ascontiguousarray(SZ, dtype=np.float64)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = sk(O, P["h"].values, P["l"].values, P["cf"].values, Lm, SZ, lastb, 0.025, 96, max(3 + cd_add, 0),
                     float(E.EPS), float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


# ------------------------------------------------------------------ model inputs
def fut_features(idx, syms):
    """(n, K, 10) futures features known at the open of each hour (NaN = no data)."""
    out = np.full((len(idx), len(syms), len(FUTS)), np.nan, np.float32)
    for j, s in enumerate(syms):
        f = FU.known_series("funding", s, idx)
        p = FU.known_series("prem", s, idx)
        oi = FU.known_series("oi", s, idx)
        cols = [f, f.rolling(72, min_periods=8).mean(), f - f.rolling(72, min_periods=8).max(),
                p, p.rolling(24, min_periods=12).mean(), p - p.rolling(24, min_periods=12).mean(),
                oi / oi.shift(1) - 1, oi / oi.shift(12) - 1, oi / oi.shift(72) - 1,
                oi / oi.rolling(168, min_periods=48).mean() - 1]
        for i, c in enumerate(cols):
            out[:, j, i] = c.values
    out[~np.isfinite(out)] = np.nan
    return out


def build():
    """Big panel (dead coins in), X (n, K, 49), candidates, fill, realized return, exit hour, mae96, guard."""
    p = os.path.join(OUT, "build_m16.npz")
    D = M.load_all(dead=True)
    syms, idx, P = M.big_panel(D)
    if os.path.exists(p):
        z = np.load(p)
        return syms, idx, P, {k: z[k] for k in z.files}
    F = M.features(P, M.DEPTH)
    X = np.stack([F[f] for f in M.FEATS], axis=2).astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    del F
    XF = fut_features(idx, syms)
    X = np.concatenate([X, XF], axis=2)
    cand, fill, y, ret, xh = M.candidates(P, M.DEPTH, M.TP, M.HM)
    cf = P["cf"].values
    n, K = cf.shape
    mae = np.full((n, K), np.nan)
    from numpy.lib.stride_tricks import sliding_window_view
    for k in range(K):
        x = np.where(np.isnan(cf[:, k]), np.inf, cf[:, k])
        w = sliding_window_view(x, 97).min(axis=1)           # w[t] = min cf[t..t+96]
        ts = np.flatnonzero(cand[:, k])
        ts = ts[ts < len(w)]
        mae[ts, k] = np.minimum(w[ts] / fill[ts, k] - 1.0, 0.0)
    guard = X[:, :, M.FEATS.index("dd7")] >= -0.35
    z = dict(X=X, cand=cand, fill=fill, y=y, ret=ret, xh=xh, mae=mae, guard=guard)
    np.savez(p, **z)
    return syms, idx, P, z


def pred_path(target, fs):
    return os.path.join(OUT, f"risk_{target}_{fs}.npy")


def train(target, fs, Z, idx, log=print):
    """Out-of-sample RISK score (higher = riskier) for every (hour, coin) from 2018-06."""
    p = pred_path(target, fs)
    if os.path.exists(p):
        return np.load(p)
    X = Z["X"]
    nf = len(M.FEATS) + (len(FUTS) if fs == "fut" else 0)
    ok = Z["cand"] & Z["guard"] & ~np.isnan(Z["mae"]) & ~np.isnan(Z["ret"])
    rt, rk = np.nonzero(ok)
    if target == "mae":
        tgt = Z["mae"][rt, rk]
    else:
        tgt = (Z["ret"][rt, rk] < -0.10).astype(float)
    pred = M.walk_forward2(X, rt, rk, tgt, idx, feat=np.arange(nf), log=log)
    risk = -pred if target == "mae" else pred
    np.save(p, risk.astype(np.float32))
    return risk


_thr = {}


def thresholds(risk, Z, idx, q, key):
    k = (key, q)
    if k not in _thr:
        t = M.trailing_threshold(risk, Z["cand"] & Z["guard"], idx, q)
        t[~np.isfinite(t)] = np.inf          # no threshold yet -> full size
        _thr[k] = t
    return _thr[k]


def size_big(risk, Z, idx, q, floor, key):
    thr = thresholds(risk, Z, idx, q, key)
    sz = np.where(np.isnan(risk), 1.0, np.where(risk > thr[:, None], floor, 1.0))
    return sz


def to_panel(sz, idx_b, syms_b, idx, syms):
    return np.ascontiguousarray(M.to_panel(sz, idx_b, syms_b, idx, syms, 1.0), dtype=np.float64)


def bot_buys(lg, idx, syms):
    """List of (timestamp, sym, ret, acct_ret) for each trade of a run log."""
    return [(idx[int(r.eb)], syms[int(r.k)], r.ret, r.ret * r.amt / r.eq0) for r in lg.itertuples()]
