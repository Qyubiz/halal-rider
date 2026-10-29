"""m12 "entry" (agent B): keep 90% mode's structure, change WHICH coin it bids on and AT WHAT PRICE.

ek = m10_velocity_lib.vk restricted to excl=1, one tier, one position, one new bid per hour (90% mode), with:
  LIM[t, k]  the coin's bid price at hour t, built in Python from closed candles only (NaN = no bid; the 35% crash
             guard is applied here as NaN). Anchor x (1 - depth), where
               anchor  'c' highest hourly close of the last 24 closed hours (90% mode / the live bot)
                       'h' highest hourly HIGH of the last 24 closed hours
                       'm' mean of the last 24 hourly closes (a VWAP-like mean)
               depth   scalar; or per group (dmaj for MAJORS, depth for the rest); or volatility-set:
                       clip(vk x median over the last 720 closed hours of the hourly range h/l-1, vlo, vhi)
  SC[t, k]   coin-choice score known at t's open (lower = preferred). Among coins whose open is at most `near` above
             their bid, the lowest score gets the bid; when no coin is that near, the closest coin (90% mode).
               'dist'   o/LIM-1 (90% mode: closest to its trigger)
               'liq'    minus trailing 30-day USDT volume (most liquid first)
               'size'   fixed coin order of engine.H19 (BTC, ETH, SOL, XRP, BNB, ...; a market-cap order)
               'bounce' minus the coin's own walk-forward bounce rate: single-coin replay of the same bid/tp/time stop
                        on its own history, counting only trades closed before t, prior 9 wins in 10
               'd7deep' / 'd7shal'  open vs the highest close of the last 168 closed hours, deepest / shallowest first
  t0         start hour ("start 2 weeks later" version)
Costs and timing exactly as vk (fee 0.1%/side, 0.02% slippage on market buys and time-stop sells, 0.1% for
LUNA/FTT/SRM when liquidate, limit buys need 0.1% trade-through, resting take-profit 0.1% through, pen/pens worse
fills, mtp_now bot timing, cooldown counted from the exit hour, cash from a sale usable next hour).
With anchor 'c', scalar depth, rank 'dist': ek == vk == 90% mode (m12_entry_check.py).
Not tested: bidding the top-2 coins with the whole account each and cancelling the other on a fill. Binance locks the
funds of every open order, so two 100% bids cannot rest at once (not fully funded); it is a sizing split (agent A).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m12/entry"))
OUT = os.environ["PB_OUT"]
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

LOGC = V.LOGC
T21, T24, T25 = V.T21, V.T24, V.T25
LATE = pd.Timestamp("2021-01-15")
PEN = 0.002
MAJORS = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT")
BASE = dict(depth=0.14, tp=0.025, hm=96, cd=3, anchor="c", rank="dist", near=0.0, dmaj=0.0, vk=0.0, vlo=0.10,
            vhi=0.25, guard=0.35)


@njit(cache=True)
def ek(o, h, l, c, LIM, SC, usesc, near, lastb, tpa, hm, cd, eps, fee, slip, pen, pens, mtp_now, t0):
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
                d = o[t, k] / LIM[t, k] - 1.0
                if usesc == 1:
                    if d <= near and not np.isnan(SC[t, k]):
                        key[m] = SC[t, k] - 1e6
                    else:
                        key[m] = d
                else:
                    key[m] = d
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
                a = min(per, cash)
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


@njit(cache=True)
def bounce_score(o, h, l, c, LIM, tpa, hm, cd, eps):
    """Per coin, walk-forward: replay the bid alone on the coin's own history (one trade at a time, cooldown cd);
    score[t] = -(wins + 9) / (trades + 10) over trades CLOSED before hour t (known at t's open)."""
    n, K = c.shape
    out = np.full((n, K), np.nan)
    for k in range(K):
        w = 0; nt = 0; held = False; ent = 0.0; eb = 0; ok = 0
        for t in range(n):
            out[t, k] = -(w + 9.0) / (nt + 10.0)
            if np.isnan(o[t, k]):
                continue
            if not held and t >= ok and not np.isnan(LIM[t, k]):
                lim = LIM[t, k]
                if o[t, k] <= lim:
                    ent = o[t, k]; held = True; eb = t
                elif l[t, k] <= lim * (1.0 - eps):
                    ent = lim; held = True; eb = t
            if held:
                tp = ent * (1.0 + tpa); px = -1.0
                if h[t, k] >= tp * (1.0 + eps):
                    px = tp
                if px < 0.0 and t - eb >= hm:
                    px = c[t, k]
                if px > 0.0:
                    held = False; nt += 1; ok = t + cd
                    if px * 0.998 > ent:
                        w += 1
    return out


# ------------------------------------------------------------------ inputs (closed candles only)
_cache = {}


def _k(P, *a):
    return (id(P["o"]),) + a


def anchor(P, a):
    k = _k(P, "anc", a)
    if k not in _cache:
        if a == "c":
            x = P["ref"].values
        elif a == "h":
            x = P["h"].rolling(24, min_periods=1).max().shift(1).values
        elif a == "m":
            x = P["c"].rolling(24, min_periods=1).mean().shift(1).values
        else:
            raise ValueError(a)
        _cache[k] = x
    return _cache[k]


def rng_med(P):
    k = _k(P, "rng")
    if k not in _cache:
        r = (P["h"] / P["l"] - 1.0)
        _cache[k] = r.rolling(720, min_periods=48).median().shift(1).values
    return _cache[k]


def guard(P, g):
    k = _k(P, "g", g)
    if k not in _cache:
        _cache[k] = V.guard_mask(P, g)
    return _cache[k]


def lim_table(P, d):
    A = anchor(P, d["anchor"])
    n, K = A.shape
    cols = list(P["o"].columns)
    if d["vk"] > 0:
        D = np.clip(d["vk"] * rng_med(P), d["vlo"], d["vhi"])
    else:
        D = np.full((n, K), d["depth"])
        if d["dmaj"] > 0:
            for j, s in enumerate(cols):
                if s in MAJORS:
                    D[:, j] = d["dmaj"]
    L = A * (1.0 - D)
    g = guard(P, d["guard"])
    return np.ascontiguousarray(np.where(g > 0, L, np.nan))


def score_table(P, d, L):
    r = d["rank"]
    if r == "dist":
        return np.zeros((1, 1)), 0
    if r == "liq":
        k = _k(P, "liq")
        if k not in _cache:
            _cache[k] = -(P["qv"].rolling(720, min_periods=24).sum().shift(1)).values
        S = _cache[k]
    elif r == "size":
        cols = list(P["o"].columns)
        S = np.tile(np.array([float(E.H19.index(s)) if s in E.H19 else 50.0 for s in cols]), (len(P["o"]), 1))
    elif r == "bounce":
        S = bounce_score(P["o"].values, P["h"].values, P["l"].values, P["cf"].values, L, float(d["tp"]), int(d["hm"]),
                         int(d["cd"]), float(E.EPS))
    elif r in ("d7deep", "d7shal"):
        k = _k(P, "d7")
        if k not in _cache:
            hi = P["c"].rolling(168, min_periods=1).max().shift(1)
            _cache[k] = (P["o"] / hi - 1.0).values
        S = _cache[k] if r == "d7deep" else -_cache[k]
    else:
        raise ValueError(r)
    return np.ascontiguousarray(S, dtype=np.float64), 1


def erun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    d = dict(BASE, **d)
    O = P["o"].values
    K = O.shape[1]
    L = lim_table(P, d)
    S, use = score_table(P, d, L)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = ek(O, P["h"].values, P["l"].values, P["cf"].values, L, S, use, float(d["near"]), lastb,
                     float(d["tp"]), int(d["hm"]), max(int(d["cd"]) + cd_add, 0), float(E.EPS), float(E.FEE), sl,
                     float(pen), float(pens), bool(mtp_now), i0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)


# ------------------------------------------------------------------ selection scores (2021-23 ONLY)
DPER = (("d", T21, T24),)
VERS = {"n": {}, "cm": dict(cd_add=-1), "cp": dict(cd_add=1), "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=LATE)}


def design_scores(d):
    """2021-23 only: 5 nearby versions (as designed, cooldown -1h, +1h, bot timing, start 2 weeks later), each with
    normal fills and with every buy and sell fill 0.2% worse. Never computes a metric after 2023."""
    syms, idx, P = V.market(False)
    out = {}
    for tag, kw in VERS.items():
        for fl, fk in (("", {}), ("w", dict(pen=PEN, pens=PEN))):
            e, lg, inv = erun(P, idx, d, **kw, **fk)
            per = (("d", LATE, T24),) if tag == "ls" else DPER
            r = V.summarize(e, lg, inv, idx, per)
            t = fl + tag
            out[f"{t}_wr"], out[f"{t}_mo"], out[f"{t}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
            if t == "n":
                out["n_tpm"], out["n_wt"] = r["d_tpm"], r["d_wt"]
    vs = list(VERS)
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in vs]))
    out["wrobust"] = float(np.mean([out[f"w{t}_mo"] for t in vs]))
    out["rwr"] = float(np.mean([out[f"{t}_wr"] for t in vs]))
    out["rdd_med"] = float(np.median([out[f"{t}_dd"] for t in vs]))
    out["w_wr"], out["w_mo"], out["w_dd"] = out["wn_wr"], out["wn_mo"], out["wn_dd"]
    return out


# ------------------------------------------------------------------ full evaluation (final look only)
def evaluate(d, worse=False, stress=False, holdout=False):
    syms, idx, P = V.market_holdout(stress) if holdout else V.market(stress)
    kw = {}
    if worse or stress:
        kw.update(pen=PEN, pens=PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1, liquidate=True)
    e, lg, inv = erun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg


def describe(d):
    d = dict(BASE, **d)
    a = {"c": "24h-high close", "h": "24h-high high", "m": "24h-mean close"}[d["anchor"]]
    if d["vk"] > 0:
        dep = f"{d['vk']:g}x rng30 [{d['vlo']*100:.0f}-{d['vhi']*100:.0f}%]"
    elif d["dmaj"] > 0:
        dep = f"maj {d['dmaj']*100:.1f}% / other {d['depth']*100:.1f}%"
    else:
        dep = f"{d['depth']*100:.1f}%"
    rk = d["rank"] + (f"(near {d['near']*100:.0f}%)" if d["rank"] != "dist" else "")
    return f"{a} -{dep} +{d['tp']*100:.1f}% {d['hm']}h cd{d['cd']} {rk}"


def key(d):
    return json.dumps(dict(BASE, **d), sort_keys=True)
