"""m11 "cashuse": 90% mode as the core of ONE account, plus a second engine (S2) that works with the core's idle cash
and gives it back when the core needs it.

ck = m10_velocity_lib.vk with one core tier (90% mode: highest hourly close of the last 24 closed hours x 0.86, +2.5%
resting take-profit from the next hourly check, 96h time stop, 3h cooldown, one position, 100% of the account, 35%
crash guard under the 168h high close) plus a dip-buying second engine with its own positions (sleeve 1). With s2 off
(s2_on=0) ck gives exactly vk / 90% mode (m11_cashuse_check.py).

Second engine (sleeve 1): fully funded resting bids at R2[t,k] x (1 - d2) on the n2 coins closest to their bid
(only coins with m2[t,k] > 0: coin set x S2 crash guard x regime, all from closed candles), each order
min(fs2 x account, free cash / bids); take-profit tp2 (resting from the next hourly check), time stop hm2 hours,
cooldown cd2 hours per coin after an S2 exit. A coin may be held by the core and by S2 at once.

Cash priority and pre-emption (the point of this angle). "Armed" = the core is flat and the coin nearest its core
trigger opens at most armx above it (dist = open / trigger - 1 <= armx).
  armed hour:   the core bids first with ALL free cash (as 90% mode). pm decides what S2 does:
                pm 0 keep    S2 may still bid with what the core left (normally nothing) and keeps its positions;
                pm 1 cancel  S2 places no bid this hour, keeps its positions (core sized from free cash only);
                pm 2 sell    S2 positions are sold at the open (market: -slippage, -fee, logged as trades, why 6),
                             S2 places no bid; the core bids with all the cash.
  unarmed hour: S2 bids first; the core keeps a resting bid on its nearest coin with the cash S2 did not reserve.
                watch 0 (hourly bot only): that is all.
                watch 1 (a 1-minute watcher, new code): when the core coin trades within armx of its trigger during
                the hour (its low <= trigger x (1+armx)), the watcher cancels S2's unfilled bids and (pm 2) sells S2's
                positions; the core bid then uses that cash too. Modelled conservatively: S2 bids that fill in that
                hour are assumed to fill first; S2 positions are sold at their coin's LOW of the hour (- slippage).
  pause_cl      after a losing core trade, S2 places no bid for that many hours (0 = off).
  cut_s2        after a losing S2 trade, the NEXT core order is sized x cut_s2 (1 = off).
Costs as the m10 engines: 0.1% fee per side, market buys/sells 0.02% slippage (0.1% LUNA/FTT/SRM with liquidate),
limit buys fill only when the low trades 0.1% through, take-profits only when the high trades 0.1% through; pen/pens
make every buy/sell fill worse (stress 0.002); mtp_now = real bot timing after a market buy; cd_add adds hours to
every cooldown. Cash from a sale is usable at the next hourly check (pre-emption sales: in the same hour).
Log: eb, xb, k, why (0 tp, 1 time stop, 5 collapse cut, 6 pre-empted sale, 9 open at end), ret, amt, entry, eq0, sl.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m11/cashuse"))
os.makedirs(OUT, exist_ok=True)
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0", "sl"]
STRESS_PEN = V.STRESS_PEN
T21, T24, T25, H18, H21 = V.T21, V.T24, V.T25, V.H18, V.H21


@njit(cache=True)
def _log(lg, nl, eb, t, k, why, proceeds, amt, entry, eq0, s):
    if nl < lg.shape[0]:
        lg[nl, 0] = eb; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why; lg[nl, 4] = proceeds / amt - 1.0
        lg[nl, 5] = amt; lg[nl, 6] = entry; lg[nl, 7] = eq0; lg[nl, 8] = s
        return nl + 1
    return nl


@njit(cache=True)
def ck(o, h, l, c, gm, Rc, R2, m2, lastb, dc, tpc, hmc, cdc, fsc, core_on,
       s2_on, d2, tp2, hm2, cd2, fs2, n2, near2, armx, pm, watch, pause_cl, cut_s2,
       eps, fee, slip, pen, pens, mtp_now, cd_add, t0):
    n, K = c.shape
    # sleeve 0 = core, 1 = S2
    units = np.zeros((2, K)); entry = np.zeros((2, K)); amt0 = np.zeros((2, K)); tpp = np.zeros((2, K))
    eq0 = np.zeros((2, K)); ebar = np.full((2, K), -1, np.int64); mkt = np.zeros((2, K), np.int64)
    okc = np.zeros(K, np.int64); ok2 = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cash = 1.0; eq = np.empty(n); inv = np.empty(n); eq_prev = 1.0
    maxlog = 800000; lg = np.zeros((maxlog, 9)); nl = 0
    d2a = np.empty(K); bk = np.empty(K, np.int64); bamt = np.empty(K)
    pause2 = 0; cmult = 1.0
    hmv = np.array([hmc, hm2]); tpv = np.array([tpc, tp2])
    for t in range(n):
        if t < t0:                               # warm-up hours (start-date shift): flat, nothing traded
            eq[t] = 1.0; inv[t] = 0.0
            continue
        # ---------------- core candidate (one position, nearest coin)
        held_c = 0
        for k in range(K):
            if units[0, k] > 0.0:
                held_c = 1
        kc = -1; best = 1e18; limc = 0.0
        if core_on == 1 and held_c == 0:
            for k in range(K):
                if t < okc[k] or np.isnan(o[t, k]) or gm[t, k] <= 0.0 or np.isnan(Rc[t, k]):
                    continue
                lim = Rc[t, k] * (1.0 - dc)
                d = o[t, k] / lim - 1.0
                if d < best:
                    best = d; kc = k; limc = lim
        armed = kc >= 0 and best <= armx
        # ---------------- pre-emption at the open
        if armed and pm == 2:
            for k in range(K):
                if units[1, k] > 0.0 and not np.isnan(o[t, k]):
                    proceeds = units[1, k] * o[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    nl = _log(lg, nl, ebar[1, k], t, k, 6.0, proceeds, amt0[1, k], entry[1, k], eq0[1, k], 1)
                    if proceeds <= amt0[1, k]:
                        cmult = cut_s2
                    units[1, k] = 0.0; ok2[k] = t + cd2 + cd_add
        # ---------------- S2 bids (first in unarmed hours)
        nb2 = 0; res2 = 0.0; fill2 = 0.0
        s2_bids = s2_on == 1 and t >= pause2 and not (armed and pm >= 1)
        if s2_bids and not armed:
            held2 = 0
            for k in range(K):
                if units[1, k] > 0.0:
                    held2 += 1
            m = 0
            for k in range(K):
                d2a[k] = 1e18
                if units[1, k] > 0.0 or t < ok2[k] or np.isnan(o[t, k]) or m2[t, k] <= 0.0 or np.isnan(R2[t, k]):
                    continue
                dd = o[t, k] / (R2[t, k] * (1.0 - d2)) - 1.0
                if dd > near2:
                    continue
                d2a[k] = dd; m += 1
            free = min(n2 - held2, m)
            if free > 0 and cash > 1e-9:
                order = np.argsort(d2a)
                per = min(fs2 * eq_prev, cash / free)
                for q in range(free):
                    k = order[q]
                    bk[nb2] = k; bamt[nb2] = per; nb2 += 1; res2 += per
                cash_open = cash
                for q in range(nb2):
                    k = bk[q]
                    lim = R2[t, k] * (1.0 - d2)
                    fill = -1.0; ismkt = 0
                    if o[t, k] <= lim:
                        fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                    elif l[t, k] <= lim * (1.0 - eps):
                        fill = lim * (1.0 + pen)
                    if fill > 0.0 and cash > 1e-9:
                        a = min(bamt[q], cash)
                        cash -= a; fill2 += a; units[1, k] = a * (1.0 - fee) / fill
                        entry[1, k] = fill; amt0[1, k] = a; ebar[1, k] = t; eq0[1, k] = eq_prev
                        tpp[1, k] = fill * (1.0 + tp2); mkt[1, k] = ismkt if mtp_now else 0
                cash = cash_open - fill2
        # ---------------- core bid
        if kc >= 0 and cash > 1e-9:
            avail = cash
            if not armed and nb2 > 0:
                avail = cash - (res2 - fill2)             # S2 still reserves its unfilled bids
                if watch == 1 and l[t, kc] <= limc * (1.0 + armx):
                    avail = cash                         # watcher cancelled S2's unfilled bids
                    if pm == 2:
                        for k in range(K):
                            if units[1, k] > 0.0 and ebar[1, k] < t and not np.isnan(l[t, k]):
                                proceeds = units[1, k] * l[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                                cash += proceeds; avail += proceeds
                                nl = _log(lg, nl, ebar[1, k], t, k, 6.0, proceeds, amt0[1, k], entry[1, k], eq0[1, k], 1)
                                units[1, k] = 0.0; ok2[k] = t + cd2 + cd_add
            elif not armed and watch == 1 and pm == 2 and l[t, kc] <= limc * (1.0 + armx):
                for k in range(K):
                    if units[1, k] > 0.0 and ebar[1, k] < t and not np.isnan(l[t, k]):
                        proceeds = units[1, k] * l[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                        cash += proceeds; avail += proceeds
                        nl = _log(lg, nl, ebar[1, k], t, k, 6.0, proceeds, amt0[1, k], entry[1, k], eq0[1, k], 1)
                        units[1, k] = 0.0; ok2[k] = t + cd2 + cd_add
            per = min(fsc * eq_prev * cmult, cash)
            k = kc
            fill = -1.0; ismkt = 0
            if o[t, k] <= limc:
                fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
            elif l[t, k] <= limc * (1.0 - eps):
                fill = limc * (1.0 + pen)
            if fill > 0.0 and avail > 1e-9:
                mkt[0, k] = ismkt if mtp_now else 0
                a = min(per * gm[t, k], avail)
                cash -= a; units[0, k] = a * (1.0 - fee) / fill
                entry[0, k] = fill; amt0[0, k] = a; ebar[0, k] = t; eq0[0, k] = eq_prev
                tpp[0, k] = fill * (1.0 + tpc)
                cmult = 1.0
        # ---------------- S2 bids in armed hours with pm 0 (from what the core left)
        if s2_bids and armed and cash > 1e-9:
            held2 = 0
            for k in range(K):
                if units[1, k] > 0.0:
                    held2 += 1
            m = 0
            for k in range(K):
                d2a[k] = 1e18
                if units[1, k] > 0.0 or t < ok2[k] or np.isnan(o[t, k]) or m2[t, k] <= 0.0 or np.isnan(R2[t, k]):
                    continue
                dd = o[t, k] / (R2[t, k] * (1.0 - d2)) - 1.0
                if dd > near2:
                    continue
                d2a[k] = dd; m += 1
            free = min(n2 - held2, m)
            if free > 0:
                order = np.argsort(d2a)
                per = min(fs2 * eq_prev, cash / free)
                for q in range(free):
                    k = order[q]
                    lim = R2[t, k] * (1.0 - d2)
                    fill = -1.0; ismkt = 0
                    if o[t, k] <= lim:
                        fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                    elif l[t, k] <= lim * (1.0 - eps):
                        fill = lim * (1.0 + pen)
                    if fill > 0.0 and cash > 1e-9:
                        a = min(per, cash)
                        cash -= a; units[1, k] = a * (1.0 - fee) / fill
                        entry[1, k] = fill; amt0[1, k] = a; ebar[1, k] = t; eq0[1, k] = eq_prev
                        tpp[1, k] = fill * (1.0 + tp2); mkt[1, k] = ismkt if mtp_now else 0
        # ---------------- exits
        for s in range(2):
            for k in range(K):
                if units[s, k] > 0.0 and (ebar[s, k] < t or mkt[s, k] == 1) and not np.isnan(o[t, k]):
                    px = -1.0; why = 0.0
                    if ebar[s, k] == t:
                        if h[t, k] >= tpp[s, k] * (1.0 + eps):
                            px = tpp[s, k]
                    elif o[t, k] >= tpp[s, k]:
                        px = o[t, k]
                    elif h[t, k] >= tpp[s, k] * (1.0 + eps):
                        px = tpp[s, k]
                    age = t - ebar[s, k] + mkt[s, k]
                    if px < 0.0 and age >= hmv[s]:
                        px = c[t, k] * (1.0 - slip[k]); why = 1.0
                    if px > 0.0:
                        proceeds = units[s, k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds
                        nl = _log(lg, nl, ebar[s, k], t, k, why, proceeds, amt0[s, k], entry[s, k], eq0[s, k], s)
                        units[s, k] = 0.0
                        win = proceeds > amt0[s, k]
                        if s == 0:
                            okc[k] = t + cdc + cd_add
                            if not win:
                                pause2 = t + 1 + pause_cl
                        else:
                            ok2[k] = t + cd2 + cd_add
                            if not win:
                                cmult = cut_s2
            for k in range(K):
                if units[s, k] > 0.0 and lastb[k] == t:
                    proceeds = units[s, k] * c[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee); cash += proceeds
                    nl = _log(lg, nl, ebar[s, k], t, k, 5.0, proceeds, amt0[s, k], entry[s, k], eq0[s, k], s)
                    units[s, k] = 0.0
                    if s == 0:
                        okc[k] = t + cdc + cd_add
                    else:
                        ok2[k] = t + cd2 + cd_add
        # ---------------- hourly mark
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = cash; w = 0.0
        for s in range(2):
            for k in range(K):
                if units[s, k] > 0.0:
                    v += units[s, k] * lastc[k]; w += units[s, k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0.0 else 0.0
    for s in range(2):
        for k in range(K):
            if units[s, k] > 0.0:
                proceeds = units[s, k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                nl = _log(lg, nl, ebar[s, k], n - 1, k, 9.0, proceeds, amt0[s, k], entry[s, k], eq0[s, k], s)
    return eq, lg[:nl].copy(), inv


# ------------------------------------------------------------------ S2 masks (closed candles only)
_cache = {}


def _ckey(P, *a):
    return (id(P["c"]),) + a


def sma_up(P, days):
    """1 where the coin's last closed DAILY close is above its `days`-day SMA (daily closes at 23:00 UTC hours),
    known from the next hour on; unknown -> 0."""
    key = _ckey(P, "sma", days)
    if key not in _cache:
        cf = P["cf"]
        dc = cf[cf.index.hour == 23]
        up = (dc > dc.rolling(days, min_periods=days).mean()).astype(float)
        up.index = up.index + pd.Timedelta(hours=1)                    # usable from the next day's 00:00 open
        _cache[key] = up.reindex(cf.index).ffill().fillna(0.0).values
    return _cache[key]


def calm(P, sym, x, W=24):
    """1 where `sym`'s last close is at most x under its highest close of the last W closed hours (all coins)."""
    key = _ckey(P, "calm", sym, x, W)
    if key not in _cache:
        cf = P["cf"]
        if sym not in cf.columns:
            v = np.ones(len(cf))
        else:
            s = cf[sym]
            v = ((s / s.rolling(W, min_periods=1).max() - 1.0) >= -x).shift(1, fill_value=True).values.astype(float)
        _cache[key] = np.repeat(v[:, None], cf.shape[1], axis=1)
    return _cache[key]


S2_DEF = dict(ref=24, d2=0.12, tp2=0.015, hm2=48, cd2=24, fs2=0.2, n2=8, near2=1e8, coins=None, guard2=0.35,
              sma=0, calm=None)
CORE = dict(dc=0.14, tpc=0.025, hmc=96, cdc=3, fsc=1.0, guard=0.35)


def crun(P, idx, s2=None, armx=0.03, pm=1, watch=0, pause_cl=0, cut_s2=1.0, core_on=1, pen=0.0, pens=0.0,
         mtp_now=False, cd_add=0, liquidate=False, t0=None, core=None):
    """One account: core (90% mode) + optional S2 (dict over S2_DEF; None = off). t0: first traded hour (Timestamp)."""
    C = dict(CORE, **(core or {}))
    O = P["o"].values
    n, K = O.shape
    syms = list(P["o"].columns)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    gm = np.ascontiguousarray(V.guard_mask(P, C["guard"]))
    Rc = V.ref_panel(P, 24)
    s2_on = 0 if s2 is None else 1
    S = dict(S2_DEF, **(s2 or {}))
    R2 = np.ascontiguousarray(V.ref_panel(P, S["ref"]))
    m2 = np.ones((n, K))
    if s2_on:
        if S["coins"] is not None:
            m2 = m2 * np.array([1.0 if s in S["coins"] else 0.0 for s in syms])[None, :]
        if S["guard2"] > 0:
            key = _ckey(P, "g", S["guard2"])
            if key not in _cache:
                _cache[key] = V.guard_mask(P, S["guard2"])
            m2 = m2 * _cache[key]
        if S["sma"] > 0:
            m2 = m2 * sma_up(P, S["sma"])
        if S["calm"] is not None:
            m2 = m2 * calm(P, *S["calm"])
    m2 = np.ascontiguousarray(m2)
    it0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = ck(O, P["h"].values, P["l"].values, P["cf"].values, gm, np.ascontiguousarray(Rc), R2, m2, lastb,
                     float(C["dc"]), float(C["tpc"]), int(C["hmc"]), int(C["cdc"]), float(C["fsc"]), int(core_on),
                     s2_on, float(S["d2"]), float(S["tp2"]), int(S["hm2"]), int(S["cd2"]), float(S["fs2"]), int(S["n2"]),
                     float(S["near2"]), float(armx), int(pm), int(watch), int(pause_cl), float(cut_s2),
                     E.EPS, E.FEE, sl, float(pen), float(pens), bool(mtp_now), int(cd_add), it0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, idx)


def summarize(e, lg, inv, idx, periods=V.MAIN):
    """m10_velocity_lib.summarize plus per-sleeve counts (tier column = sleeve)."""
    lg2 = lg.rename(columns={"sl": "tier"})
    return V.summarize(e, lg2, inv, idx, periods)


def evaluate(kw, stress=False, worse=False, holdout=False, dead=None):
    """Same conventions as m10_velocity_lib.evaluate: worse = every buy and sell 0.2% worse; stress = real bot timing
    (mtp_now, cooldowns +1h) + fills 0.2% worse + LUNA/FTT/SRM sold at their last hour."""
    dead = stress if dead is None else dead
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    k = dict(kw)
    if stress:
        k.update(mtp_now=True, cd_add=1, pen=STRESS_PEN, pens=STRESS_PEN, liquidate=True)
    if worse:
        k.update(pen=STRESS_PEN, pens=STRESS_PEN)
    if dead and not stress:
        k["liquidate"] = True
    e, lg, inv = crun(P, idx, **k)
    return summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg, inv


def fmt(tag, p, r):
    lab = {"d": "2021-23", "t": "2024-26", "s": "since 2025", "h": "2018-20 holdout"}
    s = (f"  {tag:9s} {lab[p]:16s} win {r[f'{p}_wr']*100:5.1f}% | {r[f'{p}_mo']*100:+6.2f}%/mo | max drop "
         f"{r[f'{p}_dd']*100:6.1f}% | worst trade {r[f'{p}_wt']*100:+6.1f}% of acct | {r[f'{p}_tpm']:5.1f} trades/mo")
    if f"{p}_n1" in r:
        s += f" | core {r.get(f'{p}_n0', 0)} @ {r.get(f'{p}_wr0', np.nan)*100:4.1f}% / S2 {r[f'{p}_n1']} @ {r[f'{p}_wr1']*100:4.1f}%"
    return s
