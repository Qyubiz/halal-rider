"""m10 "meanrev": intraday mean reversion on 5/15-minute candles for the 19 live coins (Binance spot, fully funded).

Bars: C-minute OHLC built from the 1-minute npz files exactly like engine.minute_panel / x10_cycle_lib.build_bars
(hour-aligned global minute index, missing minutes flat at the last close, RENDER joined to RNDR; holdout: BCH joined
to BCHABC), one coin at a time, cached in PB_OUT. No volume in the 1m files, so no VWAP: "mean" = SMA/EMA of closes.

Kernel (mrkernel): a bot that checks at every bar open (C minutes):
  - entries: for every flat coin out of cooldown with a finite limit price lim[t,k] (computed from CLOSED bars only:
    every indicator is shifted one bar), rank by rank[t,k] (default o/lim-1: closest to its bid first), put fully
    funded resting bids on the best `nord` (each min(fmax*equity, cash/nsel), never more than free cash, never more
    than `maxpos` open positions). A bid fills at the open (market, +slippage) if the bar opens at/below it, else at
    lim if the bar's low trades eps (0.1%) through it. A "market signal" is lim = +inf (always fills at the open).
  - exits: take-profit is placed at the next check after the fill (never in the fill bar): fixed entry*(1+tp), or
    (tpmode 1) the moving mean tgt[t,k] clipped to [entry*(1+tp), entry*(1+tpmax)]. It fills when the high trades
    eps through it (at the target), or at the open if a newly placed / moved order is already below the market.
    Optional stop-loss at a bar close <= entry*(1-sl) and a time stop after hm bars, both sold at the close -slippage.
  - cash from a sale is usable from the next bar; a sold coin can be bought again cd bars later.
  - pen / pens: every buy / sell fill that much worse (stress test).
  - lossblock: after any losing exit, no new buys for that many bars (0 = off).
Fees 0.1% per side, market slippage 0.02% (all 19 coins are top-50 by volume).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/meanrev"))
os.makedirs(OUT, exist_ok=True)
FEE, EPS, SLIP = 0.001, 0.001, 0.0002
T21, T24, T25, TEND = (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01"),
                       pd.Timestamp("2026-10-23"))
H18, H21 = pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")
LOG = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]      # why: 0 take-profit, 1 time stop, 2 stop-loss


# ------------------------------------------------------------------ bars
def _read1m(path):
    z = np.load(path, allow_pickle=False)
    return pd.DataFrame({f: z[f].astype(np.float64) for f in ("o", "h", "l", "c")},
                        index=pd.to_datetime(z["t"], unit="s"))


def _coin(src, s, joins):
    p = os.path.join(src, f"{s}.npz")
    d = _read1m(p) if os.path.exists(p) else None
    j = joins.get(s)
    if j and os.path.exists(os.path.join(src, f"{j}.npz")):
        r = _read1m(os.path.join(src, f"{j}.npz"))
        d = r if d is None else pd.concat([r, d])
        d = d[~d.index.duplicated(keep="last")].sort_index()
    return d


def _build(src, syms, joins, start, end, C, path):
    D0 = {}
    for s in syms:
        d = _coin(src, s, joins)
        if d is None:
            continue
        d = d[(d.index >= start) & (d.index < end)]
        if len(d) > 60 * 24 * 30:
            D0[s] = (d.index[0], d.index[-1])
    syms = [s for s in syms if s in D0]
    t0 = min(v[0] for v in D0.values()).floor("1h")
    t1 = max(v[1] for v in D0.values()).floor("1h") + pd.Timedelta(minutes=59)
    idx = pd.date_range(t0, t1, freq="1min")
    nb = len(idx) // C
    out = {f: np.full((nb, len(syms)), np.nan) for f in ("o", "h", "l", "c")}
    for j, s in enumerate(syms):
        d = _coin(src, s, joins)
        d = d[(d.index >= start) & (d.index < end)].reindex(idx)
        first = d.c.first_valid_index()
        cc = d.c.ffill()
        live = idx >= first
        A = {}
        for f in ("o", "h", "l"):
            A[f] = np.where(live, d[f].where(d[f].notna(), cc).values, np.nan)
        A["c"] = np.where(live, cc.values, np.nan)
        del d, cc
        out["o"][:, j] = A["o"][::C]
        out["c"][:, j] = A["c"][C - 1::C]
        out["h"][:, j] = np.fmax.reduce(A["h"].reshape(nb, C), axis=1)
        out["l"][:, j] = np.fmin.reduce(A["l"].reshape(nb, C), axis=1)
        del A
    np.savez(path, syms=np.array(syms), t=idx[::C].values.astype("datetime64[s]").astype(np.int64), **out)


def load(which="main", C=5):
    """5-minute bars (C=15 aggregates them). which: 'main' (19 coins, 2021-01 .. 2026-10, PB_DATA/1m) or 'hold'
    (coins that existed, 2017-12 .. 2020-12, PB_HOLDOUT/1m_pre)."""
    path = os.path.join(OUT, f"bars5_{which}.npz")
    if not os.path.exists(path):
        if which == "main":
            _build(os.path.join(E.DATA, "1m"), list(E.H19), {"RENDERUSDT": "RNDRUSDT"}, T21, TEND, 5, path)
        else:
            _build(os.path.join(E.HOLDOUT, "1m_pre"), list(E.H19), {"BCHUSDT": "BCHABCUSDT"},
                   pd.Timestamp("2017-12-01"), H21, 5, path)
    z = np.load(path, allow_pickle=False)
    B = {"syms": [str(x) for x in z["syms"]], "t": pd.to_datetime(z["t"], unit="s"),
         **{f: np.ascontiguousarray(z[f]) for f in ("o", "h", "l", "c")}}
    if C != 5:
        B = agg(B, C // 5)
    B["C"] = C
    B["cf"] = pd.DataFrame(B["c"]).ffill().values
    return B


def agg(B, m):
    n = (len(B["t"]) // m) * m
    K = B["o"].shape[1]
    r = lambda a: a[:n].reshape(n // m, m, K)   # noqa: E731
    return {"syms": B["syms"], "t": B["t"][:n:m], "o": B["o"][:n:m].copy(), "c": B["c"][m - 1:n:m].copy(),
            "h": np.fmax.reduce(r(B["h"]), axis=1), "l": np.fmin.reduce(r(B["l"]), axis=1)}


def cut(B, a, b):
    """Bars with a <= t < b (views)."""
    i, j = np.searchsorted(B["t"], a), np.searchsorted(B["t"], b)
    return {**{k: (v[i:j] if isinstance(v, np.ndarray) else v) for k, v in B.items() if k != "t"}, "t": B["t"][i:j]}


# ------------------------------------------------------------------ indicators (value at bar t uses bars <= t-1)
def _df(B, x=None):
    return pd.DataFrame(B["cf"] if x is None else x)


def sh(a):
    """Shift one bar: the value known at bar t's open."""
    out = np.empty_like(a)
    out[0] = np.nan
    out[1:] = a[:-1]
    return out


def sma(B, n):
    return sh(_df(B).rolling(n, min_periods=n).mean().values)


def std(B, n):
    return sh(_df(B).rolling(n, min_periods=n).std(ddof=0).values)


def ema(B, n):
    return sh(_df(B).ewm(span=n, adjust=False, min_periods=n).mean().values)


def rollmax(B, n):
    return sh(_df(B).rolling(n, min_periods=1).max().values)


def atr(B, n):
    pc = np.vstack([B["cf"][:1], B["cf"][:-1]])
    tr = np.fmax(B["h"] - B["l"], np.fmax(np.abs(B["h"] - pc), np.abs(B["l"] - pc)))
    return sh(_df(B, tr).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values)


def rsi(B, n):
    d = np.diff(B["cf"], axis=0, prepend=np.nan)
    up = _df(B, np.where(d > 0, d, 0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values
    dn = _df(B, np.where(d < 0, -d, 0.0)).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean().values
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 - 100.0 / (1.0 + up / dn)
    r = np.where(dn == 0, 100.0, r)
    r[np.isnan(B["cf"])] = np.nan
    return sh(r)


def red_run(B):
    """Consecutive red bars (close < open) ending at the last closed bar."""
    red = (B["c"] < B["o"]).astype(np.int32)
    out = np.zeros_like(red)
    for t in range(1, len(red)):
        out[t] = (out[t - 1] + 1) * red[t]
    return sh(out.astype(np.float64))


def ret_over(B, n):
    """Close-to-close return over the last n closed bars."""
    c = B["cf"]
    r = np.full_like(c, np.nan)
    r[n:] = c[n:] / c[:-n] - 1.0
    return sh(r)


def bars_per(B, hours):
    return int(round(hours * 60 / B["C"]))


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def mrkernel(o, h, l, c, lim, rank, tgt, nord, fmax, maxpos, tp, tpmax, tpmode, hm, cd, sl, eps, fee, slip,
             pen, pens, maxlog, lossblock):
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0; npos = 0
    lg = np.zeros((maxlog, 8)); nl = 0; over = 0
    userank = rank.shape[0] > 0
    usetgt = tgt.shape[0] > 0
    blocked = -1
    for t in range(n):
        anyfill = False
        if npos < maxpos and cash > 1e-9 and t >= blocked:
            for k in range(K):
                dist[k] = 1e9
                if units[k] > 0.0 or t < okbar[k]:
                    continue
                L = lim[t, k]
                if not (L > 0.0) or np.isnan(o[t, k]):
                    continue
                dist[k] = rank[t, k] if userank else o[t, k] / L - 1.0
                if o[t, k] <= L or l[t, k] <= L * (1.0 - eps):
                    anyfill = True
        if anyfill:          # no bid trades through: nothing can fill this bar (exact shortcut)
            order = np.argsort(dist)
            nsel = 0
            for q in range(K):
                if dist[order[q]] < 1e8 and nsel < nord and npos + nsel < maxpos:
                    nsel += 1
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                L = lim[t, k]
                fill = -1.0
                if o[t, k] <= L:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                elif l[t, k] <= L * (1.0 - eps):
                    fill = L * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; eq0[k] = eq_prev
                    npos += 1
        for k in range(K):
            if units[k] <= 0.0 or ebar[k] >= t or np.isnan(o[t, k]):
                continue
            px = -1.0; why = 0.0
            T = entry[k] * (1.0 + tp)
            fresh = t == ebar[k] + 1
            if tpmode == 1 and usetgt:
                m = tgt[t, k]
                if m > T:
                    T = m
                if T > entry[k] * (1.0 + tpmax):
                    T = entry[k] * (1.0 + tpmax)
                fresh = True
            if o[t, k] >= T:
                px = o[t, k] if fresh else T
            elif h[t, k] >= T * (1.0 + eps):
                px = T
            if px < 0.0 and sl > 0.0 and c[t, k] <= entry[k] * (1.0 - sl):
                px = c[t, k] * (1.0 - slip[k]); why = 2.0
            if px < 0.0 and t - ebar[k] >= hm:
                px = c[t, k] * (1.0 - slip[k]); why = 1.0
            if px > 0.0:
                proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee)
                cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                    lg[nl, 7] = eq0[k]; nl += 1
                else:
                    over += 1
                units[k] = 0.0; okbar[k] = t + cd; npos -= 1
                if lossblock > 0 and proceeds < amt0[k]:
                    blocked = t + lossblock
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy(), over


EMPTY = np.empty((0, 0))


def run(B, lim, nord=1, fmax=1.0, maxpos=99, tp=0.01, tpmax=1.0, tgt=None, hm=288, cd=0, sl=0.0, rank=None,
        pen=0.0, pens=0.0, fee=FEE, eps=EPS, slip=SLIP, maxlog=1500000, lossblock=0):
    K = B["o"].shape[1]
    eq, lg, over = mrkernel(B["o"], B["h"], B["l"], B["cf"], lim, EMPTY if rank is None else rank,
                            EMPTY if tgt is None else tgt, int(nord), float(fmax), int(maxpos), float(tp),
                            float(tpmax), 1 if tgt is not None else 0, int(hm), int(cd), float(sl), float(eps),
                            float(fee), np.full(K, slip), float(pen), float(pens), int(maxlog), int(lossblock))
    assert over == 0, "trade log overflow"
    return eq, lg


# ------------------------------------------------------------------ metrics
def stats(B, eq, lg, a, b):
    """Window [a, b): multiple, %/month (compounded), max drop (bar closes), trades entered in the window, win
    rate, average trade, worst trade as % of the account, trades/month. Equity rebased at the bar before a."""
    t = B["t"]
    i, j = int(np.searchsorted(t, a)), int(np.searchsorted(t, b))
    if j <= i:
        return None
    s0 = eq[i - 1] if i > 0 else 1.0
    e = np.concatenate([[1.0], eq[i:j] / s0])
    pk = np.maximum.accumulate(e)
    dd = float((e / pk - 1.0).min())
    days = (t[j - 1] - t[i]).total_seconds() / 86400 + B["C"] / 1440
    months = days / 30.4375
    mult = float(e[-1])
    r = {"mult": mult, "mo": (mult ** (1 / months) - 1) if mult > 0 else -1.0, "dd": dd, "months": months}
    hend = np.nonzero(((t[i:j] + pd.Timedelta(minutes=B["C"])).minute == 0))[0] + 1     # equity at hour closes
    eh = np.concatenate([[1.0], e[hend]])
    r["dd_h"] = float((eh / np.maximum.accumulate(eh) - 1.0).min())
    m = (lg[:, 0] >= i) & (lg[:, 0] < j) if len(lg) else np.zeros(0, bool)
    tr = lg[m]
    r["n"] = int(len(tr))
    r["tpm"] = len(tr) / months
    if len(tr):
        acct = tr[:, 4] * tr[:, 5] / tr[:, 7]
        r.update(win=float((tr[:, 4] > 0).mean()), avg=float(tr[:, 4].mean()), worst=float(tr[:, 4].min()),
                 worst_acct=float(acct.min()), tstop=float((tr[:, 3] == 1).mean()),
                 hold_h=float((tr[:, 1] - tr[:, 0]).mean() * B["C"] / 60))
    else:
        r.update(win=np.nan, avg=np.nan, worst=np.nan, worst_acct=np.nan, tstop=np.nan, hold_h=np.nan)
    return r


def yearly(B, eq):
    s = pd.Series(eq, B["t"]).resample("YE").last()
    prev = np.concatenate([[1.0], s.values[:-1]])
    return {k.year: float(v / p - 1) for k, v, p in zip(s.index, s.values, prev)}


def fmt(tag, r):
    if r is None:
        return f"{tag}: -"
    return (f"{tag:9s} {r['mo']*100:+6.2f}%/mo x{r['mult']:7.2f} dd {r['dd']*100:5.1f}% | n {r['n']:5d} "
            f"({r['tpm']:5.1f}/mo) win {r['win']*100:5.1f}% avg {r['avg']*100:+.2f}% worst {r['worst']*100:+5.1f}% "
            f"acct {r['worst_acct']*100:+5.1f}% tstop {r['tstop']*100:4.1f}% hold {r['hold_h']:5.1f}h")
