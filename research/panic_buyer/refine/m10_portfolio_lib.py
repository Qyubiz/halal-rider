"""m10 'portfolio' angle: several uncorrelated SPOT sleeves sharing one account (no leverage, no shorting, fully
funded, cash idle = 0% interest).

Sleeves (each has its own cash; the account equity is their sum):
  DIP   Panic Buyer 90% mode (ens_lib.MODE90 + 35% crash guard), the ens_lib.ekernel rules unchanged (hourly).
  MOM   a simplified halal-rider (rider.py) on 5-minute candles built from the 1-minute data:
        BTC price above the EMA50 of its completed daily closes; closed 5m candle up >= 2% over 15 min and green;
        (vmode 1) the candle's range >= 3x the average range of the 48 candles before it, standing in for rider.py's
        "volume >= 3x the 4h average" (the shared 1-minute files have no volume); stop-buy 1% above the signal close
        (limit 0.3% above that), valid 60 min; stop-loss order 5% under entry, raised at every 5-minute cycle to 8%
        under the highest closed 5m high since the fill; market sale after 24h; at most 5 trades (armed or open),
        each min(1/5 of the sleeve, 98% of its free cash). Not modelled: the -10%-in-a-week pause.
  TREND BTC and/or ETH held while the daily close is above its N-day SMA (or SMA50 > SMA200); decided on completed
        daily closes, traded at the next hour's open (market, +0.02% slippage).
  CASH  USDT, earns nothing.

Monthly rebalancing (rb=2, fixed before any result): at the first hour of each month, every sleeve's target is
w x account equity at the last close. The TREND sleeve is trimmed (sold at the open) or topped up to its target;
the DIP and MOM sleeves only give FREE cash (open positions and armed orders are never sold for it); CASH gives/takes
anything. rb=1: free cash only for all sleeves; rb=0: never (independent bots).

Costs: 0.1% fee per side everywhere, market orders +0.02% slippage, resting limit buys of the dip-buyer need a 0.1%
trade-through (engine.EPS). pen / pens make every buy / sell that much worse in every sleeve (stress: 0.002 each).
No lookahead: every signal uses only candles closed before the order time.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import ens_lib as L  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/portfolio"))
os.makedirs(OUT, exist_ok=True)
MOM_DEF = dict(rise=0.02, look=3, vmode=1, vm=3.0, vlook=48, confirm=0.01, lslip=0.003, stop=0.05, trail=0.08,
               maxh=24, restep=0.003, slots=5)
PERIODS = (("2018-20", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")),
           ("2021-23", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
           ("2024-26", pd.Timestamp("2024-01-01"), None),
           ("since 2025", pd.Timestamp("2025-01-01"), None))


# ------------------------------------------------------------------ data
_mk = {}


def market(holdout=False):
    """(syms, idx, P) hourly panel: 19 coins 2021-01-01.. (engine.load_h19) or the 2018-20 holdout coins from
    2017-12-01 (engine.load_pre)."""
    if holdout not in _mk:
        _mk[holdout] = E.panel(E.load_pre()) if holdout else E.panel(E.load_h19())
    return _mk[holdout]


_dc = {}


def daily_close(sym):
    """Completed UTC daily closes, holdout file (Dec 2017 - 2020) joined to the main file (2021 -)."""
    if sym not in _dc:
        _dc[sym] = _daily_close(sym)
    return _dc[sym]


def _daily_close(sym):
    a = os.path.join(E.HOLDOUT, "1h_pre", f"{sym}.csv")
    parts = []
    if os.path.exists(a):
        d = pd.read_csv(a, index_col=0)
        d.index = pd.to_datetime(d.index, format="mixed")
        parts.append(d.c)
    b = E.read_hourly(sym)
    if b is not None:
        parts.append(b.c)
    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.resample("D").last().dropna()


def _read1m(path):
    z = np.load(path, allow_pickle=False)
    return z["t"].astype(np.int64), z["o"], z["h"], z["l"], z["c"]


def load_1m(sym, holdout):
    """1-minute (t, o, h, l, c) for one coin; RENDER joined to RNDR, BCH to BCHABC (holdout)."""
    root = os.path.join(E.HOLDOUT, "1m_pre") if holdout else os.path.join(E.DATA, "1m")
    alias = {"RENDERUSDT": "RNDRUSDT", "BCHUSDT": "BCHABCUSDT"}
    parts = []
    if sym in alias and os.path.exists(os.path.join(root, alias[sym] + ".npz")):
        parts.append(_read1m(os.path.join(root, alias[sym] + ".npz")))
    p = os.path.join(root, sym + ".npz")
    if os.path.exists(p):
        parts.append(_read1m(p))
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    t = np.concatenate([x[0] for x in parts])
    cols = [np.concatenate([x[i] for x in parts]) for i in range(1, 5)]
    # the later file wins where both exist
    order = np.argsort(t, kind="stable")
    t = t[order]; cols = [c[order] for c in cols]
    keep = np.ones(len(t), bool)
    keep[:-1] = t[:-1] != t[1:]
    return (t[keep],) + tuple(c[keep] for c in cols)


@njit(cache=True)
def _ffill_flat(o, h, l, c):
    """Missing minutes after the first trade: flat at the last close (engine.minute_panel convention)."""
    last = np.nan
    for i in range(len(c)):
        if np.isnan(c[i]):
            if not np.isnan(last):
                o[i] = last; h[i] = last; l[i] = last; c[i] = last
        else:
            last = c[i]


def grid_1m(sym, holdout, t0, nmin):
    """Minute arrays on the global grid t0 + i minutes (float64), NaN before the coin trades."""
    x = load_1m(sym, holdout)
    A = [np.full(nmin, np.nan) for _ in range(4)]
    if x is None:
        return A
    t = x[0]
    gi = (t - t0) // 60
    ok = (gi >= 0) & (gi < nmin) & (t % 60 == 0)
    for j in range(4):
        A[j][gi[ok]] = x[j + 1][ok]
    _ffill_flat(A[0], A[1], A[2], A[3])
    return A


# ------------------------------------------------------------------ MOM sleeve: per-signal outcomes on 1-minute data
@njit(cache=True)
def mom_coin(o1, h1, l1, c1, btcup5, rise, look, vmode, vm, vlook, confirm, lslip, stop, trail, maxmin, restep,
             slip):
    """Every rider signal of one coin and what its order/trade would do (independent of the portfolio).
    Minute m = the minute starting at t0 + m. The bot's cycle runs just after each 5-minute boundary C (minute C).
    Returns rows: c0 (signal cycle minute), fill minute (-1 none), fill price (incl. slippage), exit minute,
    exit price (incl. slippage), det (cycle at which the bot frees the slot), why (1 stop/trail, 2 time, 3 end,
    4 sold at stop placement), and the 5m index."""
    nm = len(c1)
    n5 = nm // 5
    o5 = np.full(n5, np.nan); h5 = np.full(n5, np.nan); l5 = np.full(n5, np.nan); c5 = np.full(n5, np.nan)
    for i in range(n5):
        m = 5 * i
        if np.isnan(o1[m]) or np.isnan(c1[m + 4]):
            continue
        o5[i] = o1[m]; c5[i] = c1[m + 4]
        hh = h1[m]; ll = l1[m]
        for j in range(m + 1, m + 5):
            if h1[j] > hh:
                hh = h1[j]
            if l1[j] < ll:
                ll = l1[j]
        h5[i] = hh; l5[i] = ll
    cap = 200000
    out = np.full((cap, 8), -1.0)
    no = 0
    for i in range(max(look, vlook), n5 - 1):
        if np.isnan(c5[i]) or np.isnan(c5[i - look]) or btcup5[i] == 0:
            continue
        if c5[i] / c5[i - look] - 1.0 < rise or c5[i] <= o5[i]:
            continue
        if vmode == 1:
            sr = 0.0; nr = 0
            for j in range(i - vlook, i):
                if not np.isnan(c5[j]):
                    sr += (h5[j] - l5[j]) / c5[j]; nr += 1
            if nr < vlook // 2 or sr <= 0.0:
                continue
            if (h5[i] - l5[i]) / c5[i] < vm * sr / nr:
                continue
        C0 = 5 * (i + 1)
        trig = c5[i] * (1.0 + confirm)
        lim = trig * (1.0 + lslip)
        fill = -1.0; fm = -1; trg = False
        for j in range(C0, min(C0 + 60, nm)):
            if np.isnan(o1[j]):
                continue
            if not trg:
                if o1[j] >= trig:
                    trg = True
                    if o1[j] <= lim:
                        fill = min(o1[j] * (1.0 + slip), lim); fm = j
                        break
                    if l1[j] <= lim:                   # gapped over the limit, came back to it
                        fill = lim; fm = j
                        break
                elif h1[j] >= trig:                    # stop touched: bought at the trigger (+ slippage)
                    trg = True
                    fill = trig * (1.0 + slip); fm = j
                    break
            elif l1[j] <= lim:
                fill = lim; fm = j
                break
        if no >= cap:
            break
        out[no, 0] = C0; out[no, 7] = i
        if fm < 0:
            out[no, 1] = -1.0; out[no, 5] = C0 + 60
            no += 1
            continue
        out[no, 1] = fm; out[no, 2] = fill
        B = (fm // 5 + 1) * 5                          # the cycle that notices the fill and places the stop
        stp = fill * (1.0 - stop); peak = fill
        xm = -1; xp = -1.0; det = -1; why = 0
        if B >= nm:
            xm = nm - 1; xp = c1[nm - 1] * (1.0 - slip); det = nm - 1; why = 3
        elif o1[B] <= stp:
            xm = B; xp = o1[B] * (1.0 - slip); det = B; why = 4
        else:
            last5 = B // 5 - 1                         # rider: candles with open > (floor5(now) - 5 min)
            m = B
            while m < nm:
                if m % 5 == 0:                         # bot cycle at this boundary (before the minute trades)
                    if m - B >= maxmin:
                        xm = m; xp = o1[m] * (1.0 - slip); det = m; why = 2
                        break
                    ci = m // 5 - 1
                    if ci > last5:
                        if h5[ci] > peak:
                            peak = h5[ci]
                        last5 = ci
                    if o1[m] > peak:
                        peak = o1[m]
                    want = max(stp, peak * (1.0 - trail))
                    if want >= stp * (1.0 + restep):
                        if o1[m] <= want:
                            xm = m; xp = o1[m] * (1.0 - slip); det = m; why = 4
                            break
                        stp = want
                if l1[m] <= stp:                       # resting stop-loss order on the exchange
                    xm = m; xp = min(o1[m], stp) * (1.0 - slip); det = (m // 5 + 1) * 5; why = 1
                    break
                m += 1
            if xm < 0:
                xm = nm - 1; xp = c1[nm - 1] * (1.0 - slip); det = nm - 1; why = 3
        out[no, 3] = xm; out[no, 4] = xp; out[no, 5] = det; out[no, 6] = why
        no += 1
    return out[:no].copy()


def btc_up5(t0, nmin, holdout, days=50):
    """Per 5-minute candle i (closing at t0 + 5(i+1) min): BTC's last price > EMA(days) of its completed daily
    closes (rider.btc_uptrend). Unknown -> 0."""
    b = grid_1m("BTCUSDT", holdout, t0, nmin)[3]
    n5 = nmin // 5
    px = b[4::5][:n5]
    dc = daily_close("BTCUSDT")
    ema = dc.ewm(span=days, adjust=False).mean()
    close_t = pd.to_datetime(t0, unit="s") + pd.to_timedelta(5 * (np.arange(n5) + 1), unit="min")
    prev_day = close_t.floor("D") - pd.Timedelta(days=1)
    ev = ema.reindex(prev_day).values
    up = (px > ev) & ~np.isnan(ev) & ~np.isnan(px)
    return up.astype(np.int64)


def mom_events(holdout=False, prm=None, cache=True):
    """All MOM signals of the panel's coins, sorted by (cycle minute, coin order). Columns: c0, k, fm, fpx, xm, xpx,
    det, why (minute indices from the panel's first hour)."""
    prm = dict(MOM_DEF, **(prm or {}))
    syms, idx, P = market(holdout)
    key = "_".join(f"{k}{prm[k]}" for k in sorted(prm) if k != "slots")
    path = os.path.join(OUT, f"mom_events_{'H' if holdout else 'M'}_{key}.npy")
    if cache and os.path.exists(path):
        return np.load(path)
    t0 = int(idx[0].value // 10**9)
    nmin = len(idx) * 60
    up = btc_up5(t0, nmin, holdout)
    rows = []
    for k, s in enumerate(syms):
        o1, h1, l1, c1 = grid_1m(s, holdout, t0, nmin)
        r = mom_coin(o1, h1, l1, c1, up, prm["rise"], prm["look"], prm["vmode"], prm["vm"], prm["vlook"],
                     prm["confirm"], prm["lslip"], prm["stop"], prm["trail"], prm["maxh"] * 60, prm["restep"],
                     E.SLIP_MAJOR)
        if len(r):
            ev = np.column_stack([r[:, 0], np.full(len(r), k), r[:, 1], r[:, 2], r[:, 3], r[:, 4], r[:, 5], r[:, 6]])
            rows.append(ev)
        del o1, h1, l1, c1
    ev = np.vstack(rows) if rows else np.zeros((0, 8))
    ev = ev[np.lexsort((ev[:, 1], ev[:, 0]))]
    np.save(path, ev)
    return ev


# ------------------------------------------------------------------ TREND sleeve: daily state per hour
def trend_target(idx, sym, rule, n=200, unknown_on=False):
    """1 while the asset should be held during the UTC day of each hour, from completed daily closes only.
    rule 'sma': close > SMA(n); 'cross': SMA50 > SMA200. unknown_on: treat a not-yet-computable average as 'hold'
    (pessimistic check for the start of the holdout, whose data begin 2017-12-01)."""
    dc = daily_close(sym)
    if rule == "sma":
        a = dc.rolling(n, min_periods=n).mean()
        st = (dc > a).astype(float).where(a.notna())
    else:
        a50, a200 = dc.rolling(50, min_periods=50).mean(), dc.rolling(200, min_periods=200).mean()
        st = (a50 > a200).astype(float).where(a200.notna())
    st = st.fillna(1.0 if unknown_on else 0.0)
    day = idx.floor("D") - pd.Timedelta(days=1)       # decided on the previous day's close
    v = st.reindex(day).values
    return np.nan_to_num(v, nan=1.0 if unknown_on else 0.0).astype(np.int64)


# ------------------------------------------------------------------ the joint one-account kernel
@njit(cache=True)
def _log(lg, nl, eb, xb, k, why, ret, amt, eq0, s):
    if nl < lg.shape[0]:
        lg[nl, 0] = eb; lg[nl, 1] = xb; lg[nl, 2] = k; lg[nl, 3] = why; lg[nl, 4] = ret; lg[nl, 5] = amt
        lg[nl, 6] = eq0; lg[nl, 7] = s
        return nl + 1
    return nl


@njit(cache=True)
def pkernel(o, h, l, c, ref, gm, hod, mstart,
            depth, nord, fmax, tpa, hm, cd,
            ev, slots,
            tk, taw, ttgt,
            w0, rb, eps, fee, slip, pen, pens, tpwait, mtp_now):
    """Sleeves 0..S-1: dip-buyers (ens_lib.ekernel rules, part=1, no guardrails); S: MOM; S+1: TREND; S+2: CASH.
    Returns account equity, sleeve equity (n, S+3) and the trade log (eb, xb, k, why, ret, amt, eq0, sleeve)."""
    n, K = c.shape
    S = depth.shape[0]
    NS = S + 3
    iM = S; iT = S + 1; iC = S + 2
    cash = np.empty(NS)
    for s in range(NS):
        cash[s] = w0[s]
    seqp = cash.copy()
    units = np.zeros((S, K)); entry = np.zeros((S, K)); amt0 = np.zeros((S, K)); eq0 = np.zeros((S, K))
    tpp = np.zeros((S, K)); ebar = np.full((S, K), -1, np.int64); okbar = np.zeros((S, K), np.int64)
    mkt = np.zeros((S, K), np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    # MOM state (at most `slots` live orders/trades)
    MA = 32
    aid = np.full(MA, -1, np.int64); ast = np.zeros(MA, np.int64); astake = np.zeros(MA); aun = np.zeros(MA)
    aeq0 = np.zeros(MA)
    nev = ev.shape[0]; p = 0
    # TREND state
    NA = tk.shape[0]
    tun = np.zeros(NA); tent = np.zeros(NA); tbar = np.full(NA, -1, np.int64); tamt = np.zeros(NA)
    teq0 = np.zeros(NA)
    eq = np.empty(n); seq = np.empty((n, NS))
    eq_prev = 0.0
    for s in range(NS):
        eq_prev += cash[s]
    maxlog = 800000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(n):
        # ------------- monthly rebalance
        if rb > 0 and mstart[t] == 1:
            tot = 0.0
            for s in range(NS):
                tot += seqp[s]
            if rb == 2:                                   # trim the trend holdings to their target first
                exc = seqp[iT] - w0[iT] * tot
                hv = 0.0
                for a in range(NA):
                    if tun[a] > 0.0 and not np.isnan(o[t, tk[a]]):
                        hv += tun[a] * o[t, tk[a]]
                sell = exc - cash[iT]
                if sell > 1e-12 and hv > 0.0:
                    fr = min(1.0, sell / hv)
                    for a in range(NA):
                        if tun[a] > 0.0 and not np.isnan(o[t, tk[a]]):
                            qq = tun[a] * fr
                            cash[iT] += qq * o[t, tk[a]] * (1.0 - slip[tk[a]]) * (1.0 - pens) * (1.0 - fee)
                            tun[a] -= qq
                            if fr >= 1.0 - 1e-12:
                                tun[a] = 0.0
            need = np.zeros(NS); give = np.zeros(NS); G = 0.0; NE = 0.0
            for s in range(NS):
                d = w0[s] * tot - seqp[s]
                if d < 0.0:
                    give[s] = min(cash[s], -d); G += give[s]
                else:
                    need[s] = d; NE += d
            moved = min(G, NE)
            if moved > 1e-12:
                for s in range(NS):
                    if give[s] > 0.0:
                        cash[s] -= give[s] * moved / G
                    if need[s] > 0.0:
                        cash[s] += need[s] * moved / NE
            if rb == 2:                                   # top the held trend assets up to their share
                ev_t = cash[iT]; sw = 0.0
                for a in range(NA):
                    sw += taw[a]
                    if tun[a] > 0.0 and not np.isnan(o[t, tk[a]]):
                        ev_t += tun[a] * o[t, tk[a]]
                for a in range(NA):
                    if tun[a] > 0.0 and not np.isnan(o[t, tk[a]]) and ttgt[t, a] == 1:
                        cur = tun[a] * o[t, tk[a]]
                        want = taw[a] / sw * ev_t
                        bb = min(cash[iT], want - cur)
                        if bb > 1e-9 * ev_t:
                            fpx = o[t, tk[a]] * (1.0 + slip[tk[a]]) * (1.0 + pen)
                            cash[iT] -= bb; tun[a] += bb * (1.0 - fee) / fpx
        # ------------- DIP sleeves: entries (ens_lib.ekernel)
        for s in range(S):
            for k in range(K):
                dist[k] = 1e9
                if units[s, k] > 0.0 or t < okbar[s, k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or gm[t, k] <= 0.0:
                    continue
                dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth[s])) - 1.0
            order = np.argsort(dist)
            nsel = 0
            for q in range(K):
                if dist[order[q]] < 1e8 and nsel < nord[s]:
                    nsel += 1
            if nsel > 0 and cash[s] > 1e-9:
                per = min(fmax[s] * seqp[s], cash[s] / nsel)
                for q in range(nsel):
                    k = order[q]
                    lim = ref[t, k] * (1.0 - depth[s])
                    fill = -1.0; ismkt = 0
                    if o[t, k] <= lim:
                        fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                    elif l[t, k] <= lim * (1.0 - eps):
                        fill = lim * (1.0 + pen)
                    if fill > 0.0 and cash[s] > 1e-9:
                        mkt[s, k] = ismkt if mtp_now else 0
                        av = min(per * gm[t, k], cash[s])
                        cash[s] -= av; units[s, k] = av * (1.0 - fee) / fill
                        entry[s, k] = fill; amt0[s, k] = av; ebar[s, k] = t; eq0[s, k] = eq_prev
                        tpp[s, k] = fill * (1.0 + tpa[s])
        # ------------- DIP sleeves: exits
        for s in range(S):
            for k in range(K):
                if units[s, k] <= 0.0 or np.isnan(o[t, k]):
                    continue
                px = -1.0; why = 0.0
                if ebar[s, k] < t or mkt[s, k] == 1:
                    if ebar[s, k] == t:
                        if h[t, k] >= tpp[s, k] * (1.0 + eps):
                            px = tpp[s, k]
                    elif t - ebar[s, k] < tpwait:
                        pass
                    elif o[t, k] >= tpp[s, k]:
                        px = o[t, k]
                    elif h[t, k] >= tpp[s, k] * (1.0 + eps):
                        px = tpp[s, k]
                    if px < 0.0 and t - ebar[s, k] >= hm[s] - mkt[s, k]:
                        px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    pr = units[s, k] * px * (1.0 - pens) * (1.0 - fee); cash[s] += pr
                    nl = _log(lg, nl, ebar[s, k], t, k, why, pr / amt0[s, k] - 1.0, amt0[s, k], eq0[s, k], s)
                    units[s, k] = 0.0; okbar[s, k] = t + cd[s]
        # ------------- MOM sleeve: events inside this hour, in minute order
        m_hi = 60 * (t + 1)
        while True:
            T = m_hi
            newsig = p < nev and ev[p, 0] < m_hi
            if newsig:
                T = int(ev[p, 0])
            # advance live orders/trades to minute T
            for a in range(MA):
                if aid[a] < 0:
                    continue
                e_ = aid[a]
                if ast[a] == 0:
                    fmn = ev[e_, 2]
                    if fmn >= 0.0 and fmn < T:
                        ast[a] = 1
                        aun[a] = astake[a] * (1.0 - fee) / (ev[e_, 3] * (1.0 + pen))
                    elif fmn < 0.0 and ev[e_, 0] + 60 <= T:
                        cash[iM] += astake[a]; aid[a] = -1
                        continue
                if ast[a] == 1 and (ev[e_, 4] < T or ev[e_, 6] <= T):
                    pr = aun[a] * ev[e_, 5] * (1.0 - pens) * (1.0 - fee)
                    cash[iM] += pr
                    nl = _log(lg, nl, int(ev[e_, 2]) // 60, int(ev[e_, 4]) // 60, int(ev[e_, 1]), ev[e_, 7],
                              pr / astake[a] - 1.0, astake[a], aeq0[a], iM)
                    ast[a] = 2
                if ast[a] == 2 and ev[e_, 6] <= T:
                    aid[a] = -1
            if not newsig:
                break
            k = int(ev[p, 1])
            nact = 0; busy = False; free = -1
            for a in range(MA):
                if aid[a] >= 0:
                    nact += 1
                    if int(ev[aid[a], 1]) == k:
                        busy = True
                elif free < 0:
                    free = a
            if (not busy) and nact < slots and w0[iM] > 0.0:
                stake = min(seqp[iM] / slots, cash[iM] * 0.98)
                if stake > 1e-12:
                    cash[iM] -= stake
                    aid[free] = p; ast[free] = 0; astake[free] = stake; aun[free] = 0.0; aeq0[free] = eq_prev
            p += 1
        # ------------- TREND sleeve
        for a in range(NA):
            k = tk[a]
            if taw[a] <= 0.0 or np.isnan(o[t, k]):
                continue
            if tun[a] > 0.0 and ttgt[t, a] == 0:
                pr = tun[a] * o[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                cash[iT] += pr
                ret = o[t, k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee) * (1.0 - fee) / tent[a] - 1.0
                nl = _log(lg, nl, tbar[a], t, k, 6.0, ret, tamt[a], teq0[a], iT)
                tun[a] = 0.0
        for a in range(NA):
            k = tk[a]
            if taw[a] <= 0.0 or np.isnan(o[t, k]):
                continue
            if tun[a] == 0.0 and ttgt[t, a] == 1 and w0[iT] > 0.0:
                sw = 0.0
                for a2 in range(NA):
                    sw += taw[a2]
                alloc = min(cash[iT], taw[a] / sw * seqp[iT])
                if alloc > 1e-12:
                    fpx = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                    cash[iT] -= alloc; tun[a] = alloc * (1.0 - fee) / fpx
                    tent[a] = fpx; tbar[a] = t; tamt[a] = alloc; teq0[a] = eq_prev
        # ------------- marks
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
        v = 0.0
        for s in range(S):
            sv = cash[s]
            for k in range(K):
                if units[s, k] > 0.0:
                    sv += units[s, k] * lastc[k]
            seqp[s] = sv; v += sv
        sv = cash[iM]
        for a in range(MA):
            if aid[a] >= 0:
                if ast[a] == 0:
                    sv += astake[a]
                elif ast[a] == 1:
                    sv += aun[a] * lastc[int(ev[aid[a], 1])]
        seqp[iM] = sv; v += sv
        sv = cash[iT]
        for a in range(NA):
            if tun[a] > 0.0:
                sv += tun[a] * lastc[tk[a]]
        seqp[iT] = sv; v += sv
        seqp[iC] = cash[iC]; v += cash[iC]
        for s in range(NS):
            seq[t, s] = seqp[s]
        eq[t] = v; eq_prev = v
    # positions still open at the end: logged (marked at the last close) so win rates include them
    for s in range(S):
        for k in range(K):
            if units[s, k] > 0.0:
                pr = units[s, k] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
                nl = _log(lg, nl, ebar[s, k], n - 1, k, 9.0, pr / amt0[s, k] - 1.0, amt0[s, k], eq0[s, k], s)
    for a in range(MA):
        if aid[a] >= 0 and ast[a] == 1:
            k = int(ev[aid[a], 1])
            pr = aun[a] * lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee)
            nl = _log(lg, nl, int(ev[aid[a], 2]) // 60, n - 1, k, 9.0, pr / astake[a] - 1.0, astake[a], aeq0[a], iM)
    for a in range(NA):
        if tun[a] > 0.0:
            k = tk[a]
            ret = lastc[k] * (1.0 - slip[k]) * (1.0 - pens) * (1.0 - fee) * (1.0 - fee) / tent[a] - 1.0
            nl = _log(lg, nl, tbar[a], n - 1, k, 9.0, ret, tamt[a], teq0[a], iT)
    return eq, seq, lg[:nl].copy()


LOGC = ["eb", "xb", "k", "why", "ret", "amt", "eq0", "sl"]
_gm, _tt, _ev = {}, {}, {}
TREND_DEF = dict(assets=("BTCUSDT", "ETHUSDT"), aw=(0.5, 0.5), rule="sma", n=200, unknown_on=False)


def prun(holdout, w, dips=(L.MODE90,), mom=None, trend=None, rb=2, guard=0.35, pen=0.0, pens=0.0, mtp_now=False,
         cd_add=0, ev=None):
    """w = weights [dip sleeves..., MOM, TREND, CASH] (sum 1). Returns (equity, sleeve equity DataFrame, log)."""
    syms, idx, P = market(holdout)
    O = P["o"].values; n, K = O.shape
    trend = dict(TREND_DEF, **(trend or {}))
    if ev is None:
        if w[len(dips)] > 0:
            ekey = (holdout, tuple(sorted((mom or {}).items())))
            if ekey not in _ev:
                _ev[ekey] = np.ascontiguousarray(mom_events(holdout, mom), dtype=np.float64)
            ev = _ev[ekey]
        else:
            ev = np.zeros((0, 8))
    slots = int(dict(MOM_DEF, **(mom or {}))["slots"])
    gkey = (holdout, guard)
    if gkey not in _gm:
        _gm[gkey] = np.ascontiguousarray(L.guard_mask(P, guard) if guard > 0 else np.ones((n, K)), dtype=np.float64)
    gm = _gm[gkey]
    hod = np.asarray(idx.hour, np.int64)
    ms = np.asarray((idx.day == 1) & (idx.hour == 0), np.int64)
    tk = np.array([syms.index(a) for a in trend["assets"]], np.int64)
    taw = np.asarray(trend["aw"], np.float64)
    tkey = (holdout, tuple(trend["assets"]), trend["rule"], trend["n"], trend["unknown_on"])
    if tkey not in _tt:
        _tt[tkey] = np.ascontiguousarray(np.column_stack(
            [trend_target(idx, a, trend["rule"], trend["n"], trend["unknown_on"]) for a in trend["assets"]]))
    tt = _tt[tkey]
    a = lambda key, dt=np.float64: np.array([d[key] for d in dips], dt)  # noqa: E731
    eq, seq, lg = pkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                          gm, hod, ms,
                          a("depth"), a("N", np.int64), a("fmax"), a("tp"), a("hm", np.int64),
                          a("cd", np.int64) + int(cd_add),
                          np.ascontiguousarray(ev, dtype=np.float64), slots, tk, taw, tt,
                          np.asarray(w, np.float64), int(rb), E.EPS, E.FEE, np.full(K, E.SLIP_MAJOR), float(pen),
                          float(pens), 1, bool(mtp_now))
    names = [f"dip{i}" for i in range(len(dips))] + ["mom", "trend", "cash"]
    return pd.Series(eq, idx), pd.DataFrame(seq, idx, names), pd.DataFrame(lg, columns=LOGC)


STRESS = dict(pen=0.002, pens=0.002)


# ------------------------------------------------------------------ metrics
def months(e, a, b):
    """Monthly returns from a to b (first month vs the equity just before a)."""
    i0 = int(np.searchsorted(e.index.values, np.datetime64(a), side="left"))
    i1 = len(e) if b is None else int(np.searchsorted(e.index.values, np.datetime64(b), side="left"))
    start = e.values[i0 - 1] if i0 > 0 else e.values[0]
    seg = pd.Series(e.values[i0:i1] / start, e.index[i0:i1])
    me = seg.resample("ME").last()
    return pd.concat([pd.Series([1.0]), me.reset_index(drop=True)]).pct_change().dropna().set_axis(me.index)


def stats(e, lg, idx, a, b, sleeve=None):
    """%/month (compounded), max drop, worst month, share of positive months, trades and trade win rate
    (trades by entry hour; positions open at the end are marked at the last close)."""
    m, mo, dd, wm, wd = L.period(e, a, b)
    mr = months(e, a, b)
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    sel = (tb >= a) & ((tb < b) if b is not None else True)
    if sleeve is not None:
        sel = sel & np.isin(lg.sl.values, np.atleast_1d(sleeve))
    x = lg[sel]
    return dict(mult=m, pct_mo=mo * 100, dd=dd * 100, worst_mo=wm * 100, pos_mo=float((mr > 0).mean()) * 100,
                n_mo=len(mr), trades=int(len(x)), wr=float((x.ret > 0).mean()) * 100 if len(x) else np.nan,
                avg_tr=float(x.ret.mean()) * 100 if len(x) else np.nan)


def all_periods(w, label="", **kw):
    """One design on the holdout run (2018-20) and the main run (2021-23, 2024-26, since 2025)."""
    rows = []
    for hold in (True, False):
        syms, idx, P = market(hold)
        e, sq, lg = prun(hold, w, **kw)
        for name, a, b in PERIODS:
            if (name == "2018-20") != hold:
                continue
            r = stats(e, lg, idx, a, b)
            r.update(design=label, period=name)
            rows.append(r)
    return rows


def fmt_row(r):
    return (f"{r['period']:10s} up-months {r['pos_mo']:5.1f}% ({r['n_mo']:2d})  trade WR {r['wr']:5.1f}% "
            f"({r['trades']:5d})  {r['pct_mo']:+6.2f}%/mo  maxDD {r['dd']:6.1f}%  worst mo {r['worst_mo']:+6.1f}%")
