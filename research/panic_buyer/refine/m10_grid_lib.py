"""m10 "grid": spot grid trading on the user's 19 halal coins (Binance spot, fully funded, no leverage).

Mechanics (one sleeve per coin, own cash, no sharing; the account = weighted sum of sleeves):
  * Geometric lattice: level j sits at price (1+s)^j. A lot bought at level j sells at level j+tpk (a resting
    limit, one grid step up when tpk=1). One lot per level at most.
  * Buy window: the L levels A-off, A-off-1, ..., A-off-L+1 under the anchor level A. A = the lattice level
    under the price when the grid (re)starts. Lot = sleeve cash / L, fixed until the sleeve is flat again, so every
    resting buy is fully funded (orders at the start of a candle never exceed the cash at the start of it).
  * Re-centering: when the sleeve holds no coin and the close is a step above A (or below the window, or buys are
    paused), A moves to the close's level and the lot is re-sized (profits compound). trail_inv=1 also lets the
    window trail upward while lots are held (older lots keep their sell orders).
  * Pause / exit: gate[t]=False (regime filter, known at the candle's open) -> no buy orders that candle.
    liq=1: also sell everything at that open (market). stop>0: a close below the window bottom x (1-stop) sells
    everything (market, range break) and pauses buys for cd candles. hm>0: a lot held hm candles is sold at the close.
  * Fill rules (code/lab.py grid_kernel convention): the candle path is O-L-H-C (green) or O-H-L-C (red); a limit
    fills only if price trades eps (0.1%) through it; a lot can't sell in the candle it was bought; strict=1: a level
    sold in a candle can't be bought again in that candle either (an hourly bot re-places orders once an hour).
    Buy orders are only placed below the candle's open. Market orders pay slip. pen: every fill pen worse.
  * Every lot is one round-trip in the log, whatever closes it: grid take-profit (why 0), time stop (1), range-break
    stop (2), regime exit (3), per-lot hard stop hs (4, a close hs under the lot's buy level), open at the end of the data marked at the last close (9). Win = net return > 0 after
    both fees, so losing exits and bags still held at the end count against the win rate.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DATA = os.environ.get("PB_DATA", os.path.expanduser("~/pb_data"))
HOLDOUT = os.environ.get("PB_HOLDOUT", os.path.expanduser("~/pb_holdout"))
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/grid"))
os.makedirs(OUT, exist_ok=True)

FEE, EPS, SLIP = 0.001, 0.001, 0.0002
H19 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT",
       "AVAXUSDT", "SUIUSDT", "LTCUSDT", "BCHUSDT", "HBARUSDT", "DOTUSDT", "NEARUSDT", "ARBUSDT", "FETUSDT",
       "RENDERUSDT"]
T_H0, T_D0, T_T0, T_S0 = (pd.Timestamp(x) for x in ("2018-01-01", "2021-01-01", "2024-01-01", "2025-01-01"))
PERIODS = {"h": (T_H0, T_D0), "d": (T_D0, T_T0), "t": (T_T0, None), "s": (T_S0, None)}
LOGC = ["eb", "xb", "ret", "amt", "why", "k"]


# ------------------------------------------------------------------ data
def _read(path):
    d = pd.read_csv(path, index_col=0)
    d.index = pd.to_datetime(d.index, format="mixed")
    d = d[d.index == d.index.floor("1h")]
    return d[~d.index.duplicated(keep="last")].sort_index()[["o", "h", "l", "c"]]


def load_coin(sym):
    """Hourly candles Dec 2017 - Oct 2026: the holdout files before 2021, the main files from 2021 (RENDER = RNDR
    before Jul 2024, BCH = BCHABC Nov 2018 - Nov 2019). The pre-2021 part only feeds the holdout run and the
    warm-up of slow indicators (a 200-day average needs 200 days of history at 1 Jan 2021)."""
    parts = []
    pre = os.path.join(HOLDOUT, "1h_pre", f"{sym}.csv")
    if sym == "BCHUSDT":
        a = os.path.join(HOLDOUT, "1h_pre", "BCHABCUSDT.csv")
        if os.path.exists(a):
            parts.append(_read(a))
    if os.path.exists(pre):
        p = _read(pre)
        parts.append(p[p.index < T_D0])
    for s in (["RNDRUSDT", "RENDERUSDT"] if sym == "RENDERUSDT" else [sym]):
        f = os.path.join(DATA, "1h", f"{s}.csv")
        if os.path.exists(f):
            parts.append(_read(f))
    d = pd.concat(parts)
    return d[~d.index.duplicated(keep="last")].sort_index()


_CACHE = {}


def market(syms=H19):
    """Aligned hourly panel {o,h,l,c} (NaN before listing / missing hours) for syms, Dec 2017 - Oct 2026."""
    key = tuple(syms)
    if key not in _CACHE:
        D = {s: load_coin(s) for s in syms}
        idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq="1h")
        P = {f: pd.DataFrame({s: D[s][f] for s in syms}).reindex(idx) for f in ("o", "h", "l", "c")}
        _CACHE[key] = (idx, P)
    return _CACHE[key]


# ------------------------------------------------------------------ regime gates (known at each candle's open)
def daily_close(P):
    return P["c"].resample("1D").last()


def to_hours(daily_bool, idx):
    """A daily signal computed on day d's close is used during day d+1 (shift 1 day), mapped to the hours."""
    g = daily_bool.shift(1)
    g = g.reindex(idx.floor("D")).values
    return np.where(pd.isna(g), False, g).astype(bool)


def coin_sma(P, idx, n):
    dc = daily_close(P)
    res = {}
    for s in P["c"].columns:
        sm = dc[s].rolling(n, min_periods=20).mean()
        res[s] = to_hours((dc[s] > sm).where(sm.notna()), idx)
    return res


def btc_sma(P, idx, n):
    dc = daily_close(P)["BTCUSDT"]
    sm = dc.rolling(n, min_periods=20).mean()
    return to_hours((dc > sm).where(sm.notna()), idx)


def hourly_ret_gate(P, idx, sym, hours, floor):
    """sym's return over the last `hours` closed hours above `floor` (known at the open)."""
    c = P["c"][sym].ffill()
    r = (c / c.shift(hours) - 1.0).shift(1)
    return (r > floor).fillna(False).values


def er_gate(P, idx, hours, thr):
    """Kaufman efficiency ratio of each coin over `hours` closed hours below thr (a choppy, range-bound market)."""
    c = P["c"].ffill()
    net = (c - c.shift(hours)).abs()
    path = c.diff().abs().rolling(hours).sum()
    er = (net / path).shift(1)
    return {s: (er[s] < thr).fillna(False).values for s in c.columns}


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def gk(o, h, l, c, gate, liq, s, L, off, tpk, trail_inv, stop, cd, hm, eps, fee, slip, pen, strict, t0, t1, hs=0.0):
    """One coin sleeve from bar t0 to t1 (exclusive). Returns equity (1.0 at t0, cash before listing) and the
    round-trip log [buy bar, exit bar, net return, amount, why]."""
    n = len(o)
    ls = np.log(1.0 + s)
    M = 4 * L + 4
    lj = np.zeros(M, np.int64); lu = np.zeros(M); la = np.zeros(M); lb = np.full(M, -1, np.int64)
    act = np.zeros(M, np.bool_)
    oj = np.zeros(L, np.int64); on = 0                     # funded buy orders this candle (level index)
    soldj = np.zeros(M, np.int64); ns = 0
    eq = np.ones(n); cash = 1.0
    lg = np.zeros((200000, 5)); nl = 0
    A = 0; started = False; lot = 0.0; cool = -1; nact = 0; lastc = np.nan
    pts = np.empty(4)
    for t in range(t0, t1):
        if np.isnan(o[t]) or np.isnan(c[t]):
            v = cash
            if nact > 0 and not np.isnan(lastc):
                for m in range(M):
                    if act[m]:
                        v += lu[m] * lastc
            eq[t] = v
            continue
        if not started:
            A = int(np.floor(np.log(o[t]) / ls)); lot = cash / L; started = True
        ns = 0
        # regime exit at the open
        if liq and (not gate[t]) and nact > 0:
            for m in range(M):
                if act[m]:
                    px = o[t] * (1.0 - slip) * (1.0 - pen)
                    pr = lu[m] * px * (1.0 - fee); cash += pr
                    if nl < 200000:
                        lg[nl, 0] = lb[m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 3; nl += 1
                    act[m] = False
            nact = 0
        # take-profits resting from earlier candles that the open already passed
        for m in range(M):
            if act[m] and lb[m] < t:
                tg = np.exp((lj[m] + tpk) * ls)
                if o[t] >= tg:
                    pr = lu[m] * o[t] * (1.0 - pen) * (1.0 - fee); cash += pr
                    if nl < 200000:
                        lg[nl, 0] = lb[m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 0; nl += 1
                    act[m] = False; nact -= 1
                    soldj[ns] = lj[m]; ns += 1
        # place fully funded buy orders: nearest empty levels under the open, while cash lasts
        on = 0
        if gate[t] and t >= cool:
            budget = cash
            for q in range(L):
                j = A - off - q
                pj = np.exp(j * ls)
                if pj >= o[t]:
                    continue
                occ = False
                for m in range(M):
                    if act[m] and lj[m] == j:
                        occ = True
                        break
                if occ:
                    continue
                if strict:
                    for r in range(ns):
                        if soldj[r] == j:
                            occ = True
                            break
                    if occ:
                        continue
                if budget >= lot * 0.999 and lot > 1e-12:
                    oj[on] = j; on += 1; budget -= lot
        # the candle path
        if c[t] >= o[t]:
            pts[0] = o[t]; pts[1] = l[t]; pts[2] = h[t]; pts[3] = c[t]
        else:
            pts[0] = o[t]; pts[1] = h[t]; pts[2] = l[t]; pts[3] = c[t]
        for g in range(3):
            a = pts[g]; b = pts[g + 1]
            if b < a:
                for q in range(on):
                    j = oj[q]
                    if j < -10 ** 8:
                        continue
                    pj = np.exp(j * ls)
                    if a >= pj and b <= pj * (1.0 - eps):
                        ok = True
                        if strict:
                            for r in range(ns):
                                if soldj[r] == j:
                                    ok = False
                                    break
                        if not ok:
                            continue
                        for m in range(M):
                            if not act[m]:
                                amt = lot if cash >= lot else cash
                                px = pj * (1.0 + pen)
                                cash -= amt; lu[m] = amt * (1.0 - fee) / px; la[m] = amt; lb[m] = t; lj[m] = j
                                act[m] = True; nact += 1
                                break
                        oj[q] = -10 ** 9          # one fill per order per candle
            elif b > a:
                for m in range(M):
                    if act[m] and lb[m] < t:
                        tg = np.exp((lj[m] + tpk) * ls)
                        if b >= tg * (1.0 + eps):
                            pr = lu[m] * tg * (1.0 - pen) * (1.0 - fee); cash += pr
                            if nl < 200000:
                                lg[nl, 0] = lb[m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 0; nl += 1
                            act[m] = False; nact -= 1
                            soldj[ns] = lj[m]; ns += 1
        # exits at the close: time stop and range-break stop
        if nact > 0 and hm > 0:
            for m in range(M):
                if act[m] and t - lb[m] >= hm:
                    pr = lu[m] * c[t] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                    if nl < 200000:
                        lg[nl, 0] = lb[m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 1; nl += 1
                    act[m] = False; nact -= 1
        if nact > 0 and hs > 0.0:
            for m in range(M):
                if act[m] and c[t] <= np.exp(lj[m] * ls) * (1.0 - hs):
                    pr = lu[m] * c[t] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                    if nl < 200000:
                        lg[nl, 0] = lb[m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 4; nl += 1
                    act[m] = False; nact -= 1
        if nact > 0 and stop > 0.0:
            bottom = np.exp((A - off - L + 1) * ls)
            if c[t] < bottom * (1.0 - stop):
                for m in range(M):
                    if act[m]:
                        pr = lu[m] * c[t] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                        if nl < 200000:
                            lg[nl, 0] = lb[m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 2; nl += 1
                        act[m] = False
                nact = 0
                cool = t + 1 + cd
        # re-centre
        jc = int(np.floor(np.log(c[t]) / ls))
        nxt_gate = gate[t + 1] if t + 1 < n else gate[t]
        if nact == 0:
            if jc > A or jc < A - off - L + 1 or not nxt_gate or t + 1 < cool:
                A = jc
            lot = cash / L
        elif trail_inv and jc > A:
            A = jc
        lastc = c[t]
        v = cash
        for m in range(M):
            if act[m]:
                v += lu[m] * c[t]
        eq[t] = v
    # open lots at the end: marked at the last close (as if sold at market)
    for m in range(M):
        if act[m] and nl < 200000:
            pr = lu[m] * lastc * (1.0 - slip) * (1.0 - pen) * (1.0 - fee)
            lg[nl, 0] = lb[m]; lg[nl, 1] = t1 - 1; lg[nl, 2] = pr / la[m] - 1.0; lg[nl, 3] = la[m]; lg[nl, 4] = 9; nl += 1
    return eq[t0:t1].copy(), lg[:nl].copy()


# ------------------------------------------------------------------ portfolio
def gates_for(P, idx, spec, cache):
    """spec: 'none' | 'btc{N}' | 'coin{N}' | 'both{N}' (btc and coin above their N-day average) | 'btc{N}+r{H}_{x}'
    (plus BTC's H-hour return above -x%) | 'er{H}_{thr}' (choppiness) | combinations joined with '&'."""
    if spec in cache:
        return cache[spec]
    K = list(P["c"].columns)
    n = len(idx)
    if "&" in spec:
        parts = [gates_for(P, idx, x, cache) for x in spec.split("&")]
        out = {s: np.logical_and.reduce([p[s] for p in parts]) for s in K}
    elif spec == "none":
        out = {s: np.ones(n, bool) for s in K}
    elif spec.startswith("btc"):
        g = btc_sma(P, idx, int(spec[3:]))
        out = {s: g for s in K}
    elif spec.startswith("coin"):
        out = coin_sma(P, idx, int(spec[4:]))
    elif spec.startswith("r"):
        hh, x = spec[1:].split("_")
        g = hourly_ret_gate(P, idx, "BTCUSDT", int(hh), -float(x) / 100)
        out = {s: g for s in K}
    elif spec.startswith("er"):
        hh, thr = spec[2:].split("_")
        out = er_gate(P, idx, int(hh), float(thr))
    else:
        raise ValueError(spec)
    cache[spec] = out
    return out


def run(idx, P, coins, d, t0, t1, gcache, pen=0.0, eps=EPS, fee=FEE, slip=SLIP, weights=None):
    """Equal-weight sleeves (one per coin) from t0 to t1. Returns account equity (pd.Series), log (DataFrame)."""
    i0 = int(np.searchsorted(idx.values, np.datetime64(t0)))
    i1 = len(idx) if t1 is None else int(np.searchsorted(idx.values, np.datetime64(t1)))
    G = gates_for(P, idx, d["gate"], gcache)
    w = np.full(len(coins), 1.0 / len(coins)) if weights is None else np.asarray(weights, float)
    tot = np.zeros(i1 - i0); logs = []
    for k, s in enumerate(coins):
        o, h, l, c = (np.ascontiguousarray(P[f][s].values) for f in ("o", "h", "l", "c"))
        eq, lg = gk(o, h, l, c, G[s], int(d.get("liq", 0)), float(d["s"]), int(d["L"]), int(d.get("off", 0)),
                    int(d.get("tpk", 1)), int(d.get("trail", 0)), float(d.get("stop", 0.0)), int(d.get("cd", 0)),
                    int(d.get("hm", 0)), eps, fee, slip, pen, int(d.get("strict", 1)), i0, i1, float(d.get("hs", 0.0)))
        tot += w[k] * eq
        if len(lg):
            x = pd.DataFrame(lg, columns=LOGC[:5]); x["k"] = k; x["amt"] *= w[k]
            logs.append(x)
    lg = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(columns=LOGC)
    return pd.Series(tot, idx[i0:i1]), lg


# ------------------------------------------------------------------ metrics
def period_stats(e, lg, idx, a, b):
    """%/month (compounded), max drawdown, worst calendar month, round-trips (bought in the period) per month,
    round-trip win rate, average round-trip, share of round-trips closed by a loss exit, multiple."""
    e = e[(e.index >= a) & ((e.index < b) if b is not None else True)]
    if len(e) < 24:
        return None
    i0 = int(np.searchsorted(idx.values, np.datetime64(e.index[0])))
    x = e.values / e.values[0]
    months = ((e.index[-1] + pd.Timedelta(hours=1)) - e.index[0]).total_seconds() / (365.25 / 12 * 86400)
    mult = float(x[-1])
    mo = mult ** (1.0 / months) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    es = pd.Series(x, e.index)
    m = es.resample("ME").last()
    m = pd.concat([pd.Series([1.0], [e.index[0] - pd.Timedelta(seconds=1)]), m]).pct_change().dropna()
    eb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    sel = (eb >= e.index[0]) & (eb <= e.index[-1]) if len(lg) else np.zeros(0, bool)
    r = lg.ret.values[sel] if len(lg) else np.zeros(0)
    why = lg.why.values[sel] if len(lg) else np.zeros(0)
    return dict(mult=mult, mo=mo, dd=dd, worst_m=float(m.min()), pos_m=float((m > 0).mean()),
                n=int(len(r)), rt_m=len(r) / months, wr=float((r > 0).mean()) if len(r) else np.nan,
                avg=float(r.mean()) if len(r) else np.nan, worst_rt=float(r.min()) if len(r) else np.nan,
                n_open=int((why == 9).sum()), n_forced=int(((why >= 1) & (why <= 4)).sum()))


def fmt(tag, r):
    if r is None:
        return f"{tag}: n/a"
    return (f"{tag}: {r['mo']*100:+6.2f}%/mo x{r['mult']:6.2f} DD {r['dd']*100:6.1f}% worst-mo {r['worst_m']*100:+6.1f}% "
            f"up-mo {r['pos_m']*100:3.0f}% | RT win {r['wr']*100:5.1f}% ({r['n']} RT, {r['rt_m']:.0f}/mo, "
            f"avg {r['avg']*100:+.2f}%, worst {r['worst_rt']*100:+.1f}%, forced {r['n_forced']}, open {r['n_open']})")


def dname(d):
    return (f"s{d['s']*100:.2f}% L{d['L']}{' shared Ltot'+str(d['Ltot']) if d.get('Ltot') else ''} off{d.get('off',0)} tpk{d.get('tpk',1)} gate {d['gate']} liq{d.get('liq',0)} "
            f"stop{d.get('stop',0)*100:.0f}% cd{d.get('cd',0)} hm{d.get('hm',0)} hs{d.get('hs',0)*100:.0f}% trail{d.get('trail',0)} "
            f"coins {d.get('cset','?')}")


# ------------------------------------------------------------------ shared-cash kernel (all coins, one cash pool)
@njit(cache=True)
def mk(o, h, l, c, gate, liq, s, L, Ltot, off, tpk, trail_inv, stop, cd, hm, eps, fee, slip, pen, t0, t1, hs=0.0):
    """All coins share one cash pool. Lot = account equity (previous close) / Ltot. Each candle, fully funded buy
    orders go to the empty grid levels (each coin's own L-level window, under the open) nearest to the price
    across ALL coins, until the cash at the candle's start is used up. Everything else as gk (strict fills).
    Returns account equity and the log [buy bar, exit bar, net return, amount, why, coin]."""
    n, K = c.shape
    ls = np.log(1.0 + s)
    M = 2 * L + 2
    lj = np.zeros((K, M), np.int64); lu = np.zeros((K, M)); la = np.zeros((K, M)); lb = np.full((K, M), -1, np.int64)
    act = np.zeros((K, M), np.bool_); nact = np.zeros(K, np.int64)
    A = np.zeros(K, np.int64); started = np.zeros(K, np.bool_); cool = np.full(K, -1, np.int64)
    soldj = np.zeros((K, M), np.int64); ns = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan)
    cand_k = np.zeros(K * L, np.int64); cand_j = np.zeros(K * L, np.int64); cand_d = np.zeros(K * L)
    ok_k = np.zeros(K * L, np.int64); ok_j = np.zeros(K * L, np.int64); non = 0
    eq = np.ones(n); cash = 1.0; eq_prev = 1.0
    lg = np.zeros((400000, 6)); nl = 0
    pts = np.empty(4)
    for t in range(t0, t1):
        for k in range(K):
            ns[k] = 0
            if np.isnan(o[t, k]) or np.isnan(c[t, k]):
                continue
            if not started[k]:
                A[k] = int(np.floor(np.log(o[t, k]) / ls)); started[k] = True
            if liq and (not gate[t, k]) and nact[k] > 0:
                for m in range(M):
                    if act[k, m]:
                        pr = lu[k, m] * o[t, k] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                        if nl < 400000:
                            lg[nl, 0] = lb[k, m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                            lg[nl, 4] = 3; lg[nl, 5] = k; nl += 1
                        act[k, m] = False
                nact[k] = 0
            for m in range(M):
                if act[k, m] and lb[k, m] < t:
                    tg = np.exp((lj[k, m] + tpk) * ls)
                    if o[t, k] >= tg:
                        pr = lu[k, m] * o[t, k] * (1.0 - pen) * (1.0 - fee); cash += pr
                        if nl < 400000:
                            lg[nl, 0] = lb[k, m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                            lg[nl, 4] = 0; lg[nl, 5] = k; nl += 1
                        act[k, m] = False; nact[k] -= 1
                        soldj[k, ns[k]] = lj[k, m]; ns[k] += 1
        # candidate orders across coins, nearest first, funded from the cash at the candle's start
        nc = 0
        for k in range(K):
            if np.isnan(o[t, k]) or np.isnan(c[t, k]) or (not gate[t, k]) or t < cool[k]:
                continue
            for q in range(L):
                j = A[k] - off - q
                pj = np.exp(j * ls)
                if pj >= o[t, k]:
                    continue
                occ = False
                for m in range(M):
                    if act[k, m] and lj[k, m] == j:
                        occ = True
                        break
                for r in range(ns[k]):
                    if soldj[k, r] == j:
                        occ = True
                        break
                if occ:
                    continue
                cand_k[nc] = k; cand_j[nc] = j; cand_d[nc] = o[t, k] / pj - 1.0; nc += 1
        lot = eq_prev / Ltot
        non = 0
        if nc > 0 and lot > 1e-12:
            order = np.argsort(cand_d[:nc])
            budget = cash
            for q in range(nc):
                if budget < lot * 0.999:
                    break
                i = order[q]
                ok_k[non] = cand_k[i]; ok_j[non] = cand_j[i]; non += 1; budget -= lot
        for k in range(K):
            if np.isnan(o[t, k]) or np.isnan(c[t, k]):
                continue
            if c[t, k] >= o[t, k]:
                pts[0] = o[t, k]; pts[1] = l[t, k]; pts[2] = h[t, k]; pts[3] = c[t, k]
            else:
                pts[0] = o[t, k]; pts[1] = h[t, k]; pts[2] = l[t, k]; pts[3] = c[t, k]
            for g in range(3):
                a = pts[g]; b = pts[g + 1]
                if b < a:
                    for q in range(non):
                        if ok_k[q] != k:
                            continue
                        j = ok_j[q]
                        pj = np.exp(j * ls)
                        if a >= pj and b <= pj * (1.0 - eps):
                            for m in range(M):
                                if not act[k, m]:
                                    amt = lot if cash >= lot else cash
                                    if amt > 1e-12:
                                        cash -= amt; lu[k, m] = amt * (1.0 - fee) / (pj * (1.0 + pen)); la[k, m] = amt
                                        lb[k, m] = t; lj[k, m] = j; act[k, m] = True; nact[k] += 1
                                    break
                            ok_k[q] = -1
                elif b > a:
                    for m in range(M):
                        if act[k, m] and lb[k, m] < t:
                            tg = np.exp((lj[k, m] + tpk) * ls)
                            if b >= tg * (1.0 + eps):
                                pr = lu[k, m] * tg * (1.0 - pen) * (1.0 - fee); cash += pr
                                if nl < 400000:
                                    lg[nl, 0] = lb[k, m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                                    lg[nl, 4] = 0; lg[nl, 5] = k; nl += 1
                                act[k, m] = False; nact[k] -= 1
                                soldj[k, ns[k]] = lj[k, m]; ns[k] += 1
            if nact[k] > 0 and hm > 0:
                for m in range(M):
                    if act[k, m] and t - lb[k, m] >= hm:
                        pr = lu[k, m] * c[t, k] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                        if nl < 400000:
                            lg[nl, 0] = lb[k, m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                            lg[nl, 4] = 1; lg[nl, 5] = k; nl += 1
                        act[k, m] = False; nact[k] -= 1
            if nact[k] > 0 and hs > 0.0:
                for m in range(M):
                    if act[k, m] and c[t, k] <= np.exp(lj[k, m] * ls) * (1.0 - hs):
                        pr = lu[k, m] * c[t, k] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                        if nl < 400000:
                            lg[nl, 0] = lb[k, m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                            lg[nl, 4] = 4; lg[nl, 5] = k; nl += 1
                        act[k, m] = False; nact[k] -= 1
            if nact[k] > 0 and stop > 0.0:
                bottom = np.exp((A[k] - off - L + 1) * ls)
                if c[t, k] < bottom * (1.0 - stop):
                    for m in range(M):
                        if act[k, m]:
                            pr = lu[k, m] * c[t, k] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee); cash += pr
                            if nl < 400000:
                                lg[nl, 0] = lb[k, m]; lg[nl, 1] = t; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                                lg[nl, 4] = 2; lg[nl, 5] = k; nl += 1
                            act[k, m] = False
                    nact[k] = 0
                    cool[k] = t + 1 + cd
            jc = int(np.floor(np.log(c[t, k]) / ls))
            nxt = gate[t + 1, k] if t + 1 < n else gate[t, k]
            if nact[k] == 0:
                if jc > A[k] or jc < A[k] - off - L + 1 or (not nxt) or t + 1 < cool[k]:
                    A[k] = jc
            elif trail_inv and jc > A[k]:
                A[k] = jc
            lastc[k] = c[t, k]
        v = cash
        for k in range(K):
            if nact[k] > 0:
                for m in range(M):
                    if act[k, m]:
                        v += lu[k, m] * lastc[k]
        eq[t] = v; eq_prev = v
    for k in range(K):
        for m in range(M):
            if act[k, m] and nl < 400000:
                pr = lu[k, m] * lastc[k] * (1.0 - slip) * (1.0 - pen) * (1.0 - fee)
                lg[nl, 0] = lb[k, m]; lg[nl, 1] = t1 - 1; lg[nl, 2] = pr / la[k, m] - 1.0; lg[nl, 3] = la[k, m]
                lg[nl, 4] = 9; lg[nl, 5] = k; nl += 1
    return eq[t0:t1].copy(), lg[:nl].copy()


def run_shared(idx, P, coins, d, t0, t1, gcache, pen=0.0, eps=EPS, fee=FEE, slip=SLIP):
    i0 = int(np.searchsorted(idx.values, np.datetime64(t0)))
    i1 = len(idx) if t1 is None else int(np.searchsorted(idx.values, np.datetime64(t1)))
    Gd = gates_for(P, idx, d["gate"], gcache)
    gate = np.ascontiguousarray(np.stack([Gd[s] for s in coins], axis=1))
    o, h, l, c = (np.ascontiguousarray(P[f][coins].values) for f in ("o", "h", "l", "c"))
    eq, lg = mk(o, h, l, c, gate, int(d.get("liq", 0)), float(d["s"]), int(d["L"]), float(d["Ltot"]),
                int(d.get("off", 0)), int(d.get("tpk", 1)), int(d.get("trail", 0)), float(d.get("stop", 0.0)),
                int(d.get("cd", 0)), int(d.get("hm", 0)), eps, fee, slip, pen, i0, i1, float(d.get("hs", 0.0)))
    return pd.Series(eq, idx[i0:i1]), pd.DataFrame(lg, columns=LOGC)


def run_any(idx, P, coins, d, t0, t1, gcache, **kw):
    return (run_shared if d.get("Ltot") else run)(idx, P, coins, d, t0, t1, gcache, **kw)
