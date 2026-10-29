"""x10 lever "universe": more dips to catch.

The bot trades the screened halal coins (code/universe.py pass_map, data in $PB_DATA/univ, delisted coins
included: when a coin's data ends, an open position is stuck at its last price for the rest of the run).
Each month, at the first hour, coins are ranked by trailing 30-day quote volume using only data before that
hour (min 30 days of history, must have traded in the previous hour); the bot may buy only the top M that month.
Coins already held always exit normally.

Costs: 0.1% fee per side. Market buys at the open and market sells (time stop) pay 0.02% slippage when the coin
is in the top 50 by trailing 30-day volume that month (ranked among all screened coins), else 0.1%. Limit buys
fill only if the low trades 0.1% through the limit.

ukernel = engine.gkernel copied, plus three switches (all off = engine.gkernel exactly, see x10_universe_check.py):
  * slip is an (n, K) matrix (time-varying slippage by monthly volume rank) instead of a (K,) vector;
  * watch=True: a stream-watching bot instead of resting bids. Each hour it looks at every eligible coin in the
    same order the engine uses (closest to its trigger at the open first). A coin already under its trigger at
    the open is bought at market at the open (as in the engine); a coin whose low trades 0.1% through its
    trigger is bought with a market order when the price crosses the trigger: fill = trigger x (1 + slippage +
    wslip). Each buy is min(fmax x equity, cash) - funded by cash actually on hand at that moment, so total
    orders never exceed cash. nord = maximum number of open positions. Within an hour the true order in which
    coins cross is unknown on hourly data; `worder` picks the proxy (0: closest at the open first = engine
    order; 1: adversarial - the coin with the worst close relative to its trigger first, a pessimistic check
    that uses the hour's close only to make the result worse).
  * the trade log also records the equity at the entry hour (to express each trade as % of the account) and,
    after the last hour, every still-open position marked at its last price (xb = -1), e.g. a coin stuck after
    delisting.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import engine as E  # noqa: E402
import universe as U  # noqa: E402

UNIV = os.path.join(E.DATA, "univ")
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_x10_universe"))
os.makedirs(OUT, exist_ok=True)
LUNA2_START = "2022-05-28 00:00"          # LUNA 2.0 (new chain) relisted under the old ticker: separate coin
ULOG = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]

T_END = None                               # last hour of data (set on load)
PER = {"24-26": ("2024-01-01", None), "25-26": ("2025-01-01", None), "21-23": ("2021-01-01", "2024-01-01")}


# ------------------------------------------------------------------ data
def _read(p, cols):
    d = pd.read_csv(p, index_col=0, parse_dates=True, usecols=["t"] + cols)
    return d[d.index >= "2021-01-01"]


def load_univ(stress=False, cols=("o", "h", "l", "c", "qv"), only=None):
    """All screened coins from 2021 with 30+ days of data. RNDR is joined to RENDER (as idea5); LUNA classic is
    cut at its collapse (position stuck at the last price, effectively -100%), LUNA 2.0 is a separate coin.
    stress=True adds FTT (not on the screened list: exchange token) with its halt handling from engine.DEAD.
    only: restrict to these symbols."""
    pm = U.pass_map()
    cols = list(cols)
    D = {}
    for p in sorted(glob.glob(os.path.join(UNIV, "*.csv"))):
        sym = os.path.basename(p)[:-4]
        if sym[:-4] not in pm or sym == "RNDRUSDT":
            continue
        if only is not None and sym not in only and not (sym == "LUNAUSDT" and "LUNA2USDT" in only):
            continue
        d = _read(p, cols)
        if sym == "LUNAUSDT":
            D["LUNA2USDT"] = d[d.index >= LUNA2_START]
            d = d[d.index <= E.DEAD["LUNAUSDT"]]
        if len(d) >= 24 * 30:
            D[sym] = d
    if "RENDERUSDT" in D:
        r = _read(os.path.join(UNIV, "RNDRUSDT.csv"), cols)
        x = pd.concat([r, D["RENDERUSDT"]])
        D["RENDERUSDT"] = x[~x.index.duplicated(keep="last")].sort_index()
    if stress:
        f = E.read_hourly("FTTUSDT")[cols]
        D["FTTUSDT"] = f[(f.index >= "2021-01-01") & (f.index <= E.DEAD["FTTUSDT"])]
    if only is not None:
        D = {s: d for s, d in D.items() if s in only}
    return D


def month_rank(P, look_h=720, min_hist_h=720):
    """(n, K) rank (1 = most volume) by trailing quote volume, fixed at each month's first hour from data before
    it; coins without min_hist_h hours of history, without volume, or not trading in the previous hour get NaN."""
    qv = P["qv"].fillna(0.0)
    vol = qv.rolling(look_h, min_periods=1).sum().shift(1)
    age = P["c"].notna().cumsum().shift(1)
    alive = P["c"].notna().shift(1, fill_value=False)
    idx = P["c"].index
    first = ~idx.to_period("M").duplicated()
    R = np.full(P["c"].shape, np.nan)
    cur = np.full(P["c"].shape[1], np.nan)
    for t in range(len(idx)):
        if first[t]:
            v = vol.iloc[t].where((age.iloc[t] >= min_hist_h) & alive.iloc[t] & (vol.iloc[t] > 0))
            cur = v.rank(ascending=False, method="first").values
        R[t] = cur
    return R


def build(stress=False, keep_top=50, cache=True, extra=()):
    """Hourly panel of the screened coins that were ever in the monthly top `keep_top` (plus the 19 live coins and,
    for stress, LUNA/FTT/SRM). Returns dict with syms, idx, P, rank (n,K), slip (n,K), is19 (K,).
    Ranks are computed over ALL screened coins first (close + volume only), then full candles are loaded for the
    kept coins (keeps memory low)."""
    global T_END
    tag = f"_x{len(extra)}" if extra else ""
    cp = os.path.join(OUT, f"panel_{'stress' if stress else 'base'}_{keep_top}{tag}.pkl")
    if cache and os.path.exists(cp):
        U_ = pd.read_pickle(cp)
        T_END = U_["idx"][-1]
        return U_
    Dq = load_univ(stress, cols=("c", "qv"))
    s_all = list(Dq)
    idx = pd.date_range(min(d.index[0] for d in Dq.values()), max(d.index[-1] for d in Dq.values()), freq="1h")
    Pq = {f: pd.DataFrame({s: Dq[s][f] for s in s_all}).reindex(idx) for f in ("c", "qv")}
    del Dq
    rank_all = month_rank(Pq)
    srank_all = month_rank(Pq, min_hist_h=0)            # slippage tier: no history requirement
    del Pq
    always = set(E.H19) | ({"LUNAUSDT", "FTTUSDT", "SRMUSDT"} if stress else set())
    ever = np.nanmin(np.where(np.isnan(rank_all), 1e9, rank_all), axis=0) <= keep_top
    keep = [s for j, s in enumerate(s_all) if ever[j] or s in always or s in extra]
    D = load_univ(stress, only=set(keep))
    syms, idx2, P = E.panel(D)
    del D
    assert idx2.equals(idx), "panel index changed after pruning"
    col = [s_all.index(s) for s in syms]
    rank, srank = rank_all[:, col], srank_all[:, col]
    slip = np.where(srank <= 50, E.SLIP_MAJOR, E.SLIP_OTHER)
    T_END = idx[-1]
    U_ = dict(syms=syms, idx=idx, P=P, rank=rank, slip=np.ascontiguousarray(slip),
              is19=np.array([s in E.H19 for s in syms]), stress=stress, n_screened=len(s_all))
    if cache:
        pd.to_pickle(U_, cp)
    return U_


def mask(U_, kind, M=30, force=()):
    """smult (n,K): kind 'top' = monthly top-M by 30d volume; '19' = the 19 live coins only; '19+top' = the 19 live
    coins plus the monthly top M; force = coins always eligible (stress test: LUNA/FTT/SRM)."""
    rank, syms = U_["rank"], U_["syms"]
    if kind == "19":
        m = np.repeat(U_["is19"][None, :], len(U_["idx"]), axis=0).astype(float)
    elif kind == "top":
        m = (rank <= M).astype(float)
    elif kind == "19+top":
        m = ((rank <= M) | U_["is19"][None, :]).astype(float)
    else:
        raise ValueError(kind)
    for s in force:
        if s in syms:
            m[:, syms.index(s)] = 1.0
    return np.ascontiguousarray(m)


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def ukernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait=1, tprest=True, mtp_now=False, watch=False, wslip=0.0, worder=0):
    """engine.gkernel with slip (n,K), the watch switch and an extended log (see module docstring)."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    eq0 = np.zeros(K)
    lastc = np.full(K, np.nan); dist = np.empty(K); key = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(n):
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]) or smult[t, k] <= 0.0:
                continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0
        if not watch:
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
                    fill = -1.0; ismkt = 0
                    if o[t, k] <= lim:
                        fill = o[t, k] * (1.0 + slip[t, k]) * (1.0 + pen); ismkt = 1
                    elif l[t, k] <= lim * (1.0 - eps):
                        fill = lim * (1.0 + pen)
                    if fill > 0.0 and cash > 1e-9:
                        mkt[k] = ismkt if mtp_now else 0
                        a = min(per * smult[t, k], cash)
                        cash -= a; units[k] = a * (1.0 - fee) / fill
                        entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0; eq0[k] = eq_prev
                        tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
        else:
            npos = 0
            for k in range(K):
                if units[k] > 0.0:
                    npos += 1
            for k in range(K):
                key[k] = dist[k]
                if worder == 1 and dist[k] < 1e8:          # adversarial: worst close vs trigger first
                    key[k] = c[t, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0 if not np.isnan(c[t, k]) else dist[k]
            order = np.argsort(key)
            for q in range(K):
                k = order[q]
                if dist[k] >= 1e8 or npos >= nord[t] or cash <= 1e-9:
                    continue
                lim = ref[t, k] * (1.0 - depth[t, k])
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[t, k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + slip[t, k] + wslip) * (1.0 + pen)
                if fill > 0.0:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(fmax[t] * eq_prev * smult[t, k], cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0; eq0[k] = eq_prev
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
                    npos += 1
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if ebar[k] == t:
                        if h[t, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    elif t - ebar[k] < tpwait:
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
                            px = c[t, k] * (1.0 - slip[t, k]); why = 2.0
                if px < 0.0 and t - ebar[k] >= hm[t] - mkt[k]:
                    px = c[t, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = eq0[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    for k in range(K):                       # still open at the end (incl. stuck after delisting): mark to market
        if units[k] > 0.0 and nl < maxlog:
            lg[nl, 0] = ebar[k]; lg[nl, 1] = -1.0; lg[nl, 2] = k; lg[nl, 3] = 9.0
            lg[nl, 4] = (got[k] + units[k] * lastc[k]) / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
            lg[nl, 7] = eq0[k]; nl += 1
    return eq, lg[:nl].copy()


def urun(P, idx, slip, depth=0.12, N=8, fmax=0.2, smult=1.0, tp=0.015, part=1.0, tp2=0.0, trail=0.0, hm=48, cd=24,
         eps=E.EPS, fee=E.FEE, pen=0.0, tpwait=1, tprest=True, mtp_now=False, watch=False, wslip=0.0, worder=0):
    O = P["o"].values; n, K = O.shape
    sl = np.asarray(slip, np.float64)
    if sl.ndim == 0:
        sl = np.full((n, K), float(sl))
    elif sl.ndim == 1:
        sl = np.repeat(sl[None, :], n, axis=0)
    eq, lg = ukernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), float(part), float(tp2), float(trail), E.full(n, 0, hm, np.int64), int(cd),
                     eps, fee, np.ascontiguousarray(sl), pen, int(tpwait), bool(tprest), bool(mtp_now), bool(watch),
                     float(wslip), int(worder))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=ULOG)


# ------------------------------------------------------------------ reporting
def per_month(e, start, end=None):
    """(%/month, worst drop, multiple) over [start, end): rate = multiple ** (1 / months) - 1, months from the
    first to the last hour of the period (365.25/12 days each)."""
    x = e[e.index >= start]
    if end is not None:
        x = x[x.index < end]
    x = x / x.iloc[0]
    months = (x.index[-1] - x.index[0]).total_seconds() / (365.25 / 12 * 86400)
    m = float(x.iloc[-1])
    return (m ** (1 / months) - 1 if m > 0 else -1.0), float((x / x.cummax() - 1).min()), m


def year_ret(e, y):
    x = e[(e.index >= f"{y}-01-01") & (e.index < f"{y + 1}-01-01")]
    prev = e[e.index < f"{y}-01-01"]
    base = prev.iloc[-1] if len(prev) else x.iloc[0]
    return float(x.iloc[-1] / base - 1)


def trade_stats(lg, idx, is19, start, end=None):
    """Trades entered in [start, end): count, win rate, worst trade as % of the account, share of the summed
    account-level P&L (ret x amount / equity at entry) that came from coins outside the 19, stuck positions."""
    if len(lg) == 0:
        return dict(n=0)
    ts = idx[lg.eb.astype(int).values]
    sel = (ts >= pd.Timestamp(start)) & ((ts < pd.Timestamp(end)) if end else True)
    t = lg[sel]
    if len(t) == 0:
        return dict(n=0)
    acct = t.ret * t.amt / t.eq0
    out19 = ~is19[t.k.astype(int).values]
    pnl_all = float(acct.sum())
    return dict(n=int(len(t)), win=float((t.ret > 0).mean()), worst_acct=float(acct.min()),
                worst_ret=float(t.ret.min()), pnl_sum=pnl_all, pnl_out19=float(acct[out19].sum()),
                share_trades_out19=float(out19.mean()), stuck=int((t.xb < 0).sum()))


def report(e, lg, idx, is19):
    r = {}
    for name, (a, b) in PER.items():
        mo, dd, mult = per_month(e, a, b)
        r[f"{name}_mo"], r[f"{name}_dd"], r[f"{name}_x"] = mo, dd, mult
        ts = trade_stats(lg, idx, is19, a, b)
        for k_, v in ts.items():
            r[f"{name}_{k_}"] = v
    for y in (2021, 2022, 2023, 2024, 2025, 2026):
        r[f"y{y}"] = year_ret(e, y)
    return r


def fmt(r, pre=""):
    return " | ".join(f"{p} {r[pre + p + '_mo'] * 100:+5.2f}%/mo DD {r[pre + p + '_dd'] * 100:4.0f}%"
                      for p in ("24-26", "25-26", "21-23"))


# ------------------------------------------------------------------ quality-filtered universe
def standalone_scores(U_, depth, tp, hm, cd, window_h=180 * 24):
    """(n, K) per-coin 'dip record' fixed at each month's first hour: the sum of the returns of the coin's own
    standalone dip trades (same depth / take-profit / time stop / cooldown, one coin, all-in, normal costs) whose
    EXIT happened in the window_h hours before that hour. NaN = no such trade. Uses only past prices."""
    P, idx = U_["P"], U_["idx"]
    n, K = P["c"].shape
    first = np.where(~idx.to_period("M").duplicated())[0]
    S_ = np.full((n, K), np.nan)
    O, H, L, C, R = (np.ascontiguousarray(P[f].values) for f in ("o", "h", "l", "cf", "ref"))
    for k in range(K):
        sl = np.ascontiguousarray(U_["slip"][:, k:k + 1])
        _, lg = ukernel(np.ascontiguousarray(O[:, k:k + 1]), np.ascontiguousarray(H[:, k:k + 1]),
                        np.ascontiguousarray(L[:, k:k + 1]), np.ascontiguousarray(C[:, k:k + 1]),
                        np.ascontiguousarray(R[:, k:k + 1]), np.full((n, 1), depth), np.ones(n, np.int64),
                        np.ones(n), np.ones((n, 1)), np.full((n, 1), tp), 1.0, 0.0, 0.0, np.full(n, hm, np.int64),
                        int(cd), E.EPS, E.FEE, sl, 0.0, 1, True, False, False, 0.0, 0)
        lg = lg[lg[:, 1] >= 0]
        if len(lg) == 0:
            continue
        xb, ret = lg[:, 1], lg[:, 4]
        cur = np.nan
        for j, t0 in enumerate(first):
            sel = (xb < t0) & (xb >= t0 - window_h)
            cur = ret[sel].sum() if sel.any() else np.nan
            t1 = first[j + 1] if j + 1 < len(first) else n
            S_[t0:t1, k] = cur
    return S_


def qmask(U_, M, scores, plus19=False, force=()):
    """Monthly top-M by volume, keeping only coins whose trailing standalone dip record is >= 0 (no trades: kept).
    plus19: the 19 live coins are always eligible (not filtered)."""
    m = ((U_["rank"] <= M) & ~(scores < 0)).astype(float)
    if plus19:
        m[:, U_["is19"]] = 1.0
    for s in force:
        if s in U_["syms"]:
            m[:, U_["syms"].index(s)] = 1.0
    return np.ascontiguousarray(m)


def sub(U_, m):
    """Columns that are ever eligible under smult m (others can never be bought, so dropping them changes nothing)."""
    cols = np.where((m > 0).any(axis=0))[0]
    P = {f: v.iloc[:, cols] for f, v in U_["P"].items()}
    return P, np.ascontiguousarray(U_["slip"][:, cols]), np.ascontiguousarray(m[:, cols]), U_["is19"][cols], cols
