"""Task "ten-wider" (w90 round): more dips for 10% mode by adding screened coins outside the user's 19.

Universe (point in time, no lookahead, delisted coins included):
  core  = the user's 19 coins, always eligible, full size, core depth (10% mode: 14%).
  tier  = every OTHER screened coin (code/universe.py pass_map, data in $PB_DATA/univ) that is in the monthly top M
          of the non-19 coins by trailing quote volume. The ranking is fixed at each month's first hour from data
          before it (min 30 days of history, traded in the previous hour). Two rankings: 30-day volume ("v30") and
          365-day volume ("v365", as much history as exists; favours established coins). Ranks are computed over ALL
          322 screened coins, so coins outside any pre-built panel are never missed.
  Tier coins get their own depth (deeper), size multiplier and take-profit.
  Crash guard on every coin: skip a coin whose last close is more than g under its highest close of the previous
  168 hours (the live bot's crash_guard_pct).

Mechanism: the hourly resting-bid bot (engine.gkernel), NOT watch mode: each hour, fully funded limit bids on the
N eligible coins closest to their trigger (open / trigger - 1); size min(fmax x equity, cash / nbids) x size
multiplier; a coin already under its trigger at the open is bought at market.

wkernel = x10_universe_lib.ukernel with watch mode removed and two switches added (both off = ukernel exactly, and on
the 19 coins with a constant slippage = engine.gkernel exactly; w90_wider_check.py proves both):
  pens  every SELL fill pen-worse (take-profit and time-stop); `pen` already makes every BUY fill worse.
  liq   1 = a position still open at its coin's last data hour (delisting, LUNA/FTT/SRM cuts, ticker migrations) is
        sold at that hour's close with market slippage (why = 4) instead of staying frozen at its last price.
Slippage (n, K): 0.02% for coins in the monthly top 50 of all screened coins by 30-day volume, else 0.1%
(x10_universe_lib convention). FTT (stress only, not screened) always 0.1%.
Costs: 0.1% fee per side; limit buys fill only when the low trades 0.1% through.

Stressed version (this round's rule): real bot timing (take-profit placed at once after a market buy: mtp_now=True;
cooldown counted from the next hourly check: cd+1), every buy AND sell fill 0.2% worse, LUNA/FTT/SRM forced
eligible as core coins (full size, core depth), positions sold at a coin's last data hour.
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
import x10_universe_lib as X  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_w90_wider"))
os.makedirs(OUT, exist_ok=True)
T21, T24, T25 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
WLOG = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]       # why: 0 tp, 1 time stop, 4 last data hour, 9 open
STRESS_COINS = ("LUNAUSDT", "FTTUSDT", "SRMUSDT")
TENMODE = dict(dc=0.14, tp=0.05, hm=48, cd=24, N=1, fmax=1.0, g=0.35)


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def wkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, hm, cd, eps, fee, slip, pen, pens, tpwait, tprest,
            mtp_now, lastbar, liq):
    """x10_universe_lib.ukernel, bid mode, full take-profit only (part=1), plus pens and liq (see module doc)."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    eq0 = np.zeros(K)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 8)); nl = 0
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
                    tpp[k] = fill * (1.0 + tpa[t, k])
        for k in range(K):
            if units[k] > 0.0 and (ebar[k] < t or mkt[k] == 1) and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if ebar[k] == t:
                    if h[t, k] >= tpp[k] * (1.0 + eps):
                        px = tpp[k]
                elif t - ebar[k] < tpwait:
                    pass
                elif o[t, k] >= tpp[k]:
                    px = o[t, k]
                elif tprest and h[t, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]
                if px < 0.0 and t - ebar[k] >= hm[t] - mkt[k]:
                    px = c[t, k] * (1.0 - slip[t, k]); why = 1.0
                if px < 0.0 and liq == 1 and t == lastbar[k]:      # coin's last data hour: sell at its close
                    px = c[t, k] * (1.0 - slip[t, k]); why = 4.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - pens) * (1.0 - fee); cash += proceeds; got[k] += proceeds
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


def last_bars(P):
    """Last hour with data per coin; coins that trade to the panel's end get n+1 (never liquidated)."""
    O = P["o"].values
    n, K = O.shape
    ok = ~np.isnan(O)
    lb = np.array([np.nonzero(ok[:, k])[0][-1] if ok[:, k].any() else -1 for k in range(K)], np.int64)
    lb[lb >= n - 3] = n + 1
    return lb


def wrun(P, idx, depth=0.14, N=1, fmax=1.0, smult=1.0, tp=0.05, hm=48, cd=24, eps=E.EPS, fee=E.FEE, slip=None,
         pen=0.0, pens=0.0, tpwait=1, tprest=True, mtp_now=False, liq=1):
    """engine.run's signature (part/tp2/trail not supported) plus pens and liq. slip: None = 0.02% for the 19 coins,
    0.1% for any other coin; a scalar, a (K,) vector or an (n, K) matrix."""
    O = P["o"].values; n, K = O.shape
    if slip is None:
        slip = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns])
    sl = np.asarray(slip, np.float64)
    if sl.ndim == 0:
        sl = np.full((n, K), float(sl))
    elif sl.ndim == 1:
        sl = np.repeat(sl[None, :], n, axis=0)
    eq, lg = wkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     E.full(n, K, depth), E.full(n, 0, N, np.int64), E.full(n, 0, fmax), E.full(n, K, smult),
                     E.full(n, K, tp), E.full(n, 0, hm, np.int64), int(cd), eps, fee, np.ascontiguousarray(sl),
                     float(pen), float(pens), int(tpwait), bool(tprest), bool(mtp_now), last_bars(P), int(liq))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=WLOG)


# ------------------------------------------------------------------ ranks over ALL screened coins
def month_rank_pool(c, qv, pool, look_h, min_hist_h):
    """(n, K) monthly rank (1 = most volume) among the columns in `pool` (bool K) by trailing quote volume over
    look_h hours, fixed at each month's first hour from data before it. Same rule as x10_universe_lib.month_rank
    (which ranks every column): min_hist_h hours of history, traded in the previous hour, volume > 0."""
    qv = qv.fillna(0.0)
    vol = qv.rolling(look_h, min_periods=1).sum().shift(1)
    age = c.notna().cumsum().shift(1)
    alive = c.notna().shift(1, fill_value=False)
    idx = c.index
    first = np.where(~idx.to_period("M").duplicated())[0]
    R = np.full(c.shape, np.nan)
    for j, t in enumerate(first):
        v = vol.iloc[t].where((age.iloc[t] >= min_hist_h) & alive.iloc[t] & (vol.iloc[t] > 0) & pool)
        cur = v.rank(ascending=False, method="first").values
        t1 = first[j + 1] if j + 1 < len(first) else len(idx)
        R[t:t1] = cur
    return R


def ranks_all():
    """Ranks over all screened coins (cached): r30o / r365o = rank among the NON-19 coins by 30 / 365-day volume
    (min 30 days of history); srank = rank among all screened coins by 30-day volume, no history rule (slippage
    tier, as x10_universe_lib.build)."""
    cp = os.path.join(OUT, "ranks_all.pkl")
    if os.path.exists(cp):
        return pd.read_pickle(cp)
    Dq = X.load_univ(False, cols=("c", "qv"))
    s_all = list(Dq)
    idx = pd.date_range(min(d.index[0] for d in Dq.values()), max(d.index[-1] for d in Dq.values()), freq="1h")
    c = pd.DataFrame({s: Dq[s]["c"] for s in s_all}).reindex(idx)
    qv = pd.DataFrame({s: Dq[s]["qv"] for s in s_all}).reindex(idx)
    del Dq
    oth = pd.Series([s not in E.H19 for s in s_all], index=s_all)
    allp = pd.Series(True, index=s_all)
    out = dict(idx=idx, syms=s_all,
               r30o=month_rank_pool(c, qv, oth, 720, 720).astype(np.float32),
               r365o=month_rank_pool(c, qv, oth, 365 * 24, 720).astype(np.float32),
               srank=month_rank_pool(c, qv, allp, 720, 0).astype(np.float32))
    pd.to_pickle(out, cp)
    return out


# ------------------------------------------------------------------ panel
_W = {}


def build(maxM=50):
    """Hourly panel: the 19 coins, every other screened coin ever in the monthly top maxM of the non-19 coins by
    30- or 365-day volume, LUNA/SRM, and FTT (stress only). Returns dict(syms, idx, P, r30o, r365o, slip, is19)."""
    if maxM in _W:
        return _W[maxM]
    cp = os.path.join(OUT, f"panel_{maxM}.pkl")
    if os.path.exists(cp):
        W = pd.read_pickle(cp)
        _W[maxM] = W
        return W
    R = ranks_all()
    s_all = R["syms"]
    ever = (np.nan_to_num(R["r30o"], nan=1e9).min(0) <= maxM) | (np.nan_to_num(R["r365o"], nan=1e9).min(0) <= maxM)
    keep = set(E.H19) | {s for j, s in enumerate(s_all) if ever[j]} | {"LUNAUSDT", "SRMUSDT"}
    D = X.load_univ(True, only=keep | {"FTTUSDT"})
    syms, idx, P = E.panel(D)
    del D
    assert idx.equals(R["idx"]), "panel index differs from the rank index"
    P.pop("qv", None)
    col = [s_all.index(s) if s in s_all else -1 for s in syms]

    def pick(A, fill):
        out = np.full((len(idx), len(syms)), fill, np.float64)
        for j, cj in enumerate(col):
            if cj >= 0:
                out[:, j] = A[:, cj]
        return out
    srank = pick(R["srank"], np.nan)
    slip = np.where(srank <= 50, E.SLIP_MAJOR, E.SLIP_OTHER)
    W = dict(syms=syms, idx=idx, P=P, r30o=pick(R["r30o"], np.nan), r365o=pick(R["r365o"], np.nan),
             slip=np.ascontiguousarray(slip), is19=np.array([s in E.H19 for s in syms]))
    pd.to_pickle(W, cp)
    _W[maxM] = W
    return W


_gm = {}


def guard_mask(P, g, L=168):
    """1.0 = allowed: last close no more than g under the highest close of the previous L hours (known at the open).
    Same rule as panic_buyer.py crash_guard_pct and x10_universe_push2.guard_mask."""
    if g <= 0:
        return np.ones(P["o"].shape)
    key = (id(P["cf"]), P["cf"].shape, g, L)
    if key not in _gm:
        cf = P["cf"]
        hi = cf.rolling(L, min_periods=1).max()
        _gm[key] = ((cf / hi - 1) >= -g).shift(1, fill_value=True).values.astype(float)
    return _gm[key]


def pump_mask(P, ratio, L=720):
    """1.0 = allowed: last close at most `ratio` x the lowest close of the previous L hours (known at the open).
    ratio <= 0: no filter. Keeps the bot out of dips of coins that have just multiplied (e.g. LOOM Oct 2023)."""
    if ratio <= 0:
        return np.ones(P["o"].shape)
    key = (id(P["cf"]), P["cf"].shape, "pump", ratio, L)
    if key not in _gm:
        cf = P["cf"]
        lo = cf.rolling(L, min_periods=1).min()
        _gm[key] = ((cf / lo) <= ratio).shift(1, fill_value=True).values.astype(float)
    return _gm[key]


def design_arrays(W, d, stress=False):
    """Design dict -> (smult, depth, tp) (n, K) for the panel W.
    keys: dc core depth, tp core take-profit, M tier size (0 = no tier), rk 'v30'/'v365', dt tier depth,
    st tier size multiplier, tt tier take-profit (default tp), g crash guard (all coins), pg pump filter ratio on tier
    coins (0 = off; pgc True: on core coins too)."""
    n, K = len(W["idx"]), len(W["syms"])
    is19 = W["is19"]
    M = d.get("M", 0)
    core = np.repeat(is19[None, :], n, axis=0)
    if M > 0:
        R = W["r30o"] if d.get("rk", "v30") == "v30" else W["r365o"]
        tier = (np.nan_to_num(R, nan=1e9) <= M) & ~core
    else:
        tier = np.zeros((n, K), bool)
    sm = core * 1.0 + tier * d.get("st", 1.0)
    dep = np.where(tier, d.get("dt", d["dc"]), d["dc"])
    tp = np.where(tier, d.get("tt", d["tp"]), d["tp"])
    if stress:
        for s in STRESS_COINS:
            j = W["syms"].index(s)
            sm[:, j] = 1.0; dep[:, j] = d["dc"]; tp[:, j] = d["tp"]
    else:
        j = W["syms"].index("FTTUSDT")          # not screened: never in the normal runs
        sm[:, j] = 0.0
    if d.get("pg", 0) > 0:
        pm = pump_mask(W["P"], d["pg"], d.get("pgL", 720))
        sm = np.where(tier | bool(d.get("pgc", False)), sm * pm, sm)
    for s in d.get("excl", ()):                 # leave-coin-out checks
        sm[:, W["syms"].index(s)] = 0.0
    sm = sm * guard_mask(W["P"], d.get("g", 0.35))
    return np.ascontiguousarray(sm), np.ascontiguousarray(dep), np.ascontiguousarray(tp)


_sub = {}


def subpanel(W, cols):
    key = (id(W), tuple(cols))
    if key not in _sub:
        if len(_sub) > 6:
            _sub.clear()
        _sub[key] = {f: v.iloc[:, cols] for f, v in W["P"].items()}
    return _sub[key]


def run_design(W, d, stress=False, start=None, block=None, **over):
    """Run design d on panel W (normal or stressed). start: fresh account from this timestamp; block: list of
    (from, to) timestamps with no new buys. Returns (e, lg, syms_sub, is19_sub)."""
    sm, dep, tp = design_arrays(W, d, stress)
    if block:
        sm = sm.copy()
        for a, b in block:
            sm[(W["idx"] >= pd.Timestamp(a)) & (W["idx"] < pd.Timestamp(b))] = 0.0
    cols = np.where((sm > 0).any(axis=0))[0]
    P = subpanel(W, list(cols))
    kw = dict(depth=dep[:, cols], N=d.get("N", 1), fmax=d.get("fmax", 1.0), smult=sm[:, cols], tp=tp[:, cols],
              hm=d.get("hm", 48), cd=d.get("cd", 24), slip=W["slip"][:, cols], liq=1)
    if stress:
        kw.update(mtp_now=True, cd=kw["cd"] + 1, pen=0.002, pens=0.002)
    kw.update(over)
    idx = W["idx"]
    if start is not None:
        i0 = int(np.searchsorted(idx, pd.Timestamp(start)))
        P = {f: v.iloc[i0:] for f, v in P.items()}
        idx = idx[i0:]
        for k_ in ("depth", "smult", "tp", "slip"):
            kw[k_] = kw[k_][i0:]
    e, lg = wrun(P, idx, **kw)
    syms = [W["syms"][c] for c in cols]
    return e, lg, syms, W["is19"][cols]


# ------------------------------------------------------------------ metrics
def period(e, a, b=None):
    """Multiple, %/month and worst drop on [a, b) (start = value at the last hour before a); same as w90_tenh_lib."""
    idx = e.index
    ia = int(np.searchsorted(idx, a, side="left"))
    ib = len(idx) if b is None else int(np.searchsorted(idx, b, side="left"))
    if ib <= ia:
        return np.nan, np.nan, np.nan
    v = e.values
    s = v[ia - 1] if ia > 0 else 1.0
    x = np.concatenate([[1.0], v[ia:ib] / s])
    yrs = ((idx[ib - 1] + pd.Timedelta(hours=1)) - idx[ia]).total_seconds() / (365.25 * 86400)
    mult = float(x[-1])
    mo = mult ** (1.0 / (12 * yrs)) - 1.0 if mult > 0 else -1.0
    dd = float((x / np.maximum.accumulate(x) - 1.0).min())
    return mult, mo, dd


def year_ret(e, y):
    a = e[e.index < pd.Timestamp(f"{y}-01-01")]
    b = e[e.index < pd.Timestamp(f"{y + 1}-01-01")]
    if len(b) == 0 or len(b) == len(a):
        return np.nan
    s = a.iloc[-1] if len(a) else 1.0
    return float(b.iloc[-1] / s - 1.0)


PERIODS = (("d", T21, T24), ("t", T24, None), ("s", T25, None))


def closed(lg):
    """Trades that count: closed ones (incl. the sale at a coin's last data hour); positions still open at the end
    of the data (why 9) are left out."""
    return lg[lg.why != 9]


def summarize(e, lg, idx, is19=None):
    """Per period d=2021-23, t=2024-26, s=since 2025: multiple, %/month, worst drop, trades bought in the period,
    win rate, average win / loss, worst trade as % of the account; share of the summed account-level P&L
    (ret x order / account at entry) from coins outside the 19, and the trade share of those coins."""
    lg = closed(lg)
    r = {}
    eb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    ret = lg.ret.values
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.array([])
    out19 = ~is19[lg.k.astype(int).values] if (is19 is not None and len(lg)) else np.zeros(len(lg), bool)
    for tag, a, b in PERIODS:
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = period(e, a, b)
        m = np.asarray((eb >= a) & ((eb < b) if b is not None else True), bool)
        r[f"{tag}_n"] = int(m.sum())
        r[f"{tag}_win"] = float((ret[m] > 0).mean()) if m.any() else np.nan
        r[f"{tag}_avgw"] = float(ret[m & (ret > 0)].mean()) if (m & (ret > 0)).any() else np.nan
        r[f"{tag}_avgl"] = float(ret[m & (ret <= 0)].mean()) if (m & (ret <= 0)).any() else np.nan
        r[f"{tag}_wacct"] = float(acct[m].min()) if m.any() else np.nan
        r[f"{tag}_pnl"] = float(acct[m].sum()) if m.any() else 0.0
        r[f"{tag}_pnl_out"] = float(acct[m & out19].sum()) if m.any() else 0.0
        r[f"{tag}_n_out"] = int((m & out19).sum())
        r[f"{tag}_liq"] = int((m & (lg.why.values == 4)).sum())
    r["y2022"] = year_ret(e, 2022)
    r["min_mo"] = min(r["d_mo"], r["t_mo"], r["s_mo"])
    r["min_dd"] = min(r["d_dd"], r["t_dd"], r["s_dd"])
    r["min_win"] = min(r["d_win"], r["t_win"])
    r["ten"] = bool(r["d_mo"] >= 0.10 and r["t_mo"] >= 0.10)
    r["wr90"] = bool(r["d_win"] >= 0.90 and r["t_win"] >= 0.90)
    return r


def evaluate(d, stress=False, W=None, start=None, block=None, **over):
    W = build() if W is None else W
    e, lg, syms, i19 = run_design(W, d, stress, start=start, block=block, **over)
    return summarize(e, lg, e.index, i19), e, lg, syms


def name(d):
    if d.get("M", 0) == 0:
        t = "19 only"
    else:
        t = (f"19 + top{d['M']} others by {d.get('rk', 'v30')} @ {d.get('dt', d['dc']) * 100:.0f}% "
             f"x{d.get('st', 1.0):g} tp{d.get('tt', d['tp']) * 100:g}%")
    pg = f" pump<{d['pg']:g}x{'(all)' if d.get('pgc') else ''}" if d.get("pg", 0) > 0 else ""
    return (f"{t} | core {d['dc'] * 100:.0f}% tp{d['tp'] * 100:g}% N{d.get('N', 1)} f{d.get('fmax', 1.0):g} "
            f"hm{d.get('hm', 48)} cd{d.get('cd', 24)} g{d.get('g', 0.35) * 100:.0f}{pg}")


def fmt(r, pre=""):
    def p(t):
        return (f"WR {r[pre + t + '_win'] * 100:4.1f}% {r[pre + t + '_mo'] * 100:+5.2f}%/mo dd {r[pre + t + '_dd'] * 100:4.0f}%")
    return f"21-23 {p('d')} | 24-26 {p('t')} | 25-26 {p('s')}"


def by_coin(lg, idx, syms, a, b=None):
    """Account-level P&L points (ret x order / account at entry, summed) and trade count per coin for trades bought
    in [a, b)."""
    lg = closed(lg)
    eb = idx[lg.eb.astype(int).values]
    m = np.asarray((eb >= a) & ((eb < b) if b is not None else True), bool)
    t = lg[m]
    acct = t.ret * t.amt / t.eq0
    df = pd.DataFrame({"coin": [syms[int(k)] for k in t.k], "pnl": acct.values, "win": (t.ret > 0).values})
    return df.groupby("coin").agg(pnl=("pnl", "sum"), n=("pnl", "size"), win=("win", "mean")).sort_values("pnl")


# ------------------------------------------------------------------ portable runner for any hourly panel
def portable_spec(d):
    """f(P, syms) -> kwargs for wrun on ANY hourly panel P (e.g. the 2018-20 holdout). Coins in the user's 19 are
    core (and LUNA/FTT/SRM when present, as in the stressed runs); every other coin in P is a tier coin. If P carries quote volume ('qv'), the tier is the monthly top M of
    the panel's non-19 coins by trailing d['rk'] volume (ranked within the panel only, an approximation of the
    all-screened ranking); without 'qv' every non-19 coin in P is tier-eligible. On a panel of only the 19 coins the
    design is the 19-coin core alone. Slippage 0.02% for the 19 coins, 0.1% for others (wrun default)."""
    def spec(P, syms):
        n, K = P["o"].shape
        is19 = np.array([s in E.H19 or s in STRESS_COINS for s in syms])
        core = np.repeat(is19[None, :], n, axis=0)
        tier = ~core
        if d.get("M", 0) > 0 and "qv" in P and (~is19).any():
            pool = pd.Series(~is19, index=list(P["c"].columns))
            look = 720 if d.get("rk", "v30") == "v30" else 365 * 24
            R = month_rank_pool(P["c"], P["qv"], pool, look, 720)
            tier = tier & (np.nan_to_num(R, nan=1e9) <= d["M"])
        elif d.get("M", 0) == 0:
            tier = np.zeros((n, K), bool)
        sm = core * 1.0 + tier * d.get("st", 1.0)
        dep = np.where(tier, d.get("dt", d["dc"]), d["dc"])
        tp = np.where(tier, d.get("tt", d["tp"]), d["tp"])
        if d.get("pg", 0) > 0:
            pm = pump_mask(P, d["pg"], d.get("pgL", 720))
            sm = np.where(tier | bool(d.get("pgc", False)), sm * pm, sm)
        sm = sm * guard_mask(P, d.get("g", 0.35))
        return dict(depth=np.ascontiguousarray(dep), N=d.get("N", 1), fmax=d.get("fmax", 1.0),
                    smult=np.ascontiguousarray(sm), tp=np.ascontiguousarray(tp), hm=d.get("hm", 48),
                    cd=d.get("cd", 24))
    return spec


def runner(P, idx, **kw):
    """engine.run's signature (P, idx, **kw) -> (equity, log with engine.LOG_COLS-compatible first columns).
    Positions still open at the end of the data (why 9) are dropped from the log so trade counts match engine.run."""
    e, lg = wrun(P, idx, **kw)
    return e, lg[lg.why != 9].reset_index(drop=True)
