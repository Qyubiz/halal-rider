"""Shared utilities + fast simulation kernels for the strategy lab."""
import os, glob
import numpy as np, pandas as pd
from numba import njit

FEE = 0.001          # Binance spot VIP0, per side
SLIP = 0.0002        # extra for market orders
DESIGN_END = pd.Timestamp("2024-01-01")   # settings chosen on 2021-2023, tested on 2024-2026
UNIVERSE = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT", "ADAUSDT", "LINKUSDT",
            "DOTUSDT", "AVAXUSDT", "LTCUSDT", "XLMUSDT", "ATOMUSDT", "TRXUSDT", "NEARUSDT", "FILUSDT",
            "ALGOUSDT", "VETUSDT", "THETAUSDT", "ETCUSDT", "XTZUSDT"]

def load(data_dir, syms=UNIVERSE):
    D = {}
    for s in syms:
        p = os.path.join(data_dir, f"{s}.csv")
        if os.path.exists(p):
            D[s] = pd.read_csv(p, index_col=0, parse_dates=True)
    return D

def panels(D):
    idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq="1h")
    return {f: pd.DataFrame({s: d[f] for s, d in D.items()}).reindex(idx) for f in ("o", "h", "l", "c", "v")}

def daily_close(D):
    return pd.DataFrame({s: d.c.resample("1D").last() for s, d in D.items()})

def month_returns(eq):
    m = eq.resample("ME").last()
    base = pd.Series([eq.iloc[0]], index=[eq.index[0] - pd.Timedelta(seconds=1)])
    return pd.concat([base, m]).pct_change().dropna()

def metrics(eq, start=None, end=None):
    e = eq.dropna()
    if start is not None: e = e[e.index >= start]
    if end is not None: e = e[e.index < end]
    e = e / e.iloc[0]
    mr = month_returns(e)
    yrs = (e.index[-1] - e.index[0]).total_seconds() / (365.25 * 86400)
    mult = float(e.iloc[-1])
    return dict(mult=mult, cagr=(mult ** (1 / yrs) - 1) if mult > 0 else -1.0, avg_m=float(mr.mean()),
                med_m=float(mr.median()), pos_m=float((mr > 0).mean()), worst_m=float(mr.min()),
                maxdd=float((e / e.cummax() - 1).min()))

def fmt(m):
    return (f"x{m['mult']:5.2f} CAGR {m['cagr']*100:+5.0f}% avg/mo {m['avg_m']*100:+5.1f}% med/mo {m['med_m']*100:+5.1f}% "
            f"up-mo {m['pos_m']*100:3.0f}% worst-mo {m['worst_m']*100:+4.0f}% maxDD {m['maxdd']*100:4.0f}%")

def pick(rows, dd_floor=-0.5):
    """Choose settings on the DESIGN period only: best CAGR with maxDD above floor, else best Calmar."""
    ok = [r for r in rows if r["d"]["maxdd"] >= dd_floor]
    if ok: return max(ok, key=lambda r: r["d"]["cagr"])
    return max(rows, key=lambda r: r["d"]["cagr"] / max(1e-9, -r["d"]["maxdd"]))

def robustness(rows):
    t = np.array([r["t"]["cagr"] for r in rows])
    return f"all {len(rows)} settings on test: median CAGR {np.median(t)*100:+.0f}%, {np.mean(t>0)*100:.0f}% of settings positive"

# ---------------------------------------------------------------- kernels
@njit(cache=True)
def rebalance_kernel(o, c, wc, band, cost):
    """Equal-weight basket at crypto weight wc[t] (rest USDT). Rebalance at next open when any weight
    drifts more than `band` (relative) from target."""
    n, K = c.shape
    units = np.zeros(K); cash = 1.0; eq = np.empty(n)
    pending = False; target = 0.0; turnover = 0.0
    for t in range(n):
        if pending:
            V = cash
            for k in range(K): V += units[k] * o[t, k]
            each = target * V * (1.0 - cost) / K
            traded = 0.0; spent = 0.0
            for k in range(K):
                nu = each / o[t, k]
                traded += abs(nu - units[k]) * o[t, k]
                units[k] = nu; spent += nu * o[t, k]
            cash = V - spent - traded * cost
            turnover += traded / V
            pending = False
        V = cash
        for k in range(K): V += units[k] * c[t, k]
        eq[t] = V
        wt = wc[t]; tot = 0.0; dev = 0.0
        for k in range(K):
            w = units[k] * c[t, k] / V
            tot += w
            if wt > 1e-9:
                d = abs(w / (wt / K) - 1.0)
                if d > dev: dev = d
        if wt <= 1e-9:
            if tot > 1e-4: pending = True; target = wt
        elif abs(tot / wt - 1.0) > band or dev > band:
            pending = True; target = wt
    return eq, turnover

@njit(cache=True)
def grid_kernel(o, h, l, c, gate, s, L, stop, cooldown, liq_gate_off, eps, fee, slip):
    """Trailing geometric grid for one coin. Slot k buys at A(1-s)^(k+1), sells one step up.
    Conservative: no buy+sell of the same slot inside one candle; fills need price to trade through by eps."""
    n = len(o)
    eq = np.empty(n); cash = 1.0
    units = np.zeros(L); bbar = np.full(L, -1, np.int64); filled = np.zeros(L, np.bool_)
    lev = np.empty(L); tgt = np.empty(L)
    A = o[0]
    for k in range(L):
        lev[k] = A * (1.0 - s) ** (k + 1); tgt[k] = A * (1.0 - s) ** k
    lot = cash / L; cool_until = -1; rounds = 0
    pts = np.empty(4)
    for t in range(n):
        can_buy = gate[t] and t >= cool_until
        for k in range(L):
            if filled[k] and bbar[k] < t and o[t] >= tgt[k] * (1.0 + eps):
                cash += units[k] * o[t] * (1.0 - fee); units[k] = 0.0; filled[k] = False; rounds += 1
        if c[t] >= o[t]:
            pts[0] = o[t]; pts[1] = l[t]; pts[2] = h[t]; pts[3] = c[t]
        else:
            pts[0] = o[t]; pts[1] = h[t]; pts[2] = l[t]; pts[3] = c[t]
        for g in range(3):
            a = pts[g]; b = pts[g + 1]
            if b < a:
                if can_buy:
                    for k in range(L):
                        if (not filled[k]) and a >= lev[k] and b <= lev[k] * (1.0 - eps):
                            amt = lot if cash >= lot else cash
                            if amt > 1e-12:
                                cash -= amt; units[k] = amt * (1.0 - fee) / lev[k]; filled[k] = True; bbar[k] = t
            elif b > a:
                for k in range(L):
                    if filled[k] and bbar[k] < t and b >= tgt[k] * (1.0 + eps):
                        cash += units[k] * tgt[k] * (1.0 - fee); units[k] = 0.0; filled[k] = False; rounds += 1
        anyf = False
        for k in range(L):
            if filled[k]: anyf = True
        if anyf:
            stopped = stop > 0.0 and c[t] < lev[L - 1] * (1.0 - stop)
            if stopped or (liq_gate_off and not gate[t]):
                for k in range(L):
                    if filled[k]:
                        cash += units[k] * c[t] * (1.0 - fee - slip); units[k] = 0.0; filled[k] = False
                anyf = False
                if stopped: cool_until = t + cooldown
        if not anyf:
            if c[t] > A * (1.0 + s) or c[t] < lev[0] or not can_buy:
                A = c[t]
                for k in range(L):
                    lev[k] = A * (1.0 - s) ** (k + 1); tgt[k] = A * (1.0 - s) ** k
                lot = cash / L
        v = cash
        for k in range(L): v += units[k] * c[t]
        eq[t] = v
    return eq, rounds

@njit(cache=True)
def wick_kernel(o, h, l, c, ref, depths, fracs, tp, hmax, sl, cd, eps, fee, slip):
    """Standing limit buys at ref*(1-depth) on every coin (shared cash). Exit: take-profit limit,
    stop-loss, or time stop. Conservative: same-candle stop counts, same-candle take-profit does not."""
    n, K = c.shape; m = depths.shape[0]
    units = np.zeros((K, m)); entry = np.zeros((K, m)); amt0 = np.zeros((K, m))
    ebar = np.full((K, m), -1, np.int64); okbar = np.zeros((K, m), np.int64)
    lastc = np.full(K, np.nan); want = np.zeros((K, m)); fpx = np.zeros((K, m))
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 6)); nl = 0
    for t in range(n):
        for k in range(K):
            if np.isnan(o[t, k]): continue
            for j in range(m):
                if units[k, j] > 0.0 and ebar[k, j] < t:
                    e = entry[k, j]; tpp = e * (1.0 + tp)
                    slp = e * (1.0 - sl) if sl > 0.0 else -1.0
                    px = -1.0
                    if slp > 0.0 and o[t, k] <= slp: px = o[t, k] * (1.0 - slip)
                    elif o[t, k] >= tpp: px = o[t, k]
                    elif slp > 0.0 and l[t, k] <= slp: px = slp * (1.0 - slip)
                    elif h[t, k] >= tpp * (1.0 + eps): px = tpp
                    elif t - ebar[k, j] >= hmax: px = c[t, k] * (1.0 - slip)
                    if px > 0.0:
                        proceeds = units[k, j] * px * (1.0 - fee); cash += proceeds
                        if nl < maxlog:
                            lg[nl, 0] = ebar[k, j]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = j
                            lg[nl, 4] = proceeds / amt0[k, j] - 1.0; lg[nl, 5] = amt0[k, j]; nl += 1
                        units[k, j] = 0.0; okbar[k, j] = t + cd
        tot = 0.0
        for k in range(K):
            for j in range(m):
                want[k, j] = 0.0
                if np.isnan(o[t, k]) or np.isnan(ref[t, k]) or units[k, j] > 0.0 or t < okbar[k, j]: continue
                lim = ref[t, k] * (1.0 - depths[j])
                if l[t, k] <= lim * (1.0 - eps):
                    fpx[k, j] = min(lim, o[t, k]); want[k, j] = fracs[j] * eq_prev; tot += want[k, j]
        scale = 1.0
        if tot > cash: scale = cash / tot if cash > 0.0 else 0.0
        for k in range(K):
            for j in range(m):
                if want[k, j] > 0.0:
                    a = want[k, j] * scale
                    if a <= 1e-9: continue
                    cash -= a; units[k, j] = a * (1.0 - fee) / fpx[k, j]
                    entry[k, j] = fpx[k, j]; amt0[k, j] = a; ebar[k, j] = t
                    if sl > 0.0 and l[t, k] <= fpx[k, j] * (1.0 - sl):
                        px = fpx[k, j] * (1.0 - sl) * (1.0 - slip)
                        proceeds = units[k, j] * px * (1.0 - fee); cash += proceeds
                        if nl < maxlog:
                            lg[nl, 0] = t; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = j
                            lg[nl, 4] = proceeds / a - 1.0; lg[nl, 5] = a; nl += 1
                        units[k, j] = 0.0; okbar[k, j] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]): lastc[k] = c[t, k]
            for j in range(m):
                if units[k, j] > 0.0: v += units[k, j] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()

@njit(cache=True)
def wick2(o, h, l, c, ref, depths, fracs, tp, hmax, sl, cd, eps, fee, slip, pen, open_only, alive):
    """wick_kernel + realism switches.
    pen: entry fill worse by this fraction. open_only: only buy at the candle open when the open is already
    below the level (what an hourly-checking bot can do), paying taker fee+slip.
    alive[t,k]=False after a coin is delisted: positions are stuck at last price, no trading."""
    n, K = c.shape; m = depths.shape[0]
    units = np.zeros((K, m)); entry = np.zeros((K, m)); amt0 = np.zeros((K, m))
    ebar = np.full((K, m), -1, np.int64); okbar = np.zeros((K, m), np.int64)
    lastc = np.full(K, np.nan); want = np.zeros((K, m)); fpx = np.zeros((K, m))
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 7)); nl = 0
    for t in range(n):
        for k in range(K):
            if np.isnan(o[t, k]) or not alive[t, k]: continue
            for j in range(m):
                if units[k, j] > 0.0 and ebar[k, j] < t:
                    e = entry[k, j]; tpp = e * (1.0 + tp)
                    slp = e * (1.0 - sl) if sl > 0.0 else -1.0
                    px = -1.0
                    if slp > 0.0 and o[t, k] <= slp: px = o[t, k] * (1.0 - slip)
                    elif o[t, k] >= tpp: px = o[t, k]
                    elif slp > 0.0 and l[t, k] <= slp: px = slp * (1.0 - slip)
                    elif h[t, k] >= tpp * (1.0 + eps): px = tpp
                    elif t - ebar[k, j] >= hmax: px = c[t, k] * (1.0 - slip)
                    if px > 0.0:
                        proceeds = units[k, j] * px * (1.0 - fee); cash += proceeds
                        if nl < maxlog:
                            lg[nl, 0] = ebar[k, j]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = j
                            lg[nl, 4] = proceeds / amt0[k, j] - 1.0; lg[nl, 5] = amt0[k, j]; lg[nl, 6] = fpx[k, j] < 0; nl += 1
                        units[k, j] = 0.0; okbar[k, j] = t + cd
        tot = 0.0
        for k in range(K):
            for j in range(m):
                want[k, j] = 0.0
                if np.isnan(o[t, k]) or np.isnan(ref[t, k]) or not alive[t, k] or units[k, j] > 0.0 or t < okbar[k, j]: continue
                lim = ref[t, k] * (1.0 - depths[j])
                if open_only:
                    if o[t, k] <= lim:
                        fpx[k, j] = o[t, k] * (1.0 + slip) * (1.0 + pen); want[k, j] = fracs[j] * eq_prev; tot += want[k, j]
                elif l[t, k] <= lim * (1.0 - eps):
                    fpx[k, j] = min(lim, o[t, k]) * (1.0 + pen); want[k, j] = fracs[j] * eq_prev; tot += want[k, j]
        scale = 1.0
        if tot > cash: scale = cash / tot if cash > 0.0 else 0.0
        for k in range(K):
            for j in range(m):
                if want[k, j] > 0.0:
                    a = want[k, j] * scale
                    if a <= 1e-9: continue
                    lim = ref[t, k] * (1.0 - depths[j])
                    gapfill = o[t, k] <= lim
                    cash -= a; units[k, j] = a * (1.0 - fee) / fpx[k, j]
                    entry[k, j] = fpx[k, j]; amt0[k, j] = a; ebar[k, j] = t
                    if sl > 0.0 and l[t, k] <= fpx[k, j] * (1.0 - sl):
                        px = fpx[k, j] * (1.0 - sl) * (1.0 - slip)
                        proceeds = units[k, j] * px * (1.0 - fee); cash += proceeds
                        if nl < maxlog:
                            lg[nl, 0] = t; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = j
                            lg[nl, 4] = proceeds / a - 1.0; lg[nl, 5] = a; lg[nl, 6] = 1.0 if gapfill else 0.0; nl += 1
                        units[k, j] = 0.0; okbar[k, j] = t + cd
                    elif gapfill:
                        fpx[k, j] = -fpx[k, j]   # mark: this position was filled at the open (no resting order needed)
                        entry[k, j] = -fpx[k, j]
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]): lastc[k] = c[t, k]
            for j in range(m):
                if units[k, j] > 0.0: v += units[k, j] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()

@njit(cache=True)
def dyn_kernel(o, h, l, c, ref, depth, N, fmax, tp, hm, cd, eps, fee, slip, pen):
    """(Orders are sized with the cash available at the START of each candle; sells during the candle free cash only for the next candle.)
    Hourly bot with FULLY-FUNDED resting bids that move each hour to the N coins closest to their trigger
    (ref*(1-depth)). If a coin is already below its trigger at the hour's open, it is bought at the open (taker).
    Orders never exceed available cash. Exit: +tp limit sell (not same candle) or time stop after hm hours."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 400000; lg = np.zeros((maxlog, 6)); nl = 0
    for t in range(n):
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]): continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth)) - 1.0
        order = np.argsort(dist)
        nsel = 0
        for q in range(K):
            if dist[order[q]] < 1e8 and nsel < N: nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth)
                fill = -1.0; f_fee = fee
                if o[t, k] <= lim: fill = o[t, k] * (1.0 + slip) * (1.0 + pen)
                elif l[t, k] <= lim * (1.0 - eps): fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - f_fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                e = entry[k]; tpp = e * (1.0 + tp); px = -1.0
                if o[t, k] >= tpp: px = o[t, k]
                elif h[t, k] >= tpp * (1.0 + eps): px = tpp
                elif t - ebar[k] >= hm: px = c[t, k] * (1.0 - slip)
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = 0
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]): lastc[k] = c[t, k]
            if units[k] > 0.0: v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()
