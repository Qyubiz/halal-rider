"""Refinement engines for the Panic Buyer.

gkernel: lab.dyn_kernel (the validated live strategy) plus switches for the section 8 ideas: per-coin and
per-hour trigger depth, per-hour order count and order cap, per-coin size multipliers, per-coin take-profit,
partial take-profit with a trailing or second target for the rest, and a per-hour time stop. With every
switch at the live setting it gives exactly the same equity curve as lab.dyn_kernel (see check_engine.py).

mkernel: the same bot on 1-minute candles, so a take-profit can be placed minutes after the buy fills
instead of at the next hourly cycle. delay=0 is the live bot (take-profit placed at the next hour) and
matches gkernel on hourly candles built from the same minutes.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
import lab  # noqa: E402,F401  (original engine, used as the reference)
from lab import metrics  # noqa: E402

DATA = os.environ.get("PB_DATA", os.path.expanduser("~/pb_data"))
DESIGN_END = pd.Timestamp("2024-01-01")
FEE, EPS, SLIP_MAJOR, SLIP_OTHER = 0.001, 0.001, 0.0002, 0.001
H19 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT",
       "AVAXUSDT", "SUIUSDT", "LTCUSDT", "BCHUSDT", "HBARUSDT", "DOTUSDT", "NEARUSDT", "ARBUSDT", "FETUSDT",
       "RENDERUSDT"]
DEAD = {"LUNAUSDT": "2022-05-13 23:00", "FTTUSDT": "2023-09-22 07:00", "SRMUSDT": "2022-11-28 23:00"}
LIVE = dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24)
LOG_COLS = ["eb", "xb", "k", "why", "ret", "amt", "entry"]      # why: 0 take-profit, 1 time stop, 2 trail, 3 tp2


# ------------------------------------------------------------------ data
def read_hourly(sym, sub="1h"):
    p = os.path.join(DATA, sub, f"{sym}.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True) if os.path.exists(p) else None


def load_h19(dead=False, sub="1h", start="2021-01-01"):
    """The 19 live coins (RENDER joined to its old ticker RNDR), optionally plus the LUNA/FTT/SRM collapses."""
    D = {}
    for s in H19:
        d = read_hourly(s, sub)
        if d is None:
            raise FileNotFoundError(f"{DATA}/{sub}/{s}.csv missing: run fetch_all.sh")
        if s == "RENDERUSDT":
            r = read_hourly("RNDRUSDT", sub)
            if r is not None:
                d = pd.concat([r, d])
                d = d[~d.index.duplicated(keep="last")].sort_index()     # RENDER wins where both exist
        D[s] = d
    if dead:
        for s, cut in DEAD.items():
            d = read_hourly(s, sub)
            D[s] = d[d.index <= cut]
    return {s: d[d.index >= start] for s, d in D.items()}


def panel(D, freq="1h"):
    syms = list(D)
    idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq=freq)
    P = {f: pd.DataFrame({s: D[s][f] for s in syms}).reindex(idx) for f in ("o", "h", "l", "c")}
    P["cf"] = P["c"].ffill()
    P["ref"] = P["c"].rolling(24, min_periods=1).max().shift(1)
    if "qv" in next(iter(D.values())):
        P["qv"] = pd.DataFrame({s: D[s]["qv"] for s in syms}).reindex(idx)
    return syms, idx, P


# ------------------------------------------------------------------ hourly engine
@njit(cache=True)
def gkernel(o, h, l, c, ref, depth, nord, fmax, smult, tpa, part, tp2, trail, hm, cd, eps, fee, slip, pen,
            tpwait=1, tprest=True):
    """Hourly bot with fully funded resting bids on the nord[t] coins closest to ref*(1-depth[t,k]).
    Order size: min(fmax[t]*equity, cash/nsel) * smult[t,k] (smult 0 = coin skipped this hour).
    Exit: sell `part` at entry*(1+tpa[entry hour,k]) (limit, not in the fill hour); the rest (if part<1)
    sells at entry*(1+tp2) (tp2>0) or on a `trail` drop from its high, checked at each close (trail>0);
    everything left sells at the close once held hm[t] hours. Cash from sales is usable the next hour.
    tpwait: the take-profit is placed at the tpwait-th hourly cycle after the fill (live bot: 1). tprest=False:
    no resting order, the bot only sells at an hourly check when the price is already at the target."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 7)); nl = 0
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
                fill = -1.0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen)
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    a = min(per * smult[t, k], cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0
                    tpp[k] = fill * (1.0 + tpa[t, k]); stage[k] = 0; peak[k] = fill
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                px = -1.0; why = 0.0
                if stage[k] == 0:
                    if t - ebar[k] < tpwait:
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
                            px = c[t, k] * (1.0 - slip[k]); why = 2.0
                if px < 0.0 and t - ebar[k] >= hm[t]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds; got[k] += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = got[k] / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]):
                lastc[k] = c[t, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def full(n, K, x, dtype=np.float64):
    """Broadcast a scalar, an (n,) or an (n,K) setting to the shape the kernel wants."""
    a = np.asarray(x, dtype=dtype)
    if a.ndim == 0:
        return np.full((n, K), a, dtype) if K else np.full(n, a, dtype)
    if K and a.ndim == 1:
        return np.repeat(a[:, None], K, axis=1)
    return np.ascontiguousarray(a)


def run(P, idx, depth=LIVE["depth"], N=LIVE["N"], fmax=LIVE["fmax"], smult=1.0, tp=LIVE["tp"], part=1.0, tp2=0.0,
        trail=0.0, hm=LIVE["hm"], cd=LIVE["cd"], eps=EPS, fee=FEE, slip=SLIP_MAJOR, pen=0.0, tpwait=1, tprest=True):
    O = P["o"].values; n, K = O.shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = gkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values,
                     full(n, K, depth), full(n, 0, N, np.int64), full(n, 0, fmax), full(n, K, smult), full(n, K, tp),
                     float(part), float(tp2), float(trail), full(n, 0, hm, np.int64), int(cd), eps, fee, sl, pen,
                     int(tpwait), bool(tprest))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=LOG_COLS)


# ------------------------------------------------------------------ 1-minute engine
@njit(cache=True)
def mkernel(o1, h1, l1, c1, ref, depth, nord, fmax, tpa, hm, cd, eps, fee, slip, pen, delay):
    """gkernel (full take-profit only) on 1-minute candles. Orders are placed at each hour's first minute as in
    the live bot. The take-profit becomes active `delay` minutes after the buy fills (delay<=0: at the next hour,
    like the live bot). A resting take-profit fills when a minute's high trades eps through it; when it is
    first placed, or at an hour's first minute, an open already above it fills at that open."""
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); act = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(nh); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(nh):
        m0 = 60 * t
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o1[m0, k]) or np.isnan(ref[t, k]):
                continue
            dist[k] = o1[m0, k] / (ref[t, k] * (1.0 - depth[t, k])) - 1.0
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
                fill = -1.0; fm = -1
                if o1[m0, k] <= lim:
                    fill = o1[m0, k] * (1.0 + slip[k]) * (1.0 + pen); fm = m0
                else:
                    for i in range(m0, m0 + 60):
                        if l1[i, k] <= lim * (1.0 - eps):
                            fill = lim * (1.0 + pen); fm = i
                            break
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; tpp[k] = fill * (1.0 + tpa[t, k])
                    act[k] = fm + delay if delay > 0 else m0 + 60
        for k in range(K):
            if units[k] <= 0.0:
                continue
            px = -1.0; why = 0.0; xm = -1
            s = act[k] if act[k] > m0 else m0
            for i in range(s, m0 + 60):
                if np.isnan(o1[i, k]):
                    continue
                if (i == act[k] or i == m0) and o1[i, k] >= tpp[k]:
                    px = o1[i, k]; xm = i
                    break
                if h1[i, k] >= tpp[k] * (1.0 + eps):
                    px = tpp[k]; xm = i
                    break
            if px < 0.0 and ebar[k] < t and t - ebar[k] >= hm[t] and not np.isnan(c1[m0 + 59, k]):
                px = c1[m0 + 59, k] * (1.0 - slip[k]); why = 1.0; xm = m0 + 59
            if px > 0.0:
                proceeds = units[k] * px * (1.0 - fee); cash += proceeds
                if nl < maxlog:
                    lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                    lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]; lg[nl, 7] = xm
                    nl += 1
                units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c1[m0 + 59, k]):
                lastc[k] = c1[m0 + 59, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy()


def minute_panel(syms, start="2021-01-01", sub="1m", cuts=None):
    """Aligned 1-minute arrays (hour-aligned, missing minutes flat at the last close) + hourly candles built
    from them. cuts: {symbol: last hour kept} for the collapse coins."""
    def read1m(s):
        p = os.path.join(DATA, sub, f"{s}.npz")
        if os.path.exists(p):                      # shared format: plain arrays, no pickle
            z = np.load(p, allow_pickle=False)
            return pd.DataFrame({f: z[f] for f in ("o", "h", "l", "c")}, index=pd.to_datetime(z["t"], unit="s"))
        p = os.path.join(DATA, sub, f"{s}.pkl")
        return pd.read_pickle(p) if os.path.exists(p) else None

    D = {}
    for s in syms:
        d = read1m(s)
        if s == "RENDERUSDT" and read1m("RNDRUSDT") is not None:
            r = read1m("RNDRUSDT")
            d = pd.concat([r, d])
            d = d[~d.index.duplicated(keep="last")].sort_index()
        D[s] = d[d.index >= start]
    t0 = min(d.index[0] for d in D.values()).floor("1h")
    t1 = max(d.index[-1] for d in D.values()).floor("1h") + pd.Timedelta(minutes=59)
    idx = pd.date_range(t0, t1, freq="1min")
    A = {f: np.full((len(idx), len(syms)), np.nan) for f in ("o", "h", "l", "c")}
    for j, s in enumerate(syms):
        d = D[s].reindex(idx)
        first = d.c.first_valid_index()
        cc = d.c.ffill()
        live = idx >= first
        for f in ("o", "h", "l"):
            col = d[f].where(d[f].notna(), cc)
            A[f][:, j] = np.where(live, col.values, np.nan)
        A["c"][:, j] = np.where(live, cc.values, np.nan)
        if cuts and s in cuts:
            gone = idx > pd.Timestamp(cuts[s]) + pd.Timedelta(minutes=59)
            for f in A:
                A[f][gone, j] = np.nan
    hidx = idx[::60]
    nh = len(hidx)
    Hh = {"o": A["o"][::60], "c": A["c"][59::60],
          "h": np.fmax.reduce(A["h"].reshape(nh, 60, -1), axis=1),
          "l": np.fmin.reduce(A["l"].reshape(nh, 60, -1), axis=1)}
    Ph = {f: pd.DataFrame(v, hidx, syms) for f, v in Hh.items()}
    Ph["cf"] = Ph["c"].ffill()
    Ph["ref"] = Ph["c"].rolling(24, min_periods=1).max().shift(1)
    return idx, A, hidx, Ph


def mrun(A, hidx, Ph, delay, depth=LIVE["depth"], N=LIVE["N"], fmax=LIVE["fmax"], tp=LIVE["tp"], hm=LIVE["hm"],
         cd=LIVE["cd"], eps=EPS, fee=FEE, slip=SLIP_MAJOR, pen=0.0):
    nh, K = Ph["c"].shape
    sl = np.full(K, slip) if np.ndim(slip) == 0 else np.asarray(slip, np.float64)
    eq, lg = mkernel(A["o"], A["h"], A["l"], A["c"], Ph["ref"].values, full(nh, K, depth), full(nh, 0, N, np.int64),
                     full(nh, 0, fmax), full(nh, K, tp), full(nh, 0, hm, np.int64), int(cd), eps, fee, sl, pen,
                     int(delay))
    return pd.Series(eq, hidx), pd.DataFrame(lg, columns=LOG_COLS + ["xm"])


# ------------------------------------------------------------------ scoring
def years(e):
    y = e.resample("YE").last()
    y = pd.concat([pd.Series([e.iloc[0]], [e.index[0]]), y]).pct_change().dropna()
    return {k.year: float(v) for k, v in y.items()}


def score(e, lg=None, idx=None):
    """Design (2021-23) and test (2024-) metrics, the 2022 bear year, and trade stats since 2024."""
    d, t = metrics(e, end=DESIGN_END), metrics(e, start=DESIGN_END)
    y = years(e)
    out = {"d_mult": d["mult"], "d_dd": d["maxdd"], "d_cagr": d["cagr"], "d_worst_m": d["worst_m"], "d_pos_m": d["pos_m"],
           "t_mult": t["mult"], "t_dd": t["maxdd"], "t_cagr": t["cagr"], "t_worst_m": t["worst_m"], "t_pos_m": t["pos_m"],
           "y2022": y.get(2022, np.nan)}
    if lg is not None and idx is not None and len(lg):
        tt = lg[idx[lg.eb.astype(int).values] >= DESIGN_END]
        out.update(t_trades=len(tt), t_win=float((tt.ret > 0).mean()) if len(tt) else np.nan,
                   t_avg=float(tt.ret.mean()) if len(tt) else np.nan, t_worst=float(tt.ret.min()) if len(tt) else np.nan)
    return out


DD_TOL = 0.01   # a drawdown up to 1 point deeper still counts as "not worse"


def beats(c, b, period="both"):
    """The user's rule: better compounded return than the live settings, with no deeper max drawdown (1-point
    tolerance), in 2021-23 and in 2024-26."""
    d = c["d_mult"] > b["d_mult"] and c["d_dd"] >= b["d_dd"] - DD_TOL
    t = c["t_mult"] > b["t_mult"] and c["t_dd"] >= b["t_dd"] - DD_TOL
    return {"design": d, "test": t, "both": d and t}[period]


def pick_on_design(rows, base):
    """Choose a variant using 2021-23 only: it must beat the live settings in 2021-23, keep 2022 above -5% and
    the 2021-23 drawdown above -35%; among those, the best 2021-23 CAGR / |max drawdown|. None if nothing beats."""
    ok = [r for r in rows if beats(r, base, "design") and r["y2022"] >= -0.05 and r["d_dd"] >= -0.35]
    return max(ok, key=lambda r: r["d_cagr"] / max(1e-9, -r["d_dd"])) if ok else None


def line(name, r):
    return (f"{name:46s} 21-23 x{r['d_mult']:5.2f} DD {r['d_dd']*100:4.0f}% 2022 {r['y2022']*100:+4.0f}% | "
            f"24-26 x{r['t_mult']:5.2f} DD {r['t_dd']*100:4.0f}% worst-mo {r['t_worst_m']*100:+3.0f}% up-mo {r['t_pos_m']*100:3.0f}%")
