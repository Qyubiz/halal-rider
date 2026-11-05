"""m10 "scalp": tiny-target, high-frequency dip scalping on 1-minute candles.

Kernel `skernel` is x10_velocity_lib.vkernel (itself proven equal to engine.mkernel) with the trailing exit, the second
"flash" trigger, re-dip and losing-exit cooldown removed (they stay off here) and these switches added:

  W      reference window in MINUTES: the trigger is max(1-minute close of the last W closed minutes) * (1 - depth),
         kept with a monotone deque inside the kernel (no lookahead: minute i sees closes up to minute i-1).
         W=0: the hourly 24h reference `ref` (vkernel).
  wick   wick trimming as in x10_critic_wicks.py: a minute's high counts at most max(open, close) * (1 + wick), its
         low at least min(open, close) * (1 - wick), for every resting-order fill test (bids, take-profits). 0 = off.
  tpo    1: vkernel take-profit (an open already above the target at the activation minute / an hour's first minute
         fills at that open). 0: a take-profit only ever fills at its limit, when the (trimmed) high trades eps through.
  hmm    time stop in minutes after the fill minute (market sell at that minute's close - slippage). -1: hourly `hm`.
  sl     stop-loss: a 1-minute close <= entry * (1 - sl) sells at the NEXT minute's open - slippage. 0 = off.
  bem, tpbe  after bem minutes the take-profit target drops to entry * (1 + tpbe) (break-even exit). bem<0 = off.
  guard  crash guard: no new bid on a coin whose price is more than `guard` under gref (its highest hourly close of the
         168 closed hours before the current hour). 0 = off.
  mdd    per-minute marking: the kernel also returns eqlo[t] = lowest marked equity (1-minute closes) inside hour t.

With W=0, wick=0, tpo=1, hmm=-1, sl=0, bem=-1, guard=0 it reproduces vkernel(xmode=0, look2=0, redip=0, cdl=-1)
(m10_scalp_check.py). Fills: limit buys fill at their limit only when the (trimmed) low trades eps=0.1% through it;
a coin already under its trigger at the cycle minute's open is bought at that open + slippage (market). With delay=1
a take-profit is active from the minute after the buy fills (no same-minute fill-and-exit). pen/spen make every buy /
sell fill that much worse (stress). Fees 0.1% per side. Orders are fully funded: bids reserve cash, sum <= cash.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
from lab import metrics, month_returns  # noqa: E402

SCR = "/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad"
OUT = os.environ.get("PB_OUT", os.path.join(SCR, "pb_out_m10", "scalp"))
XCACHE = os.environ.get("PB_XCACHE", os.path.join(SCR, "pb_out_x10", "velocity", "cache"))   # x10 minute cache
HCACHE = os.path.join(OUT, "hcache")
SYMS = E.H19 + list(E.DEAD)
T18, T21, T24, T25 = (pd.Timestamp(x) for x in ("2018-01-01", "2021-01-01", "2024-01-01", "2025-01-01"))
MD = 365.25 / 12
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "xm", "fm", "eqb"]   # why 0 tp, 1 time stop, 2 stop-loss, 3 be
HOLD_MAJOR = {"BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "LTCUSDT", "ADAUSDT", "TRXUSDT", "LINKUSDT", "BCHUSDT"}


# ------------------------------------------------------------------ data
def load(K=19):
    """2021-26 minute panel from the x10 float32 cache (memory-mapped, shared): A (first K coins), hidx, Ph (hourly
    o h l c cf ref), SLIP (hours, K) market slippage (0.02% top-50 by volume, else 0.1%)."""
    A = {f: np.load(os.path.join(XCACHE, f"m_{f}.npy"), mmap_mode="r")[:, :K] for f in ("o", "h", "l", "c")}
    hidx = pd.DatetimeIndex(pd.to_datetime(np.load(os.path.join(XCACHE, "hidx.npy")), unit="s"))
    Ph = {f: pd.DataFrame(np.load(os.path.join(XCACHE, f"h_{f}.npy"))[:, :K], hidx, SYMS[:K]) for f in ("o", "h", "l", "c")}
    Ph["cf"] = Ph["c"].ffill()
    Ph["ref"] = Ph["c"].rolling(24, min_periods=1).max().shift(1)
    return A, hidx, Ph, np.load(os.path.join(XCACHE, "slip.npy"))[:, :K]


def build_holdout():
    """2017-12 .. 2020-12 minute panel from PB_HOLDOUT/1m_pre (BCH = BCHABC until the rename, as engine.load_pre),
    same alignment as engine.minute_panel (hour-aligned, missing minutes flat at the last close), stored float32."""
    os.makedirs(HCACHE, exist_ok=True)
    src = os.path.join(E.HOLDOUT, "1m_pre")

    def rd(s):
        p = os.path.join(src, f"{s}.npz")
        if not os.path.exists(p):
            return None
        z = np.load(p, allow_pickle=False)
        d = pd.DataFrame({f: z[f] for f in ("o", "h", "l", "c")}, index=pd.to_datetime(z["t"], unit="s"))
        d = d[(d.index == d.index.floor("1min")) & (d.index < E.H_END)]
        return d[~d.index.duplicated()].sort_index()
    D = {}
    for s in E.H19:
        d = rd(s)
        if s == "BCHUSDT":
            a = rd("BCHABCUSDT")
            if a is not None:
                d = pd.concat([a, d]) if d is not None else a
                d = d[~d.index.duplicated(keep="last")].sort_index()
        if d is not None and len(d) > 60 * 24 * 30:
            D[s] = d
    syms = list(D)
    t0 = min(d.index[0] for d in D.values()).floor("1h")
    t1 = E.H_END - pd.Timedelta(minutes=1)
    idx = pd.date_range(t0, t1, freq="1min")
    nm = len(idx)
    for f in ("o", "h", "l", "c"):
        np.lib.format.open_memmap(os.path.join(HCACHE, f"m_{f}.npy"), "w+", np.float32, (nm, len(syms))).flush()
    M = {f: np.load(os.path.join(HCACHE, f"m_{f}.npy"), mmap_mode="r+") for f in ("o", "h", "l", "c")}
    for j, s in enumerate(syms):
        d = D[s].reindex(idx)
        first = d.c.first_valid_index()
        cc = d.c.ffill()
        live = idx >= first
        for f in ("o", "h", "l"):
            col = d[f].where(d[f].notna(), cc)
            M[f][:, j] = np.where(live, col.values, np.nan).astype(np.float32)
        M["c"][:, j] = np.where(live, cc.values, np.nan).astype(np.float32)
        del d, cc
    for f in M:
        M[f].flush()
    hidx = idx[::60]
    np.save(os.path.join(HCACHE, "hidx.npy"), hidx.values.astype("datetime64[s]").astype(np.int64))
    pd.Series(syms).to_csv(os.path.join(HCACHE, "syms.csv"), index=False)
    print(f"holdout cache: {len(syms)} coins, {idx[0]} .. {idx[-1]} -> {HCACHE}", flush=True)


def load_holdout():
    if not os.path.exists(os.path.join(HCACHE, "syms.csv")):
        build_holdout()
    syms = list(pd.read_csv(os.path.join(HCACHE, "syms.csv")).iloc[:, 0])
    A = {f: np.load(os.path.join(HCACHE, f"m_{f}.npy"), mmap_mode="r") for f in ("o", "h", "l", "c")}
    hidx = pd.DatetimeIndex(pd.to_datetime(np.load(os.path.join(HCACHE, "hidx.npy")), unit="s"))
    nh = len(hidx)
    H = {"o": np.asarray(A["o"][::60]), "c": np.asarray(A["c"][59::60]),
         "h": np.fmax.reduce(np.asarray(A["h"]).reshape(nh, 60, -1), axis=1),
         "l": np.fmin.reduce(np.asarray(A["l"]).reshape(nh, 60, -1), axis=1)}
    Ph = {f: pd.DataFrame(v, hidx, syms) for f, v in H.items()}
    Ph["cf"] = Ph["c"].ffill()
    Ph["ref"] = Ph["c"].rolling(24, min_periods=1).max().shift(1)
    SLIP = np.tile(np.array([E.SLIP_MAJOR if s in HOLD_MAJOR else 0.001 for s in syms]), (nh, 1))
    return A, hidx, Ph, SLIP, syms


def guard_ref(Ph):
    """Highest hourly close of the 168 closed hours before each hour (crash-guard reference)."""
    return np.ascontiguousarray(Ph["c"].rolling(168, min_periods=1).max().shift(1).values)


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def skernel(o1, h1, l1, c1, ref, depth, nord, fmax, tpa, hm, cd, eps, fee, slip, pen, delay,
            cyc, rearm, cdm, near, spen, W, wick, tpo, hmm, sl, bem, tpbe, gref, guard, maxlog):
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okm = np.zeros(K, np.int64); act = np.zeros(K, np.int64)
    fmin = np.zeros(K, np.int64); eqb = np.zeros(K); lastm = np.zeros(K); pend = np.zeros(K, np.int64)
    resv = np.zeros(K); rlim = np.zeros(K); nres = 0; rsum = 0.0
    limk = np.zeros(K)
    lastc = np.full(K, np.nan); dist = np.empty(K); order = np.zeros(K, np.int64)
    # monotone deque (ring buffer) per coin for the W-minute max of closes
    cap = W + 2 if W > 0 else 1
    dq = np.zeros((K, cap), np.int64); dh = np.zeros(K, np.int64); dn = np.zeros(K, np.int64)
    cash = 1.0; eq = np.empty(nh); eqlo = np.empty(nh); inv = np.empty(nh); eq_prev = 1.0
    lg = np.zeros((maxlog, 10)); nl = 0; ntr = 0
    due = -1
    for t in range(nh):
        m0 = 60 * t
        lo_t = 1e300
        for i in range(m0, m0 + 60):
            j = i - m0
            # ---- 1. buy cycle: cancel resting bids, re-place them on the nord eligible coins closest to the trigger
            if j % cyc == 0 or i == due:
                if nres > 0:
                    for k in range(K):
                        resv[k] = 0.0
                    nres = 0; rsum = 0.0
                if cash > 1e-9:
                    for k in range(K):
                        dist[k] = 1e9
                        if units[k] > 0.0 or i < okm[k] or np.isnan(o1[i, k]):
                            continue
                        if W > 0:
                            if dn[k] == 0:
                                continue
                            r_ = c1[dq[k, dh[k]], k]
                        else:
                            r_ = ref[t, k]
                            if np.isnan(r_):
                                continue
                        if guard > 0.0 and not np.isnan(gref[t, k]) and o1[i, k] < gref[t, k] * (1.0 - guard):
                            continue
                        lim_ = r_ * (1.0 - depth[t, k])
                        limk[k] = lim_
                        d_ = o1[i, k] / lim_ - 1.0
                        if d_ <= near:
                            dist[k] = d_
                    nel = 0
                    for k in range(K):
                        if dist[k] < 1e8:
                            nel += 1
                    if nel <= nord[t]:              # every eligible coin gets a bid: the order cannot matter
                        nsel = 0
                        for k in range(K):
                            if dist[k] < 1e8:
                                order[nsel] = k; nsel += 1
                    elif nord[t] == 1:
                        order[0] = np.argmin(dist); nsel = 1
                    else:
                        order = np.argsort(dist)
                        nsel = nord[t]
                    if nsel > 0:
                        if j == 0:
                            eqm = eq_prev
                        else:
                            eqm = cash
                            for k in range(K):
                                if units[k] > 0.0:
                                    eqm += units[k] * lastm[k]
                        per = min(fmax[t] * eqm, cash / nsel)
                        for q in range(nsel):
                            k = order[q]
                            lim = limk[k]
                            if o1[i, k] <= lim:
                                fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                                if cash - rsum > 1e-9:
                                    a = min(per, cash - rsum)
                                    cash -= a; units[k] = a * (1.0 - fee) / fill
                                    entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = eqm
                                    tpp[k] = fill * (1.0 + tpa[t, k])
                                    act[k] = i + delay if delay > 0 else m0 + 60
                                    pend[k] = 0; lastm[k] = fill
                            else:
                                resv[k] = per; rlim[k] = lim; rsum += per; nres += 1
                                eqb[k] = eqm
                due = -1
            # ---- 2. resting bids that trade eps through their limit this minute (trimmed low)
            if nres > 0:
                for k in range(K):
                    if resv[k] > 0.0:
                        lo = l1[i, k]
                        if wick > 0.0:
                            fl = min(o1[i, k], c1[i, k]) * (1.0 - wick)
                            if fl > lo:
                                lo = fl
                        if lo <= rlim[k] * (1.0 - eps):
                            fill = rlim[k] * (1.0 + pen)
                            a = min(resv[k], cash)
                            rsum -= resv[k]; resv[k] = 0.0; nres -= 1
                            if a > 1e-9:
                                cash -= a; units[k] = a * (1.0 - fee) / fill
                                entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i
                                tpp[k] = fill * (1.0 + tpa[t, k])
                                act[k] = i + delay if delay > 0 else m0 + 60
                                pend[k] = 0; lastm[k] = fill
            # ---- 3. exits
            for k in range(K):
                if units[k] <= 0.0:
                    continue
                px = -1.0; why = 0.0
                if i >= act[k] and not np.isnan(o1[i, k]):
                    if pend[k] == 1:
                        px = o1[i, k] * (1.0 - slip[t, k]); why = 2.0
                    else:
                        tgt = tpp[k]
                        if bem >= 0 and i - fmin[k] >= bem:
                            tb = entry[k] * (1.0 + tpbe)
                            if tb < tgt:
                                tgt = tb; why = 3.0
                        hi = h1[i, k]
                        if wick > 0.0:
                            fh = max(o1[i, k], c1[i, k]) * (1.0 + wick)
                            if fh < hi:
                                hi = fh
                        if tpo == 1 and (i == act[k] or j == 0) and o1[i, k] >= tgt:
                            px = o1[i, k]
                        elif hi >= tgt * (1.0 + eps):
                            px = tgt
                        else:
                            why = 0.0
                if px < 0.0 and not np.isnan(c1[i, k]):
                    if hmm >= 0:
                        if i - fmin[k] >= hmm:
                            px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                    elif j == 59 and ebar[k] < t and t - ebar[k] >= hm[t]:
                        px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - spen) * (1.0 - fee); cash += proceeds
                    ntr += 1
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = i; lg[nl, 8] = fmin[k]; lg[nl, 9] = eqb[k]
                        nl += 1
                    units[k] = 0.0; pend[k] = 0
                    okm[k] = (t + cd) * 60 if cdm < 0 else i + cdm
                    if rearm == 1:
                        due = i + 1
                elif not np.isnan(c1[i, k]):
                    lastm[k] = c1[i, k]
                    if sl > 0.0 and i >= act[k] and c1[i, k] <= entry[k] * (1.0 - sl):
                        pend[k] = 1
            # ---- 4. minute mark (for the intra-hour low) and the W-minute deque update with this minute's close
            v_ = cash
            for k in range(K):
                if units[k] > 0.0:
                    v_ += units[k] * lastm[k]
            if v_ < lo_t:
                lo_t = v_
            if W > 0:
                for k in range(K):
                    x = c1[i, k]
                    if np.isnan(x):
                        continue
                    while dn[k] > 0 and c1[dq[k, (dh[k] + dn[k] - 1) % cap], k] <= x:
                        dn[k] -= 1
                    dq[k, (dh[k] + dn[k]) % cap] = i
                    dn[k] += 1
                    while dn[k] > 0 and dq[k, dh[k]] <= i - W:
                        dh[k] = (dh[k] + 1) % cap; dn[k] -= 1
        # ---- hourly mark (exactly as engine.mkernel)
        v = cash; w = 0.0
        for k in range(K):
            if not np.isnan(c1[m0 + 59, k]):
                lastc[k] = c1[m0 + 59, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0 else 0.0
        eqlo[t] = min(lo_t, v)
    return eq, lg[:nl].copy(), inv, eqlo, ntr


DEFAULTS = dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24, eps=E.EPS, fee=E.FEE, pen=0.0, delay=0,
                cyc=60, rearm=0, cdm=-1, near=1e9, spen=0.0,
                W=0, wick=0.0, tpo=1, hmm=-1, sl=0.0, bem=-1, tpbe=0.0, guard=0.0, maxlog=400000)


def srun(A, hidx, Ph, SLIP, i0=None, i1=None, gref=None, **kw):
    """Run skernel on hours [i0, i1) of the panel (minute rows 60*i0 .. 60*i1). Returns (equity, log, invested, eqlo,
    trades) with hour indices in the log relative to the slice (use hidx[i0:i1])."""
    p = dict(DEFAULTS); p.update(kw)
    bad = set(kw) - set(DEFAULTS)
    assert not bad, f"unknown settings {bad}"
    nh_all = len(hidx)
    i0 = 0 if i0 is None else i0; i1 = nh_all if i1 is None else i1
    nh = i1 - i0
    K = A["c"].shape[1]
    sl = np.ascontiguousarray(SLIP[i0:i1], np.float64) if np.ndim(SLIP) == 2 else np.full((nh, K), float(SLIP))
    o1, h1, l1, c1 = (np.asarray(A[f][60 * i0:60 * i1]) for f in ("o", "h", "l", "c"))
    g = np.full((nh, K), np.nan) if gref is None else np.ascontiguousarray(gref[i0:i1])
    eq, lg, inv, eqlo, ntr = skernel(o1, h1, l1, c1, np.ascontiguousarray(Ph["ref"].values[i0:i1]),
                                     E.full(nh, K, p["depth"]), E.full(nh, 0, p["N"], np.int64), E.full(nh, 0, p["fmax"]),
                                     E.full(nh, K, p["tp"]), E.full(nh, 0, p["hm"], np.int64), int(p["cd"]),
                                     float(p["eps"]), float(p["fee"]), sl, float(p["pen"]), int(p["delay"]),
                                     int(p["cyc"]), int(p["rearm"]), int(p["cdm"]), float(p["near"]), float(p["spen"]),
                                     int(p["W"]), float(p["wick"]), int(p["tpo"]), int(p["hmm"]), float(p["sl"]),
                                     int(p["bem"]), float(p["tpbe"]), g, float(p["guard"]), int(p["maxlog"]))
    hx = hidx[i0:i1]
    return pd.Series(eq, hx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, hx), pd.Series(eqlo, hx), int(ntr)


# ------------------------------------------------------------------ scalp presets
def scalp(W, depth, tp, hmm, N=1, fmax=1.0, cdm=0, sl=0.0, bem=-1, tpbe=0.0, guard=0.0, wick=0.005, pen=0.0, spen=0.0,
          maxlog=400000):
    """Minute-cycle scalper: bids re-placed every minute, take-profit from the next minute at its limit only."""
    return dict(W=W, depth=depth, tp=tp, hmm=hmm, N=N, fmax=fmax, cdm=cdm, sl=sl, bem=bem, tpbe=tpbe, guard=guard,
                wick=wick, pen=pen, spen=spen, cyc=1, delay=1, tpo=0, hm=10 ** 6, cd=0, maxlog=maxlog)


# ------------------------------------------------------------------ reporting
def pm(mult, months):
    return mult ** (1.0 / months) - 1.0 if mult > 0 else -1.0


def period_stats(e, lg, eqlo, hx, a, b, ntr_total=None):
    """%/month compounded, multiple, worst drop on hourly marks (engine convention) and on 1-minute marks, win rate,
    trades/month, average trade, worst single trade as % of the equity that sized it (trades bought in the period)."""
    sel = (e.index >= a) & ((e.index < b) if b is not None else True)
    ee = e[sel]
    if len(ee) < 2:
        return None
    m = metrics(e, start=a, end=b)
    months = (ee.index[-1] - ee.index[0]).total_seconds() / 86400 / MD
    base = ee.iloc[0]
    en = ee / base
    lo = eqlo[sel] / base
    pk = en.cummax()
    dd_min = float((lo / pk.shift(1).fillna(1.0).clip(lower=1.0) - 1).min())
    out = dict(pm=pm(m["mult"], months), mult=m["mult"], dd=m["maxdd"], dd_min=min(dd_min, m["maxdd"]),
               worst_m=m["worst_m"], pos_m=m["pos_m"], months=months)
    if lg is not None and len(lg):
        tin = hx[lg.eb.astype(int).values]
        g = lg[(tin >= a) & ((tin < b) if b is not None else True)]
        loss = g.ret * g.amt / g.eqb
        out.update(trades=len(g), tpm=len(g) / months, win=float((g.ret > 0).mean()) if len(g) else np.nan,
                   avg=float(g.ret.mean()) if len(g) else np.nan,
                   worst_trade=float(loss.min()) if len(g) else np.nan,
                   worst_ret=float(g.ret.min()) if len(g) else np.nan,
                   hold_med=float((g.xm - g.fm).median()) if len(g) else np.nan,
                   n_ts=int((g.why == 1).sum()), n_sl=int((g.why == 2).sum()), n_be=int((g.why == 3).sum()))
    else:
        out.update(trades=0, tpm=0.0, win=np.nan, avg=np.nan, worst_trade=np.nan, worst_ret=np.nan, hold_med=np.nan,
                   n_ts=0, n_sl=0, n_be=0)
    return out


PERIODS = {"d": (T21, T24), "t": (T24, None), "s": (T25, None), "h": (T18, pd.Timestamp("2021-01-01"))}
PLAB = {"h": "2018-20", "d": "2021-23", "t": "2024-26", "s": "since 2025"}


def line(r, lab=""):
    if r is None:
        return f"{lab:10s} (no data)"
    return (f"{lab:10s} win {r['win']*100:5.1f}% | {r['pm']*100:+6.2f}%/mo x{r['mult']:9.2f} | DD {r['dd']*100:4.0f}% "
            f"(1-min {r['dd_min']*100:4.0f}%) | worst trade {r['worst_trade']*100:+6.1f}% of acct | "
            f"{r['tpm']:6.1f} trades/mo | avg {r['avg']*100:+.3f}% | worst mo {r['worst_m']*100:+4.0f}% | "
            f"TS {r['n_ts']} SL {r['n_sl']} BE {r['n_be']}")
