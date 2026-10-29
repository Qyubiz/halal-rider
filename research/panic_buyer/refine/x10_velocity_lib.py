"""Lever "velocity" (x10 study): make each dollar do more round trips. 1-minute kernel `vkernel`, copied from
engine.mkernel and rewritten as one chronological minute loop so that cash freed by a sale can be reused inside the hour.

With cyc=60, rearm=0, cdm=-1, xmode=0, near=1e9, spen=0 it is engine.mkernel (proved in x10_velocity_check.py: same
trade log, equity equal to float rounding). New switches:
  cyc    buy cycle every `cyc` minutes (60 = live, hourly). At each cycle every resting bid is cancelled and re-placed
         on the `nord` eligible coins closest to their trigger, sized min(fmax * equity, free cash / nsel). A coin
         already under its trigger at the cycle minute's open is bought at that open (+ slippage), as the hourly bot does.
  rearm  1: also run a buy cycle in the minute after any sale (re-arm as soon as cash is freed).
  cdm    cooldown in minutes after a sale (-1: the engine's hourly cooldown `cd`, eligible from hour sale_hour + cd).
  near   only coins at most `near` above their trigger get a bid (1e9 = off: the nord closest, however far). With
         fmax 1 this puts all free cash on a lone dip and splits it only when several coins dip together.
  xmode  0: fixed take-profit at entry*(1+tp) (engine). 1: trailing exit on 1-minute closes: armed when a close is at
         least entry*(1+arm); then sell (market, next minute's open, - slippage) at the first close <= peak*(1-trail)
         (trail 0: the first close below the highest close since arming). cap>0 adds a resting limit sell at entry*(1+cap).
  delay  minutes after the fill until the exit is active (0: the next hour, like the live bot).
  spen   every SELL fill `spen` worse (pen already makes buys worse), for the "every fill 0.2% worse" stress.
  look2, depth2, tpf  (round 2) a second "flash dip" trigger: also bid at max-close-of-the-last-look2-hours*(1-depth2)
         when that is above the 24h trigger; such buys take profit at tpf (0 / 0.0: off).
  redip, rewin  (round 2) after a profitable exit, the coin may be re-bought at exit*(1-redip) for rewin minutes, when
         that is above its normal trigger (needs a cooldown shorter than rewin; 0: off).
  cdl    (round 3) cooldown in minutes after a LOSING exit (-1: off, losing exits use the normal cooldown).
  slip   (hour, coin) market slippage: 0.02% in the trailing-30-day top 50 by quote volume, else 0.1% (slip_panel).
Only information known at decision time is used: the reference is the highest hourly close of the 24 closed hours
before the current hour; a cycle sees the cycle minute's open; trailing decisions see closed minutes and fill a minute later.
"""
import glob
import os
import sys
import time

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
from lab import metrics, month_returns  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out"))
CACHE = os.path.join(OUT, "cache")
SYMS = E.H19 + list(E.DEAD)            # 22 columns: the 19 live coins first, then LUNA, FTT, SRM
K19 = 19
LOGC = ["eb", "xb", "k", "why", "ret", "amt", "entry", "xm", "fm", "eqb"]   # why 0 tp/cap, 1 time stop, 2 trailing
T21, T24, T25 = pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")
MD = 365.25 / 12


# ------------------------------------------------------------------ data cache (built once, then memory-mapped)
def build_cache():
    """22-coin minute panel (engine.minute_panel), collapse coins blanked where Binance had no trading (as hr_stress),
    stored as float32 (the source files are float32, so the values are exact) plus the hourly panel and slippage."""
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    idxm, A, hidx, Ph = E.minute_panel(SYMS, cuts=E.DEAD)
    for s_ in E.DEAD:
        k_ = SYMS.index(s_)
        tr = pd.to_datetime(np.load(os.path.join(E.DATA, "1m", f"{s_}.npz"))["t"], unit="s")
        tr = tr[tr <= pd.Timestamp(E.DEAD[s_]) + pd.Timedelta(minutes=59)]
        gaps = [(tr[i], tr[i + 1]) for i in np.flatnonzero(np.diff(tr.values) > np.timedelta64(6, "h"))]
        for g0, g1 in gaps + [(tr[-1], idxm[-1] + pd.Timedelta(minutes=1))]:
            i0_, i1_ = idxm.searchsorted(g0, side="right"), idxm.searchsorted(g1, side="left")
            for f in A:
                A[f][i0_:i1_, k_] = np.nan
            print(f"{s_}: no trading {g0} .. {g1} -> minutes {idxm[i0_]} .. {idxm[i1_-1]} blanked")
    for f in ("o", "h", "l", "c"):
        a32 = A[f].astype(np.float32)
        ok = np.isnan(A[f]) | (a32.astype(np.float64) == A[f])
        assert ok.all(), f"{f}: float32 cache would change values"
        np.save(os.path.join(CACHE, f"m_{f}.npy"), a32)
        del a32
    nh = len(hidx)
    H = {"o": A["o"][::60], "c": A["c"][59::60], "h": np.fmax.reduce(A["h"].reshape(nh, 60, -1), axis=1),
         "l": np.fmin.reduce(A["l"].reshape(nh, 60, -1), axis=1)}
    for f, v in H.items():
        np.save(os.path.join(CACHE, f"h_{f}.npy"), v)
    np.save(os.path.join(CACHE, "hidx.npy"), hidx.values.astype("datetime64[s]").astype(np.int64))
    del A
    np.save(os.path.join(CACHE, "slip.npy"), slip_panel(pd.DatetimeIndex(hidx)))
    print(f"cache built in {time.time()-t0:.0f}s -> {CACHE}")


def load():
    """(A, hidx, Ph, SLIP): A = 1-minute float32 arrays (memory-mapped, 22 coins), Ph = hourly DataFrames built from
    the same minutes (o h l c cf ref), SLIP = (hours, 22) market slippage."""
    if not os.path.exists(os.path.join(CACHE, "slip.npy")):
        build_cache()
    A = {f: np.load(os.path.join(CACHE, f"m_{f}.npy"), mmap_mode="r") for f in ("o", "h", "l", "c")}
    hidx = pd.DatetimeIndex(pd.to_datetime(np.load(os.path.join(CACHE, "hidx.npy")), unit="s"))
    Ph = {f: pd.DataFrame(np.load(os.path.join(CACHE, f"h_{f}.npy")), hidx, SYMS) for f in ("o", "h", "l", "c")}
    Ph["cf"] = Ph["c"].ffill()
    Ph["ref"] = Ph["c"].rolling(24, min_periods=1).max().shift(1)
    return A, hidx, Ph, np.load(os.path.join(CACHE, "slip.npy"))


def slip_panel(hidx):
    """0.02% when the coin is in the top 50 by trailing 30-day quote volume (known at the hour's start) among the 322
    screened coins plus FTT (RNDR joined to RENDER), 0.1% otherwise (also when it has no volume history yet).
    The pool holds only screened coins, so true Binance ranks are lower: this understates the 0.1% hours."""
    qv = {}
    for p in glob.glob(os.path.join(E.DATA, "univ", "*.csv")):
        qv[os.path.basename(p)[:-4]] = pd.read_csv(p, index_col=0, parse_dates=True, usecols=["t", "qv"]).qv
    for s in SYMS + ["RNDRUSDT"]:
        if s not in qv:
            d = E.read_hourly("RNDRUSDT" if s == "RNDRUSDT" else s)
            qv[s] = d.qv
    r = qv.pop("RNDRUSDT")
    qv["RENDERUSDT"] = pd.concat([r, qv["RENDERUSDT"]]).groupby(level=0).last()
    Q = pd.DataFrame(qv).sort_index()
    Q = Q[Q.index >= hidx[0] - pd.Timedelta(days=40)]
    rk = Q.fillna(0).rolling(24 * 30, min_periods=24).sum().shift(1).rank(axis=1, ascending=False)
    rk = rk.reindex(hidx)[SYMS]
    S = np.where(rk.values <= 50, E.SLIP_MAJOR, 0.001)
    return np.ascontiguousarray(S, dtype=np.float64)


# ------------------------------------------------------------------ kernel
@njit(cache=False)
def vkernel(o1, h1, l1, c1, ref, depth, nord, fmax, tpa, hm, cd, eps, fee, slip, pen, delay,
            cyc, rearm, cdm, near, xmode, arm, trail, cap, spen, ref2, depth2, tpf, redip, rewin, cdl):
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K); capp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okm = np.zeros(K, np.int64); act = np.zeros(K, np.int64)
    fmin = np.zeros(K, np.int64); eqb = np.zeros(K); lastm = np.zeros(K)
    armed = np.zeros(K, np.int64); peak = np.zeros(K); pend = np.zeros(K, np.int64)
    resv = np.zeros(K); rlim = np.zeros(K); nres = 0; rsum = 0.0
    rex = np.zeros(K); rexu = np.full(K, -1, np.int64); rsrc = np.zeros(K, np.int64)
    limk = np.zeros(K); srck = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(nh); inv = np.empty(nh); eq_prev = 1.0
    maxlog = 3000000; lg = np.zeros((maxlog, 10)); nl = 0
    due = -1
    for t in range(nh):
        m0 = 60 * t
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
                        if units[k] > 0.0 or i < okm[k] or np.isnan(o1[i, k]) or np.isnan(ref[t, k]):
                            continue
                        lim_ = ref[t, k] * (1.0 - depth[t, k]); s_ = 0
                        if depth2 > 0.0 and not np.isnan(ref2[t, k]):
                            l2_ = ref2[t, k] * (1.0 - depth2)
                            if l2_ > lim_:
                                lim_ = l2_; s_ = 1
                        if redip > 0.0 and i < rexu[k] and rex[k] > lim_:
                            lim_ = rex[k]; s_ = 2
                        limk[k] = lim_; srck[k] = s_
                        d_ = o1[i, k] / lim_ - 1.0
                        if d_ <= near:
                            dist[k] = d_
                    order = np.argsort(dist)
                    nsel = 0
                    for q in range(K):
                        if dist[order[q]] < 1e8 and nsel < nord[t]:
                            nsel += 1
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
                                    tpp[k] = fill * (1.0 + (tpf if srck[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                                    act[k] = i + delay if delay > 0 else m0 + 60
                                    armed[k] = 0; pend[k] = 0; lastm[k] = fill
                            else:
                                resv[k] = per; rlim[k] = lim; rsum += per; nres += 1; rsrc[k] = srck[k]
                                eqb[k] = eqm
                due = -1
            # ---- 2. resting bids that trade eps through their limit this minute
            if nres > 0:
                for k in range(K):
                    if resv[k] > 0.0 and l1[i, k] <= rlim[k] * (1.0 - eps):
                        fill = rlim[k] * (1.0 + pen)
                        a = min(resv[k], cash)
                        rsum -= resv[k]; resv[k] = 0.0; nres -= 1
                        if a > 1e-9:
                            cash -= a; units[k] = a * (1.0 - fee) / fill
                            entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i
                            tpp[k] = fill * (1.0 + (tpf if rsrc[k] == 1 else tpa[t, k])); capp[k] = fill * (1.0 + cap)
                            act[k] = i + delay if delay > 0 else m0 + 60
                            armed[k] = 0; pend[k] = 0; lastm[k] = fill
            # ---- 3. exits
            for k in range(K):
                if units[k] <= 0.0:
                    continue
                px = -1.0; why = 0.0
                if i >= act[k] and not np.isnan(o1[i, k]):
                    if xmode == 0:
                        if (i == act[k] or j == 0) and o1[i, k] >= tpp[k]:
                            px = o1[i, k]
                        elif h1[i, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    else:
                        if pend[k] == 1:
                            px = o1[i, k] * (1.0 - slip[t, k]); why = 2.0
                        elif cap > 0.0 and (i == act[k] or j == 0) and o1[i, k] >= capp[k]:
                            px = o1[i, k]
                        elif cap > 0.0 and h1[i, k] >= capp[k] * (1.0 + eps):
                            px = capp[k]
                        else:
                            cc = c1[i, k]
                            if armed[k] == 0:
                                if cc >= entry[k] * (1.0 + arm):
                                    armed[k] = 1; peak[k] = cc
                            elif cc > peak[k]:
                                peak[k] = cc
                            elif (trail > 0.0 and cc <= peak[k] * (1.0 - trail)) or (trail <= 0.0 and cc < peak[k]):
                                pend[k] = 1
                if px < 0.0 and j == 59 and ebar[k] < t and t - ebar[k] >= hm[t] and not np.isnan(c1[i, k]):
                    px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - spen) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = i; lg[nl, 8] = fmin[k]; lg[nl, 9] = eqb[k]
                        nl += 1
                    units[k] = 0.0; pend[k] = 0; armed[k] = 0
                    okm[k] = (t + cd) * 60 if cdm < 0 else i + cdm
                    if cdl >= 0 and proceeds < amt0[k]:
                        okm[k] = i + cdl
                    if redip > 0.0 and proceeds > amt0[k]:
                        rex[k] = px * (1.0 - redip); rexu[k] = i + rewin
                    if rearm == 1:
                        due = i + 1
                elif not np.isnan(c1[i, k]):
                    lastm[k] = c1[i, k]
        # ---- hourly mark (exactly as engine.mkernel)
        v = cash; w = 0.0
        for k in range(K):
            if not np.isnan(c1[m0 + 59, k]):
                lastc[k] = c1[m0 + 59, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0 else 0.0
    return eq, lg[:nl].copy(), inv


def ref2_(Ph, look, K):
    """Second, shorter trigger reference: the highest hourly close of the last `look` closed hours (0: off).
    Computed on every call (an earlier id()-keyed cache could hand a freed slice's array to a new slice)."""
    c = Ph["c"].iloc[:, :K]
    return np.ascontiguousarray((c.rolling(look, min_periods=1).max().shift(1) if look > 0 else c * np.nan).values)


DEFAULTS = dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24, eps=E.EPS, fee=E.FEE, pen=0.0, delay=0,
                cyc=60, rearm=0, cdm=-1, near=1e9, xmode=0, arm=0.015, trail=0.0, cap=0.0, spen=0.0,
                look2=0, depth2=0.0, tpf=0.015, redip=0.0, rewin=0, cdl=-1)


def vrun(A, hidx, Ph, SLIP, K=None, **kw):
    """Run vkernel on the first K columns (19 = live coins, 22 = + LUNA/FTT/SRM). Returns (equity, log, invested)."""
    p = dict(DEFAULTS); p.update(kw)
    bad = set(kw) - set(DEFAULTS)
    assert not bad, f"unknown settings {bad}"
    K = K or A["c"].shape[1]
    nh = len(hidx)
    sl = SLIP[:, :K] if np.ndim(SLIP) == 2 else np.full((nh, K), float(SLIP))
    o1, h1, l1, c1 = (A[f][:, :K] if A[f].shape[1] != K else A[f] for f in ("o", "h", "l", "c"))
    eq, lg, inv = vkernel(np.asarray(o1), np.asarray(h1), np.asarray(l1), np.asarray(c1), Ph["ref"].values[:, :K],
                          E.full(nh, K, p["depth"]), E.full(nh, 0, p["N"], np.int64), E.full(nh, 0, p["fmax"]),
                          E.full(nh, K, p["tp"]), E.full(nh, 0, p["hm"], np.int64), int(p["cd"]), float(p["eps"]),
                          float(p["fee"]), np.ascontiguousarray(sl, np.float64), float(p["pen"]), int(p["delay"]),
                          int(p["cyc"]), int(p["rearm"]), int(p["cdm"]), float(p["near"]), int(p["xmode"]),
                          float(p["arm"]), float(p["trail"]), float(p["cap"]), float(p["spen"]),
                          ref2_(Ph, p["look2"], K), float(p["depth2"]), float(p["tpf"]), float(p["redip"]), int(p["rewin"]), int(p["cdl"]))
    return pd.Series(eq, hidx), pd.DataFrame(lg, columns=LOGC), pd.Series(inv, hidx)


# ------------------------------------------------------------------ reporting
PERIODS = {"t": (T24, None), "s": (T25, None), "d": (T21, T24)}     # 2024-26, since 2025, 2021-23


def pm(mult, months):
    return mult ** (1.0 / months) - 1.0 if mult > 0 else -1.0


def report(e, lg=None, hidx=None, syms=SYMS):
    """%/month, multiple, worst drop and worst month per period (engine/lab conventions: the period's equity is
    divided by its first hourly mark); months from the period's span. Worst single trade = trade P&L as % of the
    equity used to size it (bought in that period)."""
    out = {}
    for p, (a, b) in PERIODS.items():
        ee = e[(e.index >= a) & ((e.index < b) if b is not None else True)]
        if len(ee) < 2:
            continue
        m = metrics(e, start=a, end=b)
        months = (ee.index[-1] - ee.index[0]).total_seconds() / 86400 / MD
        out.update({f"{p}_pm": pm(m["mult"], months), f"{p}_mult": m["mult"], f"{p}_dd": m["maxdd"],
                    f"{p}_worst_m": m["worst_m"], f"{p}_pos_m": m["pos_m"]})
        if lg is not None and len(lg):
            tin = hidx[lg.eb.astype(int).values]
            g = lg[(tin >= a) & ((tin < b) if b is not None else True)]
            loss = g.ret * g.amt / g.eqb
            out[f"{p}_trades"] = len(g)
            out[f"{p}_worst_trade"] = float(loss.min()) if len(g) else np.nan
            out[f"{p}_worst_trade_coin"] = syms[int(g.k.values[int(np.argmin(loss.values))])] if len(g) else ""
            out[f"{p}_win"] = float((g.ret > 0).mean()) if len(g) else np.nan
            out[f"{p}_avg"] = float(g.ret.mean()) if len(g) else np.nan
    out["y2022"] = E.years(e).get(2022, np.nan)
    return out


def fmt(r, pre=""):
    s = []
    for p, lab in (("t", "24-26"), ("s", "25-26"), ("d", "21-23")):
        if f"{p}_pm" in r and not pd.isna(r[f"{p}_pm"]):
            s.append(f"{lab} {r[f'{p}_pm']*100:+5.2f}%/mo DD {r[f'{p}_dd']*100:4.0f}%")
    return pre + " | ".join(s)


def fmt_d(r):
    return (f"21-23 {r['d_pm']*100:+6.2f}%/mo x{r['d_mult']:8.2f} DD {r['d_dd']*100:4.0f}% 2022 {r['y2022']*100:+5.0f}% "
            f"worst-mo {r['d_worst_m']*100:+4.0f}% trades {r.get('d_trades', -1):5.0f} worst trade {r.get('d_worst_trade', np.nan)*100:+5.1f}%")
