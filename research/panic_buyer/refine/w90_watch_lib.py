"""w90 task "ten-watch": 10% mode (1 coin x 100%, 14% dip, +5%, 48h, 24h cooldown, 35% crash guard) run as a
minute-level WATCH bot on 1-minute candles, with realistic fills.

Mechanics of the watch bot (mode=1, the new mechanic):
  * Every minute the bot looks at every eligible coin (no position, no pending buy, cooldown over, not excluded by the
    crash guard). The trigger is the live one: highest hourly close of the 24 closed hours before the current hour
    x (1 - depth). The crash guard skips a coin whose last hourly close is more than `guard` under its highest close of
    the previous 168 hours (x10_critic_lib.guard_ref, known at the hour's start).
  * det=0 ("stream"): a coin is triggered when the minute's low trades through the trigger (deps through, default 0).
    det=1 ("poll"): a coin is triggered when the minute's CLOSE is at or under the trigger (a bot that polls the last
    price once a minute).
  * A triggered coin is bought with a MARKET order at the NEXT minute's open x (1 + mslip) (default 0.05%), sized
    min(fmax x equity, cash), as long as fewer than nord positions (incl. pending buys) are open. When several coins
    trigger in the same minute they are served deepest-under-trigger (at that minute's close) first (ordm=0) or in
    coin order (ordm=1). If the next minute has no trade (NaN) the order is dropped.
  * The take-profit limit sell entry x (1 + tp) is placed at once; it can fill from the minute `tpd` after the fill
    minute (tpd=1: the next minute) when a minute's high trades eps (0.1%) through it, at the limit price
    (gap=0: never better than the limit; gap=1: an open already above it fills at that open, as engine.mkernel).
  * Time stop: tsm=0 the engine rule (sell at the close of the first hour >= hm hours after the fill hour, x (1-mslip));
    tsm=1 at the open of the minute hm*60 after the fill minute.
  * Cooldown: the coin can trigger again from the minute cdmin minutes after its sale (cdmin = cd*60 by default).
  * Wicks: wick(A, w) trims every minute's high to max(open, close) x (1 + w) and low to min(open, close) x (1 - w)
    (x10_critic_wicks.py method); the runner applies it with w = 1% by default.
  * Stress: pen (every buy 0.x% worse), spen (every sell 0.x% worse), K=22 (LUNA/FTT/SRM added), cdmin + 60.
Equity is marked hourly on the last minute's close (engine convention); eqlo also records the lowest minute-close mark
inside each hour, so the minute-level worst drop can be reported.

mode=0 is x10_critic_lib.xkernel's minute watch path (watch=1, mlat=0, xmode=0, no flash/redip/loss cooldown), copied
line for line: w90_watch_check.py proves it gives identical trade logs and equity. Spot only, fully funded orders,
no leverage, no shorting.
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_velocity_lib as V  # noqa: E402
from lab import metrics  # noqa: E402

VEL_CACHE = ("/tmp/claude-0/-home-user-halal-rider/ec562e35-ea9e-5a74-ab92-4ff22c5eeb77/scratchpad/pb_out_x10/velocity/"
             "cache")
V.CACHE = os.environ.get("PB_MIN_CACHE", VEL_CACHE)   # the velocity agent's float32 minute cache (read-only)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_w90_watch"))
os.makedirs(OUT, exist_ok=True)
SYMS = V.SYMS
MSLIP = 0.0005


# ------------------------------------------------------------------ data
def load():
    """(A, hidx, Ph, SLIP) from the shared minute cache (22 coins: the 19 live coins, then LUNA, FTT, SRM)."""
    return V.load()


def wick(A, K, w):
    """Minute candles of the first K coins with every wick trimmed to w beyond the candle body (w None: untouched)."""
    o = np.ascontiguousarray(A["o"][:, :K]); c = np.ascontiguousarray(A["c"][:, :K])
    if w is None:
        return {"o": o, "c": c, "h": np.ascontiguousarray(A["h"][:, :K]), "l": np.ascontiguousarray(A["l"][:, :K])}
    dt = o.dtype                      # float32 cache: exactly the x10_critic_wicks.py arithmetic
    hb = np.fmax(o, c); lb = np.fmin(o, c)
    h = np.fmin(A["h"][:, :K], (hb * dt.type(1 + w)).astype(dt))
    del hb
    lo = np.fmax(A["l"][:, :K], (lb * dt.type(1 - w)).astype(dt))
    del lb
    return {"o": o, "c": c, "h": np.ascontiguousarray(h), "l": np.ascontiguousarray(lo)}


def guard_ref(Ph, g, K):
    """Trigger reference with the crash guard (NaN = coin skipped that hour). g <= 0 or >= 1: no guard."""
    r = np.ascontiguousarray(Ph["ref"].values[:, :K].astype(np.float64))
    if g is None or g <= 0 or g >= 1:
        return r
    c = Ph["cf"].iloc[:, :K]
    hi = c.rolling(168, min_periods=1).max()
    ok = ((c / hi - 1) >= -g).shift(1, fill_value=True).values
    r = r.copy()
    r[~ok] = np.nan
    return np.ascontiguousarray(r)


# ------------------------------------------------------------------ kernel
@njit(cache=False)
def wkernel(o1, h1, l1, c1, ref, depth, nord, fmax, tpa, hm, cd, eps, fee, slip, pen, spen,
            mode, cyc, rearm, delay, wslip, det, deps, tpd, gap, cdmin, ordm, tsm):
    nm, K = c1.shape
    nh = nm // 60
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); tpp = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okm = np.zeros(K, np.int64); act = np.zeros(K, np.int64)
    fmin = np.zeros(K, np.int64); eqb = np.zeros(K); lastm = np.zeros(K)
    wl = np.zeros(K); wok = np.zeros(K, np.int64); word = np.arange(K); wper = 0.0
    pq = np.zeros(K); pqeq = np.zeros(K)                       # mode 1: market buys decided, filled next minute
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(nh); eqlo = np.empty(nh); inv = np.empty(nh); eq_prev = 1.0
    maxlog = 3000000; lg = np.zeros((maxlog, 11)); nl = 0
    due = -1
    for t in range(nh):
        m0 = 60 * t
        lo_h = 1e300
        for i in range(m0, m0 + 60):
            j = i - m0
            if mode == 0:
                # ================= x10_critic_lib.xkernel, watch=1, mlat=0, xmode=0 (copied)
                if j % cyc == 0 or i == due:
                    for k in range(K):
                        wok[k] = 0
                    if cash > 1e-9:
                        for k in range(K):
                            dist[k] = 1e9
                            if units[k] > 0.0 or i < okm[k] or np.isnan(o1[i, k]) or np.isnan(ref[t, k]):
                                continue
                            lim_ = ref[t, k] * (1.0 - depth[t, k])
                            wl[k] = lim_
                            dist[k] = o1[i, k] / lim_ - 1.0
                        order = np.argsort(dist)
                        if j == 0:
                            eqm = eq_prev
                        else:
                            eqm = cash
                            for k in range(K):
                                if units[k] > 0.0:
                                    eqm += units[k] * lastm[k]
                        npos = 0
                        for k in range(K):
                            if units[k] > 0.0:
                                npos += 1
                        wper = fmax[t] * eqm
                        for q in range(K):
                            k = order[q]
                            word[q] = k
                            if dist[k] >= 1e8:
                                continue
                            lim = wl[k]
                            if o1[i, k] <= lim:
                                if npos < nord[t] and cash > 1e-9:
                                    a = min(wper, cash)
                                    fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                                    cash -= a; units[k] = a * (1.0 - fee) / fill
                                    entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = eqm
                                    tpp[k] = fill * (1.0 + tpa[t, k])
                                    act[k] = i + delay if delay > 0 else m0 + 60
                                    lastm[k] = fill; npos += 1
                            else:
                                wok[k] = 1
                    due = -1
                npos = 0
                for k in range(K):
                    if units[k] > 0.0:
                        npos += 1
                for q in range(K):
                    k = word[q]
                    if wok[k] == 0 or units[k] > 0.0:
                        continue
                    if npos >= nord[t] or cash <= 1e-9:
                        break
                    if i >= okm[k] and l1[i, k] <= wl[k] * (1.0 - eps):
                        fill = wl[k] * (1.0 + slip[t, k] + wslip) * (1.0 + pen)
                        a = min(wper, cash)
                        cash -= a; units[k] = a * (1.0 - fee) / fill
                        entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = wper / fmax[t]
                        tpp[k] = fill * (1.0 + tpa[t, k])
                        act[k] = i + delay if delay > 0 else m0 + 60
                        lastm[k] = fill; npos += 1; wok[k] = 0
            else:
                # ================= mode 1: market buys decided at the previous minute's close fill at this open
                for k in range(K):
                    if pq[k] > 0.0:
                        a = pq[k]; pq[k] = 0.0
                        if np.isnan(o1[i, k]):
                            cash += a                      # no trade this minute: the order is dropped
                            continue
                        fill = o1[i, k] * (1.0 + slip[t, k]) * (1.0 + pen)
                        units[k] = a * (1.0 - fee) / fill
                        entry[k] = fill; amt0[k] = a; ebar[k] = t; fmin[k] = i; eqb[k] = pqeq[k]
                        tpp[k] = fill * (1.0 + tpa[t, k])
                        act[k] = i + tpd
                        lastm[k] = fill
            # ---- exits
            for k in range(K):
                if units[k] <= 0.0:
                    continue
                px = -1.0; why = 0.0
                if i >= act[k] and not np.isnan(o1[i, k]):
                    if mode == 0:
                        if (i == act[k] or j == 0) and o1[i, k] >= tpp[k]:
                            px = o1[i, k]
                        elif h1[i, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                    else:
                        if gap == 1 and (i == act[k] or j == 0) and o1[i, k] >= tpp[k]:
                            px = o1[i, k]
                        elif h1[i, k] >= tpp[k] * (1.0 + eps):
                            px = tpp[k]
                if px < 0.0:
                    if mode == 1 and tsm == 1:
                        if i - fmin[k] >= hm[t] * 60 and not np.isnan(o1[i, k]):
                            px = o1[i, k] * (1.0 - slip[t, k]); why = 1.0
                    elif j == 59 and ebar[k] < t and t - ebar[k] >= hm[t] and not np.isnan(c1[i, k]):
                        px = c1[i, k] * (1.0 - slip[t, k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - spen) * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = entry[k]
                        lg[nl, 7] = i; lg[nl, 8] = fmin[k]; lg[nl, 9] = eqb[k]; lg[nl, 10] = px
                        nl += 1
                    units[k] = 0.0
                    if mode == 0:
                        okm[k] = (t + cd) * 60
                        if rearm == 1:
                            due = i + 1
                    else:
                        okm[k] = (t + cd) * 60 if cdmin < 0 else i + 1 + cdmin
                elif not np.isnan(c1[i, k]):
                    lastm[k] = c1[i, k]
            # ---- mode 1: watch every coin at this minute's close; market buy at the next minute's open
            if mode == 1 and cash > 1e-9 and i + 1 < nm:
                npos = 0
                for k in range(K):
                    if units[k] > 0.0 or pq[k] > 0.0:
                        npos += 1
                if npos < nord[t]:
                    nc = 0
                    for k in range(K):
                        dist[k] = 1e9
                        if units[k] > 0.0 or pq[k] > 0.0 or i + 1 < okm[k] or np.isnan(ref[t, k]) or np.isnan(c1[i, k]):
                            continue
                        lim = ref[t, k] * (1.0 - depth[t, k])
                        if det == 0:
                            hit = l1[i, k] <= lim * (1.0 - deps)
                        else:
                            hit = c1[i, k] <= lim
                        if hit:
                            dist[k] = c1[i, k] / lim - 1.0 if ordm == 0 else float(k)
                            nc += 1
                    if nc > 0:
                        order = np.argsort(dist)
                        eqm = cash
                        for k in range(K):
                            if units[k] > 0.0:
                                eqm += units[k] * lastm[k]
                            eqm += pq[k]
                        for q in range(K):
                            k = order[q]
                            if dist[k] >= 1e8 or npos >= nord[t] or cash <= 1e-9:
                                break
                            a = min(fmax[t] * eqm, cash)
                            cash -= a; pq[k] = a; pqeq[k] = eqm; npos += 1
            # ---- lowest minute-close mark inside the hour
            v = cash
            for k in range(K):
                v += pq[k]
                if units[k] > 0.0:
                    if not np.isnan(c1[i, k]):
                        v += units[k] * c1[i, k]
                    else:
                        v += units[k] * lastm[k]
            if v < lo_h:
                lo_h = v
        # ---- hourly mark (as engine.mkernel)
        v = cash; w = 0.0
        for k in range(K):
            if not np.isnan(c1[m0 + 59, k]):
                lastc[k] = c1[m0 + 59, k]
            if units[k] > 0.0:
                v += units[k] * lastc[k]; w += units[k] * lastc[k]
            v += pq[k]
        eq[t] = v; eq_prev = v; inv[t] = w / v if v > 0 else 0.0
        eqlo[t] = min(lo_h, v)
    opn = np.zeros((K, 3))                                     # positions still open at the end: units, entry, fill minute
    for k in range(K):
        opn[k, 0] = units[k]; opn[k, 1] = entry[k]; opn[k, 2] = fmin[k]
    return eq, lg[:nl].copy(), inv, eqlo, opn


WDEF = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, eps=E.EPS, fee=E.FEE, pen=0.0, spen=0.0,
            mode=1, cyc=60, rearm=0, delay=0, wslip=0.0, det=0, deps=0.0, tpd=1, gap=0, cdmin=None, ordm=0, tsm=0,
            mslip=MSLIP, guard=0.35, wick=0.01, cdx=0)
LOGW = V.LOGC + ["px"]
LAST_OPEN = []          # after wrun: [(coin, entry, fill hour)] of positions still open at the end of the data


def wrun(A, hidx, Ph, SLIP=None, K=19, B=None, ref=None, **kw):
    """Run wkernel on the first K coins (19 live, 22 = + LUNA/FTT/SRM). Returns (equity, log, invested, eq_low).
    B: pre-built (trimmed) minute arrays for these K coins (else built from A with `wick`). ref: precomputed trigger
    reference (else guard_ref(Ph, guard, K)). SLIP: (hours, coins) market slippage; None = flat mslip.
    cdx: extra cooldown minutes (stress: 60 = the bot noticing the sale one hour later)."""
    p = dict(WDEF); p.update(kw)
    bad = set(kw) - set(WDEF)
    assert not bad, f"unknown settings {bad}"
    nh = len(hidx)
    if B is None:
        B = wick(A, K, p["wick"])
    if ref is None:
        ref = guard_ref(Ph, p["guard"], K)
    sl = np.full((nh, K), float(p["mslip"])) if SLIP is None else np.ascontiguousarray(SLIP[:, :K], np.float64)
    cdmin = int(p["cd"]) * 60 if p["cdmin"] is None else int(p["cdmin"])
    cdmin += int(p["cdx"])
    global LAST_OPEN
    eq, lg, inv, eqlo, opn = wkernel(B["o"], B["h"], B["l"], B["c"], np.ascontiguousarray(ref[:, :K], np.float64),
                                E.full(nh, K, p["depth"]), E.full(nh, 0, p["N"], np.int64), E.full(nh, 0, p["fmax"]),
                                E.full(nh, K, p["tp"]), E.full(nh, 0, p["hm"], np.int64), int(p["cd"]), float(p["eps"]),
                                float(p["fee"]), sl, float(p["pen"]), float(p["spen"]), int(p["mode"]), int(p["cyc"]),
                                int(p["rearm"]), int(p["delay"]), float(p["wslip"]), int(p["det"]), float(p["deps"]),
                                int(p["tpd"]), int(p["gap"]), cdmin, int(p["ordm"]), int(p["tsm"]))
    LAST_OPEN = [(SYMS[k] if K <= len(SYMS) else k, opn[k, 1], hidx[int(opn[k, 2]) // 60]) for k in range(K)
                 if opn[k, 0] > 0]
    return pd.Series(eq, hidx), pd.DataFrame(lg, columns=LOGW), pd.Series(inv, hidx), pd.Series(eqlo, hidx)


# ------------------------------------------------------------------ reporting
PERIODS = {"d": (V.T21, V.T24), "t": (V.T24, None), "s": (V.T25, None)}     # 2021-23, 2024-26, since 2025


def report(e, lg, hidx, eqlo=None, syms=SYMS):
    """V.report (%/month compounded over the period span, worst drop on hourly marks, win rate of trades BOUGHT in
    the period, worst trade as % of the equity that sized it) plus the worst drop on minute-close marks."""
    r = V.report(e, lg, hidx, syms)
    if eqlo is not None:
        for p, (a, b) in PERIODS.items():
            m = (e.index >= a) & ((e.index < b) if b is not None else True)
            ee, lo = e[m], eqlo[m]
            if len(ee) < 2:
                continue
            base = ee.iloc[0]
            peak = (ee / base).cummax()
            r[f"{p}_dd_min"] = float(((lo / base) / peak.shift(1, fill_value=1.0) - 1).clip(upper=0).min())
    return r


def worst3(r):
    return min(r["d_pm"], r["t_pm"], r["s_pm"])


def line(r):
    s = []
    for p, lab in (("d", "21-23"), ("t", "24-26"), ("s", "25-26")):
        s.append(f"{lab} {r[f'{p}_pm']*100:+6.2f}%/mo wr {r[f'{p}_win']*100:5.1f}% DD {r[f'{p}_dd']*100:4.0f}%"
                 + (f" (min {r[f'{p}_dd_min']*100:4.0f}%)" if f"{p}_dd_min" in r else ""))
    return " | ".join(s) + (f" | trades {r['d_trades']:.0f}/{r['t_trades']:.0f} | worst trade {r['d_worst_trade']*100:+.1f}%"
                            f"/{r['t_worst_trade']*100:+.1f}%")


# ------------------------------------------------------------------ evaluation helpers
STRESS = dict(pen=0.002, spen=0.002, cdx=60)        # fills 0.2% worse both ways, cooldown noticed 1h later; + K=22


class Ctx:
    """Minute data plus caches of trimmed minute arrays and guarded references (per process)."""

    def __init__(self):
        self.A, self.hidx, self.Ph, self.SLIP = load()
        self._B, self._ref = {}, {}

    def B(self, K, w):
        if (K, w) not in self._B:
            self._B[(K, w)] = wick(self.A, K, w)
        return self._B[(K, w)]

    def ref(self, g, K):
        if (g, K) not in self._ref:
            self._ref[(g, K)] = guard_ref(self.Ph, g, K)
        return self._ref[(g, K)]

    def run(self, K=19, **kw):
        p = dict(WDEF); p.update(kw)
        return wrun(self.A, self.hidx, self.Ph, None, K=K, B=self.B(K, p["wick"]), ref=self.ref(p["guard"], K), **kw)

    def rep(self, K=19, **kw):
        e, lg, _, lo = self.run(K=K, **kw)
        return report(e, lg, self.hidx, lo)


def evaluate(ctx, kw, stress=True):
    """Base run (19 coins) and the stressed run (22 coins, fills 0.2% worse, cooldown +1h) -> one flat dict."""
    r = ctx.rep(19, **kw)
    if stress:
        s = dict(kw); s.update(STRESS)
        x = ctx.rep(22, **s)
        r.update({f"x_{k}": v for k, v in x.items()})
    return r
