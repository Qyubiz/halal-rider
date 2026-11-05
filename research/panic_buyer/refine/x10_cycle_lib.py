"""Lever "cycle": a faster Panic Buyer that checks every C minutes (5 or 15) instead of every hour.

Bars: the 1-minute candles of engine.minute_panel (same alignment and gap filling, built one coin at a time so
memory stays small) aggregated to C-minute bars. A bot that checks every C minutes and places its take-profit at
the next check (the live bot's rule, at C=60) only needs C-minute OHLC: at each check it places fully funded
resting bids, a bid fills if the bar opens below it (market buy, + slippage) or trades eps through it; the
take-profit becomes active at the next check (open above target -> sells at the open, high eps through -> sells
at the target); the time stop sells at the bar close once held hm bars; a sold coin can be bought again cd bars
later. So ckernel is engine.gkernel with scalar settings, run on C-minute bars, with:
  - trigger ref = highest bar close of the last W bars (known at the check: shifted one bar),
  - pens: sell-side fill penalty (0 = engine behaviour; engine's `pen` only touches buys),
  - an extra log column: account equity when the order was sized (for the worst trade as % of the account).
With C=60, W=24 and pens=0 it is engine.gkernel / engine.run (and engine.mrun delay=0) exactly
(x10_cycle_check.py proves it).
"""
import os
import sys

import numpy as np
import pandas as pd
from numba import njit

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_x10_cycle"))
os.makedirs(OUT, exist_ok=True)
T_DESIGN0, T_TEST0, T_25, T_END = (pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01"),
                                    pd.Timestamp("2025-01-01"), pd.Timestamp("2026-10-22 23:59"))
LOG = ["eb", "xb", "k", "why", "ret", "amt", "entry", "eq0"]


# ------------------------------------------------------------------ bars from 1-minute candles
def _read1m(s, sub="1m"):
    p = os.path.join(E.DATA, sub, f"{s}.npz")
    z = np.load(p, allow_pickle=False)
    return pd.DataFrame({f: z[f] for f in ("o", "h", "l", "c")}, index=pd.to_datetime(z["t"], unit="s"))


def _coin1m(s, start):
    d = _read1m(s)
    if s == "RENDERUSDT":
        r = _read1m("RNDRUSDT")
        d = pd.concat([r, d])
        d = d[~d.index.duplicated(keep="last")].sort_index()
    return d[d.index >= start]


def _bars_path(C, dead):
    return os.path.join(OUT, f"bars_{C}m{'_dead' if dead else ''}.npz")


def build_bars(Cs, dead=False, start="2021-01-01"):
    """C-minute bars (for each C in Cs) for the 19 live coins (+ LUNA/FTT/SRM cut at their collapse), built from
    minute arrays made exactly like engine.minute_panel makes them, one coin at a time. Cached in OUT."""
    syms = list(E.H19) + (list(E.DEAD) if dead else [])
    Cs = [C for C in Cs if not os.path.exists(_bars_path(C, dead))]
    if not Cs:
        return
    # global minute index, as minute_panel: first hour of any coin .. last hour of any coin + 59 min
    firsts, lasts = [], []
    for s in syms:
        d = _coin1m(s, start)
        firsts.append(d.index[0]); lasts.append(d.index[-1])
        del d
    t0 = min(firsts).floor("1h")
    t1 = max(lasts).floor("1h") + pd.Timedelta(minutes=59)
    idx = pd.date_range(t0, t1, freq="1min")
    assert len(idx) % 60 == 0 and all(60 % C == 0 for C in Cs)
    out = {C: {f: np.full((len(idx) // C, len(syms)), np.nan) for f in ("o", "h", "l", "c")} for C in Cs}
    for j, s in enumerate(syms):
        d = _coin1m(s, start).reindex(idx)
        first = d.c.first_valid_index()
        cc = d.c.ffill()
        live = idx >= first
        A = {}
        for f in ("o", "h", "l"):
            col = d[f].where(d[f].notna(), cc)
            A[f] = np.where(live, col.values, np.nan).astype(np.float64)
        A["c"] = np.where(live, cc.values, np.nan).astype(np.float64)
        del d, cc
        if dead and s in E.DEAD:
            gone = idx > pd.Timestamp(E.DEAD[s]) + pd.Timedelta(minutes=59)
            for f in A:
                A[f][gone] = np.nan
        for C in Cs:
            nb = len(idx) // C
            out[C]["o"][:, j] = A["o"][::C]
            out[C]["c"][:, j] = A["c"][C - 1::C]
            out[C]["h"][:, j] = np.fmax.reduce(A["h"].reshape(nb, C), axis=1)
            out[C]["l"][:, j] = np.fmin.reduce(A["l"].reshape(nb, C), axis=1)
        del A
    for C in Cs:
        t = idx[::C]
        np.savez(_bars_path(C, dead), syms=np.array(syms), t=t.values.astype("datetime64[s]").astype(np.int64), **out[C])


def load_bars(C, dead=False):
    build_bars([C], dead)
    z = np.load(_bars_path(C, dead), allow_pickle=False)
    return {"syms": [str(x) for x in z["syms"]], "t": pd.to_datetime(z["t"], unit="s"), "C": C,
            **{f: z[f] for f in ("o", "h", "l", "c")}}


def make_ref(B, W):
    """Highest bar close of the last W bars, known at the bar's open (rolling max of raw closes, shifted 1)."""
    return pd.DataFrame(B["c"]).rolling(W, min_periods=1).max().shift(1).values


def ffill_close(B):
    return pd.DataFrame(B["c"]).ffill().values


def slip_vec(syms):
    """0.02% market-order slippage for coins in the top 50 of the screened universe by median daily volume (all
    19 live coins are, ranks 1-43), 0.1% for the collapse coins (conservative: they are stress-only)."""
    return np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in syms])


# ------------------------------------------------------------------ kernel
@njit(cache=True)
def ckernel(o, h, l, c, ref, depth, nord, fmax, tp, part, tp2, trail, hm, cd, eps, fee, slip, pen, pens,
            tpwait, tprest, mtp_now):
    """engine.gkernel with scalar settings (depth, nord, fmax, tp, hm, cd) on any bar length, plus a sell-side
    penalty `pens` and the sizing equity in the log. All counts (hm, cd, tpwait) are in bars."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K); got = np.zeros(K); tpp = np.zeros(K)
    stage = np.zeros(K, np.int64); sbar = np.zeros(K, np.int64); peak = np.zeros(K); eq0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64); mkt = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 2000000; lg = np.zeros((maxlog, 8)); nl = 0
    for t in range(n):
        anyfill = False
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]):
                continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth)) - 1.0
            lim = ref[t, k] * (1.0 - depth)
            if o[t, k] <= lim or l[t, k] <= lim * (1.0 - eps):
                anyfill = True
        nsel = 0
        if anyfill and cash > 1e-9:          # no eligible coin trades through its bid: nothing can fill (exact)
            order = np.argsort(dist)
            for q in range(K):
                if dist[order[q]] < 1e8 and nsel < nord:
                    nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth)
                fill = -1.0; ismkt = 0
                if o[t, k] <= lim:
                    fill = o[t, k] * (1.0 + slip[k]) * (1.0 + pen); ismkt = 1
                elif l[t, k] <= lim * (1.0 - eps):
                    fill = lim * (1.0 + pen)
                if fill > 0.0 and cash > 1e-9:
                    mkt[k] = ismkt if mtp_now else 0
                    a = min(per * 1.0, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t; got[k] = 0.0; eq0[k] = eq_prev
                    tpp[k] = fill * (1.0 + tp); stage[k] = 0; peak[k] = fill
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
                        p1 = q * px * (1.0 - pens) * (1.0 - fee)
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
                if px < 0.0 and t - ebar[k] >= hm - mkt[k]:
                    px = c[t, k] * (1.0 - slip[k]); why = 1.0
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
    return eq, lg[:nl].copy()


def crun(B, ref, cf, depth, N, fmax, tp, hm, cd, part=1.0, tp2=0.0, trail=0.0, eps=E.EPS, fee=E.FEE, slip=None,
         pen=0.0, pens=0.0, tpwait=1, tprest=True, mtp_now=False):
    """Run the cycle bot. hm, cd, tpwait in bars. ref from make_ref(B, W); cf from ffill_close(B)."""
    sl = slip_vec(B["syms"]) if slip is None else (np.full(len(B["syms"]), slip) if np.ndim(slip) == 0
                                                   else np.asarray(slip, np.float64))
    eq, lg = ckernel(B["o"], B["h"], B["l"], cf, ref, float(depth), int(N), float(fmax), float(tp), float(part),
                     float(tp2), float(trail), int(hm), int(cd), float(eps), float(fee), sl, float(pen), float(pens),
                     int(tpwait), bool(tprest), bool(mtp_now))
    return eq, lg


# ------------------------------------------------------------------ metrics
class Clock:
    """Period boundaries for an equity array on index t (bar opens; eq[i] is the account at bar i's close)."""

    def __init__(self, t):
        self.t = t
        self.tc = t + (t[1] - t[0])
        self.i = {name: int(np.searchsorted(t, x, side="left")) for name, x in
                  (("d0", T_DESIGN0), ("t0", T_TEST0), ("y25", T_25))}
        self.n = len(t)
        ar = pd.Series(np.arange(len(t)), index=t)
        self.yend = {k.year: int(v) for k, v in ar.resample("YE").last().items()}
        me = ar.resample("ME").last()
        self.mend = me.values.astype(np.int64)
        self.mlab = me.index

    def period(self, eq, a, b):
        """Metrics on eq[a:b] with the account value just before bar a as the start (1.0 at bar 0)."""
        start = eq[a - 1] if a > 0 else 1.0
        e = eq[a:b] / start
        yrs = (self.tc[b - 1] - self.t[a]).total_seconds() / (365.25 * 86400)
        mult = float(e[-1])
        mo = mult ** (1.0 / (12 * yrs)) - 1.0 if mult > 0 else -1.0
        pk = np.maximum.accumulate(np.concatenate([[1.0], e]))
        dd = float((np.concatenate([[1.0], e]) / pk - 1.0).min())
        return mult, mo, dd

    def year(self, eq, y):
        a = self.yend.get(y - 1); b = self.yend.get(y)
        if b is None:
            return np.nan
        s = eq[a] if a is not None else 1.0
        return float(eq[b] / s - 1.0)

    def months(self, eq):
        v = eq[self.mend]
        prev = np.concatenate([[1.0], v[:-1]])
        return pd.Series(v / prev - 1.0, index=self.mlab)


def summarize(clk, eq, lg=None):
    """The three report periods + 2022 + trade stats. %/month = multiple ** (1 / months) - 1."""
    r = {}
    for tag, a, b in (("d", clk.i["d0"], clk.i["t0"]), ("t", clk.i["t0"], clk.n), ("s", clk.i["y25"], clk.n)):
        m, mo, dd = clk.period(eq, a, b)
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = m, mo, dd
    for y in (2021, 2022, 2023, 2024, 2025, 2026):
        r[f"y{y}"] = clk.year(eq, y)
    if lg is not None:
        r["trades"] = len(lg)
        if len(lg):
            acct = lg[:, 4] * lg[:, 5] / lg[:, 7]
            tt = lg[:, 0] >= clk.i["t0"]
            r["worst_trade_acct"] = float(acct.min())
            r["worst_trade_acct_t"] = float(acct[tt].min()) if tt.any() else np.nan
            r["worst_trade_ret"] = float(lg[:, 4].min())
            r["t_trades"] = int(tt.sum())
            r["t_win"] = float((lg[tt, 4] > 0).mean()) if tt.any() else np.nan
            r["t_avg"] = float(lg[tt, 4].mean()) if tt.any() else np.nan
    return r


def fmt4(name, r, rs=None):
    """One design, the four reports. r: normal run; rs: stressed run (fills 0.2% worse + LUNA/FTT/SRM)."""
    s = (f"{name}\n  (1) 2024-26  {r['t_mo']*100:+6.2f}%/mo  x{r['t_mult']:7.2f}  drop {r['t_dd']*100:5.1f}%\n"
         f"  (2) 2025-26  {r['s_mo']*100:+6.2f}%/mo  x{r['s_mult']:7.2f}  drop {r['s_dd']*100:5.1f}%\n"
         f"  (3) 2021-23  {r['d_mo']*100:+6.2f}%/mo  x{r['d_mult']:7.2f}  drop {r['d_dd']*100:5.1f}%  2022 {r['y2022']*100:+5.1f}%")
    if "worst_trade_acct" in r:
        s += (f"\n      trades {r['trades']}, worst trade {r['worst_trade_acct']*100:+.1f}% of account "
              f"(2024-26: {r['worst_trade_acct_t']*100:+.1f}%), 2024-26 win {r['t_win']*100:.0f}% avg {r['t_avg']*100:+.2f}%")
    if rs is not None:
        s += (f"\n  (4) fills 0.2% worse + LUNA/FTT/SRM: 2024-26 {rs['t_mo']*100:+6.2f}%/mo drop {rs['t_dd']*100:5.1f}% | "
              f"2025-26 {rs['s_mo']*100:+6.2f}%/mo drop {rs['s_dd']*100:5.1f}% | 2021-23 {rs['d_mo']*100:+6.2f}%/mo "
              f"drop {rs['d_dd']*100:5.1f}% 2022 {rs['y2022']*100:+5.1f}%")
        if "worst_trade_acct" in rs:
            s += f" | worst trade {rs['worst_trade_acct']*100:+.1f}% of account"
    return s
