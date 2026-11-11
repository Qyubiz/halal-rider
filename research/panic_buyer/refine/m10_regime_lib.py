"""m10 "regime" angle: raise the win rate of 90% mode by NOT buying in dangerous market regimes.

90% mode (panic_config.90wr.example.json, w90_wr90_final SPEC F2): 1 coin x 100% of the money, 14% dip under the
24h high, +2.5% take-profit, 96h time stop, 3h cooldown, crash guard 35% (skip a coin whose last close is > 35% under
its 168h high). Runner = w90_wr90_lib.wrun (== engine.run with every new switch off), with an extra entry mask (smult)
built from the regime features below. smult 0 on (hour, coin) = no bid on that coin that hour (the bid then goes to
the next-closest coin, exactly as engine.gkernel does with smult).

Every feature is known at the OPEN of hour t: it uses closes up to hour t-1 only (shift(1) everywhere). Features are
computed once on a LONG panel (Dec 2017 - Oct 2026: the 2018-20 holdout candles joined to the 2021-26 candles), so
30/200-day lookbacks are warmed up in Jan 2021, then re-indexed to whatever panel is being run. Market-wide features
use only the user's 19 coins (whichever of them were trading at t); LUNA/FTT/SRM never enter them.

Periods: h = 2018-20 holdout (separate 16-coin panel, engine.load_pre), d = 2021-23 (design), t = 2024-26, s = since
2025-01-01. Win rate: trades bought in the period with ret > 0 (positions open at the end marked at the last close,
why=9, as w90_wr90_lib). Worst trade = ret x money / account equity at entry. Trades/month = trades bought / months.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_m10/regime"))
os.makedirs(OUT, exist_ok=True)

MODE90 = dict(N=1, fmax=1.0, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.35)
PERIODS = (("h", pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")),
           ("d", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
           ("t", pd.Timestamp("2024-01-01"), None),
           ("s", pd.Timestamp("2025-01-01"), None))
PEN = 0.002            # "fills 0.2% worse": every buy AND every sell 0.2% worse


# ------------------------------------------------------------------ data
def _long_dict(dead=False):
    """sym -> hourly candles Dec 2017 .. Oct 2026 (holdout files joined to the 2021-26 files)."""
    pre = E.load_pre(dead=dead)
    post = E.load_h19(dead=dead, start="2017-01-01")
    D = {}
    for s in set(pre) | set(post):
        parts = [x for x in (pre.get(s), post.get(s)) if x is not None]
        d = pd.concat([x[["o", "h", "l", "c", "qv"]] for x in parts])
        D[s] = d[~d.index.duplicated(keep="last")].sort_index()
    return D


_mk = {}


def market(which):
    """which: 'd' 2021-26 panel (19 coins), 'dx' 2021-26 + LUNA/FTT/SRM, 'h' 2018-20 holdout (16 coins),
    'hx' holdout + LUNA/FTT/SRM. Same panels as w90_wr90_lib.market / holdout_eval.evaluate."""
    if which not in _mk:
        if which in ("d", "dx"):
            _mk[which] = E.panel(E.load_h19(dead=which == "dx"))
        else:
            _mk[which] = E.panel(E.load_pre(dead=which == "hx"))
    return _mk[which]


# ------------------------------------------------------------------ features (known at the open of hour t)
_F = {}


def features():
    """dict name -> DataFrame (coin features, columns = symbols) or Series (market features), on the long index."""
    if _F:
        return _F
    D = _long_dict(dead=True)
    syms = sorted(D)
    idx = pd.date_range(min(d.index[0] for d in D.values()), max(d.index[-1] for d in D.values()), freq="1h")
    C = pd.DataFrame({s: D[s]["c"] for s in syms}).reindex(idx)
    QV = pd.DataFrame({s: D[s]["qv"] for s in syms}).reindex(idx)
    O = pd.DataFrame({s: D[s]["o"] for s in syms}).reindex(idx)
    cf = C.ffill()
    c1 = cf.shift(1)
    live = C.shift(1).rolling(24, min_periods=1).count() > 0          # a real close in the last 24h
    core = [s for s in syms if s in E.H19]
    lv = live[core]
    nlive = lv.sum(axis=1).clip(lower=1)

    def ret(h):
        return c1 / cf.shift(1 + h) - 1

    F = {}
    # ---- coin features
    F["c_dip24"] = c1 / cf.rolling(24, min_periods=1).max().shift(1) - 1     # last close vs 24h high (the ref)
    F["c_r1"] = ret(1)
    F["c_r4"] = ret(4)
    F["c_r24"] = ret(24)
    F["c_pre3d"] = cf.shift(25) / cf.shift(97) - 1                           # 3 days BEFORE the last 24h
    F["c_pre7d"] = cf.shift(25) / cf.shift(193) - 1                          # 7 days before the last 24h
    F["c_r7d"] = ret(168)
    F["c_r30d"] = ret(720)
    F["c_dd30"] = c1 / cf.rolling(720, min_periods=24).max().shift(1) - 1
    F["c_sma7"] = c1 / cf.rolling(168, min_periods=84).mean().shift(1) - 1
    F["c_sma30"] = c1 / cf.rolling(720, min_periods=360).mean().shift(1) - 1
    lr = np.log(cf).diff()
    F["c_rv"] = lr.rolling(24, min_periods=12).std().shift(1) / lr.rolling(720, min_periods=240).std().shift(1)
    F["c_vol30"] = lr.rolling(720, min_periods=240).std().shift(1) * np.sqrt(24)          # daily vol, last 30 days
    F["c_vol7"] = lr.rolling(168, min_periods=84).std().shift(1) * np.sqrt(24)            # daily vol, last 7 days
    q24 = QV.rolling(24, min_periods=12).sum()
    F["c_qv"] = (q24 / q24.rolling(720, min_periods=240).median()).shift(1)
    # ---- market features (19 core coins only)
    btc = cf["BTCUSDT"]
    b1 = btc.shift(1)
    for h in (1, 4, 12, 24, 72, 168):
        F[f"btc_r{h}"] = b1 / btc.shift(1 + h) - 1
    F["btc_dd7"] = b1 / btc.rolling(168, min_periods=1).max().shift(1) - 1
    F["btc_dd30"] = b1 / btc.rolling(720, min_periods=1).max().shift(1) - 1
    F["btc_sma50"] = b1 / btc.rolling(24 * 50, min_periods=24 * 20).mean().shift(1) - 1
    F["btc_sma200"] = b1 / btc.rolling(24 * 200, min_periods=24 * 60).mean().shift(1) - 1
    blr = np.log(btc).diff()
    F["btc_rv"] = blr.rolling(24, min_periods=12).std().shift(1) / blr.rolling(720, min_periods=240).std().shift(1)
    F["btc_rv72"] = blr.rolling(72, min_periods=36).std().shift(1) / blr.rolling(720, min_periods=240).std().shift(1)
    bq = QV["BTCUSDT"].rolling(24, min_periods=12).sum()
    F["btc_qv"] = (bq / bq.rolling(720, min_periods=240).median()).shift(1)
    tq = QV[core].sum(axis=1, min_count=1).rolling(24, min_periods=12).sum()
    F["mkt_qv"] = (tq / tq.rolling(720, min_periods=240).median()).shift(1)
    with np.errstate(invalid="ignore"):
        dip = F["c_dip24"][core]
        r24 = F["c_r24"][core]
        for x in (0.05, 0.08, 0.10, 0.14):
            F[f"brd_dip{int(x * 100)}"] = ((dip <= -x) & lv).sum(axis=1) / nlive          # share of coins in a dip
        for x in (0.05, 0.10, 0.15):
            F[f"brd_r24_{int(x * 100)}"] = ((r24 <= -x) & lv).sum(axis=1) / nlive
        F["brd_sma7"] = ((F["c_sma7"][core] < 0) & lv).sum(axis=1) / nlive              # share under 7-day average
        F["brd_sma30"] = ((F["c_sma30"][core] < 0) & lv).sum(axis=1) / nlive
    F["mkt_r24"] = r24.where(lv).median(axis=1)
    F["mkt_r72"] = ret(72)[core].where(lv).median(axis=1)
    F["mkt_r7d"] = F["c_r7d"][core].where(lv).median(axis=1)
    mrv = lr[core].rolling(24, min_periods=12).std() / lr[core].rolling(720, min_periods=240).std()
    F["mkt_rv"] = mrv.shift(1).where(lv).median(axis=1)
    # ---- consecutive red UTC days (as of the last COMPLETED day): BTC, and the median coin's daily return
    dc = btc.resample("1D").last()
    do = O["BTCUSDT"].resample("1D").first()
    red = (dc < do).astype(int)
    streak = red.groupby((red == 0).cumsum()).cumsum()
    F["btc_red"] = streak.shift(1).reindex(idx, method="ffill").fillna(0)              # day d uses days < d
    dret = cf[core].resample("1D").last().pct_change(fill_method=None)
    mred = (dret.median(axis=1) < 0).astype(int)
    mstreak = mred.groupby((mred == 0).cumsum()).cumsum()
    F["mkt_red"] = mstreak.shift(1).reindex(idx, method="ffill").fillna(0)
    # ---- hours since the last market crash hour (event known at t uses closes <= t-1)
    for name, ev in (("since_b10", F["btc_r24"] <= -0.10), ("since_b7", F["btc_r24"] <= -0.07),
                     ("since_brd50", F["brd_dip10"] >= 0.5), ("since_brd30", F["brd_dip10"] >= 0.3)):
        ev = ev.fillna(False).values
        last = np.full(len(ev), -10 ** 9, np.int64)
        lt = -10 ** 9
        for i in range(len(ev)):
            if ev[i]:
                lt = i
            last[i] = lt
        F[name] = pd.Series(np.arange(len(ev)) - last, idx).astype(float)
    F["_idx"] = idx
    _F.update(F)
    return _F


def aligned(name, P):
    """Feature `name` re-indexed to panel P: (n, K) array for coin features, (n,) for market features."""
    F = features()
    x = F[name]
    idx = P["o"].index
    if isinstance(x, pd.DataFrame):
        return x.reindex(index=idx, columns=P["o"].columns).values
    return x.reindex(idx).values


# ------------------------------------------------------------------ rules -> entry mask
OPS = {"<=": np.less_equal, ">=": np.greater_equal, "<": np.less, ">": np.greater}


def block(P, rules):
    """rules: list of conditions; each condition is a list of (feature, op, value) AND-ed together; the conditions
    are OR-ed. Returns the (n, K) boolean 'blocked' array. NaN features never block."""
    n, K = P["o"].shape
    out = np.zeros((n, K), bool)
    for cond in rules:
        m = np.ones((n, K), bool)
        for f, op, v in cond:
            a = aligned(f, P)
            with np.errstate(invalid="ignore"):
                b = OPS[op](a, v) & ~np.isnan(a)
            m &= b if b.ndim == 2 else b[:, None]
        out |= m
    return out


def run(which, rules=(), spec=None, deep=None, depth_fn=None, **over):
    """90% mode on panel `which` with the regime rules. deep=None: blocked = no bid (smult 0); deep=x: blocked
    hours use depth x instead (a deeper bid, not a skip). depth_fn(P) -> (n, K) trigger depth (e.g. scaled by the
    coin's volatility). Returns (summary dict, equity, log, idx)."""
    syms, idx, P = market(which)
    kw = dict(MODE90 if spec is None else spec)
    kw.update(over)
    if depth_fn is not None:
        kw["depth"] = np.ascontiguousarray(depth_fn(P), dtype=np.float64)
    if rules:
        b = block(P, rules)
        if deep is None:
            kw["smult"] = np.where(b, 0.0, 1.0)
        else:
            kw["depth"] = np.where(b, float(deep), kw["depth"])
    if which in ("dx", "hx"):
        kw["liquidate"] = True
    e, lg = W.wrun(P, idx, **kw)
    return summarize(e, lg, idx, which), e, lg, idx


def stress_kw():
    return dict(pen=PEN, pens=PEN)


def bot_kw(spec=None):
    s = MODE90 if spec is None else spec
    return dict(pen=PEN, pens=PEN, mtp_now=True, cd=int(s["cd"]) + 1)


# ------------------------------------------------------------------ metrics
def summarize(e, lg, idx, which="d"):
    r = {}
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    acct = (lg.ret * lg.amt / lg.eq0).values if len(lg) else np.zeros(0)
    pers = [p for p in PERIODS if (p[0] == "h") == (which in ("h", "hx"))]
    for tag, a, b in pers:
        m, mo, dd = W.period(e, a, b)
        sel = (tb >= a) & ((tb < b) if b is not None else True)
        x = lg[sel]
        end = e.index[-1] + pd.Timedelta(hours=1) if b is None else b
        months = (end - a).total_seconds() / (365.25 * 86400) * 12
        r[f"{tag}_mult"], r[f"{tag}_mo"], r[f"{tag}_dd"] = m, mo, dd
        r[f"{tag}_n"] = int(len(x))
        r[f"{tag}_tpm"] = len(x) / months
        r[f"{tag}_wr"] = float((x.ret > 0).mean()) if len(x) else np.nan
        r[f"{tag}_nl"] = int((x.ret <= 0).sum())
        r[f"{tag}_wt"] = float(acct[sel].min()) if sel.any() else np.nan
        r[f"{tag}_avgw"] = float(x.ret[x.ret > 0].mean()) if (x.ret > 0).any() else np.nan
        r[f"{tag}_avgl"] = float(x.ret[x.ret <= 0].mean()) if (x.ret <= 0).any() else np.nan
    return r


def fmt(r, tags=("h", "d", "t", "s")):
    out = []
    for t in tags:
        if f"{t}_mo" not in r:
            continue
        out.append(f"{t}: WR {r[f'{t}_wr']*100:5.1f}% {r[f'{t}_mo']*100:+6.2f}%/mo DD {r[f'{t}_dd']*100:4.0f}% "
                   f"wt {r[f'{t}_wt']*100:+4.0f}% {r[f'{t}_tpm']:4.1f}tr/mo ({r[f'{t}_nl']}L)")
    return " | ".join(out)


def rules_str(rules):
    return " OR ".join("(" + " & ".join(f"{f}{op}{v:g}" for f, op, v in c) + ")" for c in rules) if rules else "none"
