"""m13 "futures" (arena contestant 1): 90% mode, with the bid depth set per coin-hour by a Binance USD-M futures
signal (forced-liquidation flush). The bot still trades SPOT only; futures data is only read.

Signals at bid hour t (all from data timestamped before t's open; the bid rests during hour t):
  'oi'    open interest (5-min metrics, hourly buckets; bucket h = create_time in [h, h+1h), known at h+1h) of the
          last closed bucket fell >= x vs N hours earlier: oi[t-1] / oi[t-1-N] - 1 <= -x.
  'fund'  last SETTLED funding rate (calc_time; known from the next hour) <= x.
  'fdrop' last settled funding rate minus its highest value over the previous 72h <= -x.
  'prem'  premium index (perp vs spot index) 1h close of hour t-1 <= -x (futures at a discount).
Depth: d_on where the signal is on, d_off where it is off (None = no bid), 90% mode's 14% where the coin-hour
has NO futures data for that signal (fallback decided before any run; oi data starts ~2021-12, funding / premium
2019-09+ depending on the coin). Everything else is 90% mode (24h highest close anchor, +2.5%, 96h, 3h cooldown,
35% crash guard, 1 coin x 100%, closest coin to its trigger), run through m12_entry_lib.ek (== vk == 90% mode).
Futures data: $PB_FUT/fut/{oi,funding,premium,klines}/SYM.csv.gz (release pb-futures-v1).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m13/futures"))
OUT = os.environ["PB_OUT"]
os.makedirs(OUT, exist_ok=True)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m12_entry_lib as L  # noqa: E402

FUT = os.path.join(os.environ.get("PB_FUT", "."), "fut")
PEN = L.PEN
BASE = dict(sig="none", N=12, x=0.0, d_on=0.14, d_off=0.14)
_raw, _sig = {}, {}


def _read(kind, sym):
    k = (kind, sym)
    if k not in _raw:
        parts = []
        for s in (["RNDRUSDT", "RENDERUSDT"] if sym == "RENDERUSDT" else [sym]):
            p = os.path.join(FUT, kind, f"{s}.csv.gz")
            if os.path.exists(p):
                parts.append(pd.read_csv(p, parse_dates=["t"]))
        if parts:
            d = pd.concat(parts).drop_duplicates("t", keep="last").sort_values("t").set_index("t")
        else:
            d = None
        _raw[k] = d
    return _raw[k]


def known_series(kind, sym, idx):
    """Value known at the OPEN of each hour of idx (NaN = no data)."""
    d = _read({"prem": "premium"}.get(kind, kind), sym)
    if d is None:
        return pd.Series(np.nan, idx)
    if kind == "oi":
        s = d["oi"].where(d["n"] >= 6)              # bucket needs >= 6 of its 12 five-minute records
        s.index = s.index + pd.Timedelta(hours=1)   # known when the bucket closes
        return s.reindex(idx).ffill(limit=2)
    if kind == "prem":
        s = d["c"].copy()
        s.index = s.index + pd.Timedelta(hours=1)
        return s.reindex(idx).ffill(limit=2)
    if kind == "funding":
        s = d["rate"].copy()
        s.index = s.index.floor("h") + pd.Timedelta(hours=1)
        s = s[~s.index.duplicated(keep="last")]
        return s.reindex(idx).ffill(limit=9)
    raise ValueError(kind)


def signal(P, idx, sig, N, x):
    """1 on / 0 off / NaN no data, per coin-hour (n x K)."""
    key = (id(P["o"]), sig, N, x)
    if key in _sig:
        return _sig[key]
    cols = list(P["o"].columns)
    S = np.full((len(idx), len(cols)), np.nan)
    for j, s in enumerate(cols):
        if sig == "oi":
            v = known_series("oi", s, idx)
            r = (v / v.shift(N) - 1.0).values
            S[:, j] = np.where(np.isnan(r), np.nan, (r <= -x).astype(float))
        elif sig in ("fund", "fdrop"):
            v = known_series("funding", s, idx)
            r = v.values if sig == "fund" else (v - v.rolling(72, min_periods=1).max()).values
            S[:, j] = np.where(np.isnan(r), np.nan, (r <= (x if sig == "fund" else -x)).astype(float))
        elif sig == "prem":
            r = known_series("prem", s, idx).values
            S[:, j] = np.where(np.isnan(r), np.nan, (r <= -x).astype(float))
        else:
            raise ValueError(sig)
    _sig[key] = S
    return S


def lim_table(P, idx, d):
    A = P["ref"].values
    n, K = A.shape
    if d["sig"] == "none":
        D = np.full((n, K), 0.14)
    else:
        S = signal(P, idx, d["sig"], int(d["N"]), float(d["x"]))
        off = np.nan if d["d_off"] is None else d["d_off"]
        D = np.where(np.isnan(S), 0.14, np.where(S > 0, d["d_on"], off))
    Lm = A * (1.0 - D)
    g = L.guard(P, 0.35)
    return np.ascontiguousarray(np.where(g > 0, Lm, np.nan))


def frun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    d = dict(BASE, **d)
    O = P["o"].values
    K = O.shape[1]
    Lm = lim_table(P, idx, d)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = L.ek(O, P["h"].values, P["l"].values, P["cf"].values, Lm, np.zeros((1, 1)), 0, 0.0, lastb,
                       0.025, 96, max(3 + cd_add, 0), float(E.EPS), float(E.FEE), sl, float(pen), float(pens),
                       bool(mtp_now), i0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


VERS = L.VERS


def design_scores(d):
    """2021-23 only: 5 nearby versions (as designed, cooldown -1h, +1h, bot timing, start 2021-01-15) x normal and
    every fill 0.2% worse. Never computes a metric after 2023."""
    syms, idx, P = V.market(False)
    out = {}
    for tag, kw in VERS.items():
        for fl, fk in (("", {}), ("w", dict(pen=PEN, pens=PEN))):
            e, lg, inv = frun(P, idx, d, **kw, **fk)
            r = V.summarize(e, lg, inv, idx, (("d", L.LATE if tag == "ls" else V.T21, V.T24),))
            t = fl + tag
            out[f"{t}_wr"], out[f"{t}_mo"], out[f"{t}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
    vs = list(VERS)
    for p in ("", "w"):
        out[f"{p}robust"] = float(np.mean([out[f"{p}{t}_mo"] for t in vs]))
        out[f"{p}rwr"] = float(np.mean([out[f"{p}{t}_wr"] for t in vs]))
        out[f"{p}rdd"] = float(np.median([out[f"{p}{t}_dd"] for t in vs]))
    return out


def evaluate(d, worse=False, stress=False, holdout=False):
    syms, idx, P = V.market_holdout(stress) if holdout else V.market(stress)
    kw = {}
    if worse or stress:
        kw.update(pen=PEN, pens=PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1, liquidate=True)
    e, lg, inv = frun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg


def coverage(holdout=False):
    syms, idx, P = V.market_holdout(False) if holdout else V.market(False)
    rows = []
    for s in syms:
        r = {"sym": s}
        for kind, kk in (("oi", "oi"), ("funding", "funding"), ("prem", "premium")):
            if _read(kk, s) is None:
                r[kind] = "-"
                continue
            v = known_series(kind, s, idx).dropna()
            r[kind] = f"{v.index[0]:%Y-%m-%d}" if len(v) else "-"
        rows.append(r)
    return pd.DataFrame(rows)


def key(d):
    return json.dumps(dict(BASE, **d), sort_keys=True)
