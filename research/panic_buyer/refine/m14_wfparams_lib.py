"""m14 wfparams: walk-forward re-tuning of 90% mode's depth and take-profit (engine = m11 fk, time-varying depth/tp).

Time-varying depth is done without touching the kernel: the reference table R (24h highest close) is multiplied at
each hour by (1-depth_t)/(1-0.14) while the kernel keeps depth 0.14; the take-profit table tpS is set per hour.
Both are fixed at the entry hour's open, so no lookahead. A refit at month start D picks (depth, tp) from standalone
runs of every candidate over the W months before D (calendar months), using only completed months and trades that
exited before D. Months before 2021 come from the 2018-20 holdout panel (past data only; a refit date D only ever
reads months < D).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import m11_fusion_lib as F  # noqa: E402

DEF = (0.14, 0.025)
DEPTHS = [0.12, 0.13, 0.14, 0.15, 0.16]
TPS = [0.02, 0.025, 0.03]
CANDSETS = {
    "full": [(d, t) for d in DEPTHS for t in TPS],
    "depth": [(d, 0.025) for d in DEPTHS],
    "tp": [(0.14, t) for t in TPS],
    "coarse": [(d, t) for d in (0.12, 0.14, 0.16) for t in (0.02, 0.03)],
}
MODES = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN),
         "st": dict(pen=F.PEN, pens=F.PEN, mtp_now=True, cd_add=1, liquidate=True)}


def run(P, idx, dep_t, tp_t, **kw):
    """dep_t, tp_t: per-hour arrays (len n). Same as F.frun for the 90% tier with time-varying depth/tp."""
    O = P["o"].values
    n, K = O.shape
    liq = kw.get("liquidate", False)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liq else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liq else np.full(K, -1, np.int64)
    R = np.ascontiguousarray(F.ref(P, 24)[None] * ((1.0 - dep_t) / (1.0 - 0.14))[None, :, None])
    gm = np.ascontiguousarray((F.guard(P, 0.35))[None])
    tpS = np.ascontiguousarray(np.repeat(tp_t[:, None], K, 1)[None])
    t0 = kw.get("t0")
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    one = lambda x, dt=np.float64: np.array([x], dt)  # noqa: E731
    cda = kw.get("cd_add", 0)
    eq, lg, inv = F.fk(O, P["h"].values, P["l"].values, P["cf"].values, gm, R, np.zeros(1, np.int64), lastb, one(0.14), tpS,
                       one(96, np.int64), np.maximum(one(3, np.int64) + cda, 0), np.maximum(one(3, np.int64) + cda, 0),
                       one(1.0), 99, 1, float(E.EPS), float(E.FEE), sl, float(kw.get("pen", 0.0)), float(kw.get("pens", 0.0)),
                       bool(kw.get("mtp_now", False)), i0, 0.0, 0.0, 0.0, 720, 168, 0, 0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


def const(P, idx, d, tp, **kw):
    n = len(idx)
    return run(P, idx, np.full(n, d), np.full(n, tp), **kw)


# ---------------------------------------------------------------- candidate monthly stats
_st = {}


def _monthly(e, lg, idx):
    m = idx.to_period("M")
    ls = np.log(e.groupby(m).last())
    lr = ls.diff()
    lr.iloc[0] = ls.iloc[0]
    x = lg[lg.why != 9]
    xm = m[x.xb.astype(int).values]
    w = pd.Series((x.ret > 0).values, index=xm).groupby(level=0).sum()
    c = pd.Series(1, index=xm).groupby(level=0).sum()
    return pd.DataFrame({"lr": lr, "w": w, "n": c}).fillna({"w": 0, "n": 0})


def cand_stats(mode):
    """dict (d,tp) -> monthly DataFrame, months of 2018-01..2026-10 (holdout panel for months < 2021, main from 2021)."""
    if mode in _st:
        return _st[mode]
    kw = MODES[mode]
    dead = mode == "st"
    _, ih, Ph = V.market_holdout(dead)
    _, im, Pm = V.market(dead)
    out = {}
    for d in DEPTHS:
        for t in TPS:
            a = _monthly(*const(Ph, ih, d, t, **kw)[:2], ih)
            b = _monthly(*const(Pm, im, d, t, **kw)[:2], im)
            a = a[a.index < pd.Period("2021-01", "M")]
            out[(d, t)] = pd.concat([a, b])
    _st[mode] = out
    return out


# ---------------------------------------------------------------- the re-tuning rule
def pick(stats, cands, D, W, score, minn=20, minm=12):
    """D: pd.Period of the refit month (data strictly before it). Returns (depth, tp)."""
    best = None; bv = -9
    for c in cands:
        s = stats[c]
        s = s[(s.index < D) & (s.index >= D - W)]
        if len(s) < min(W, minm) or s.n.sum() < minn:
            return DEF
        mo = np.exp(s.lr.mean()) - 1.0
        wr = s.w.sum() / s.n.sum()
        if score == "wr92":
            ok, v = wr >= 0.92, mo
        elif score == "wr94":
            ok, v = wr >= 0.94, mo
        elif score == "mo":
            ok, v = True, mo
        else:                                  # "wr": best win rate among candidates with positive %/month
            ok, v = mo > 0, wr
        if ok and v > bv:
            bv, best = v, c
    return best if best is not None else DEF


def schedule(idx, stats, des):
    """Per-hour (depth, tp) arrays for a design dict(W, R, cs, sc): refit at the first of every R-th month (Jan, ...)."""
    cands = CANDSETS[des["cs"]]
    per = idx.to_period("M")
    ums = per.unique()
    dep = np.empty(len(idx)); tp = np.empty(len(idx))
    cur = DEF
    last = None
    for m in ums:
        if (m.month - 1) % des["R"] == 0:
            cur = pick(stats, cands, m, des["W"], des["sc"])
        sel = np.asarray(per == m)
        dep[sel], tp[sel] = cur
    return dep, tp


def wf_run(des, panel="main", mode="n", **extra):
    kw = dict(MODES[mode]); kw.update(extra)
    dead = mode == "st"
    _, idx, P = V.market_holdout(dead) if panel == "hold" else V.market(dead)
    dep, tp = schedule(idx, cand_stats(mode if mode != "n" or True else "n"), des)
    return run(P, idx, dep, tp, **kw), dep, tp, idx
