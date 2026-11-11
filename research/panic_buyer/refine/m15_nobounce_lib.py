"""m15 nobounce: copy of m11_fusion_lib.fk (single tier, guardrails off) plus the "never bounced" exit.
Per position we track mxh = highest hourly high over the bars AFTER the fill bar (ebar+1 .. t-1; no lookahead).
At the open of hour t with t - ebar == N+1 (so N full hours observed), if mxh < entry*(1+y):
  mode 1: sell at that open (market, -slip), why=6
  mode 2: move the take-profit to entry*(1+tp2) (limit; keeps the 96h time stop), why stays 0/1
nbmode 0 = off (== 90% mode, proven in m15_nobounce_check.py)."""
import os, sys, re, inspect
import numpy as np, pandas as pd
from numba import njit
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m11_fusion_lib as F, m10_velocity_lib as V, engine as E

src = inspect.getsource(F.fk).replace("@njit(cache=True)\n", "")
src = src.replace("def fk(", "def fk2(").replace("hs, zcap, killm, killw, klock, cln, clT):",
                  "hs, zcap, killm, killw, klock, cln, clT, nbN, nby, nbmode, nbtp):")
src = src.replace("eq0 = np.zeros((S, K)); ebar", "mxh = np.zeros((S, K)); eq0 = np.zeros((S, K)); ebar")
src = src.replace("entry[s, k] = fill; amt0", "mxh[s, k] = 0.0; entry[s, k] = fill; amt0")
old = "                    if ebar[s, k] < t or mkt[s, k] == 1:\n"
new = """                    if nbmode > 0 and t - ebar[s, k] == nbN + 1 and mxh[s, k] < entry[s, k] * (1.0 + nby):
                        if nbmode == 2:
                            tpp[s, k] = entry[s, k] * (1.0 + nbtp)
                        elif o[t, k] < tpp[s, k]:
                            px = o[t, k] * (1.0 - slip[k]); why = 6.0
                    if px < 0.0 and (ebar[s, k] < t or mkt[s, k] == 1):
"""
assert old in src
src = src.replace(old, new)
# update mxh at end of exits for positions still held (after exit block, bars after fill bar)
old2 = "            for k in range(K):\n                if units[s, k] > 0.0 and lastb[k] == t:"
new2 = """            for k in range(K):
                if units[s, k] > 0.0 and ebar[s, k] < t and h[t, k] > mxh[s, k]:
                    mxh[s, k] = h[t, k]
            for k in range(K):
                if units[s, k] > 0.0 and lastb[k] == t:"""
assert old2 in src
src = src.replace(old2, new2)
exec(compile(src, "fk2_src", "exec"), globals())
fk2 = njit(cache=False)(fk2)

OFF = dict(N=24, y=0.01, mode=0, tp2=0.005, depth=0.14, tp=0.025, hm=96)


def frun(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    O = P["o"].values; n, K = O.shape
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    g = F.guard(P, 0.35)
    gm = np.ascontiguousarray(g[None])
    tpS = np.ascontiguousarray(F.tp_table(P, d["tp"])[None])
    R = np.ascontiguousarray(np.stack([F.ref(P, 24)])); rid = np.zeros(1, np.int64)
    f = lambda v, dt=np.float64: np.array(v, dt)  # noqa: E731
    cd = max(3 + cd_add, 0)
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    eq, lg, inv = fk2(O, P["h"].values, P["l"].values, P["cf"].values, gm, R, rid, lastb, f([d["depth"]]), tpS,
                      f([d["hm"]], np.int64), f([cd], np.int64), f([cd], np.int64), f([1.0]), 99, 1, float(E.EPS),
                      float(E.FEE), sl, float(pen), float(pens), bool(mtp_now), i0, 0.0, 0.0, 0.0, 720, 168, 0, 0,
                      int(d["N"]), float(d["y"]), int(d["mode"]), float(d["tp2"]))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


PER_D = (("d", V.T21, V.T24),)


def scores(d):
    """2021-23 only: normal, worse, 5 nearby versions."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = frun(P, idx, d, **kw)
        per = (("d", F.LATE, V.T24),) if tag == "ls" else PER_D
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    out["rmin"] = float(np.min([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


def evaluate(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress:
        kw.update(pen=F.PEN, pens=F.PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1)
    if dead:
        kw["liquidate"] = True
    e, lg, inv = frun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
