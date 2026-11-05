"""m14 speed lib: two tiers split by how fast the coin fell. age[t,k] = hours since the 24h highest close (0 = the last
closed hour is the high), known at hour open. fast = age<=A. fast tier bids depth df, slow tier depth ds (None = no bid)."""
import os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import m11_fusion_lib as F, m10_velocity_lib as V, engine as E
OUT = F.OUT
OFF = dict(A=12, df=0.14, ds=0.14)
_c = {}


def age(P):
    k = ("a", id(P["o"]))
    if k not in _c:
        c = P["c"].values; ref = P["ref"].values; n, K = c.shape
        a = np.full((n, K), np.nan)
        for L in range(24, 0, -1):          # smallest lag wins -> loop descending, overwrite
            sh = np.full((n, K), np.nan); sh[L:] = c[:-L]
            with np.errstate(invalid="ignore"):
                a = np.where(sh == ref, L - 1, a)
        _c[k] = a
    return _c[k]


def frun(P, idx, d, **kw):
    O = P["o"].values; n, K = O.shape
    g = F.guard(P, 0.35)
    ag = age(P)
    with np.errstate(invalid="ignore"):
        fast = np.where(np.isfinite(ag), ag <= d["A"], True)
    mf = g * fast * (0.0 if d["df"] is None else 1.0)
    ms = g * (~fast) * (0.0 if d["ds"] is None else 1.0)
    gm = np.ascontiguousarray(np.stack([mf, ms]))
    tp = F.tp_table(P, 0.025)
    tpS = np.ascontiguousarray(np.stack([tp, tp]))
    refs = np.ascontiguousarray(np.stack([F.ref(P, 24)]))
    rid = np.zeros(2, np.int64)
    liquidate = kw.get("liquidate", False)
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate else np.full(K, E.SLIP_MAJOR)
    lastb = V.last_bars(P) if liquidate else np.full(K, -1, np.int64)
    cd_add = kw.get("cd_add", 0); t0 = kw.get("t0")
    i0 = 0 if t0 is None else int(np.searchsorted(idx.values, np.datetime64(t0)))
    f = lambda v, dt=np.float64: np.array(v, dt)
    cdw = f([3, 3], np.int64) + cd_add
    dep = f([d["df"] or .14, d["ds"] or .14])
    eq, lg, inv = F.fk(O, P["h"].values, P["l"].values, P["cf"].values, gm, refs, rid, lastb, dep, tpS,
                       f([96, 96], np.int64), np.maximum(cdw, 0), np.maximum(cdw, 0), f([1.0, 1.0]), 99, 1, float(E.EPS),
                       float(E.FEE), sl, float(kw.get("pen", 0.0)), float(kw.get("pens", 0.0)), bool(kw.get("mtp_now", False)),
                       i0, 0.0, 0.0, 0.0, 720, 168, 0, 0)
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=V.LOGC), pd.Series(inv, idx)


DPER = (("d", V.T21, V.T24),)


def scores(d):
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=F.PEN, pens=F.PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = frun(P, idx, d, **kw)
        per = (("d", F.LATE, V.T24),) if tag == "ls" else DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n": out["n_tpm"] = r["d_tpm"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    return out


def ok_s(s, ver=("n", "w")):
    return all(s[f"{t}_wr"] >= 0.925 and s[f"{t}_dd"] > -0.644 - 1e-9 for t in ver) and s["n_mo"] > 0.1317 and s["w_mo"] > 0.0979


def evaluate(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress: kw.update(pen=F.PEN, pens=F.PEN)
    if stress: kw.update(mtp_now=True, cd_add=1, liquidate=True)
    e, lg, inv = frun(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
