"""m13 "split": 90% mode's one all-money bid split into 2-3 fully funded resting bids on the SAME coin.
Engine = m10_ladder_lib.lkernel (reserve mode: the whole account is locked in the resting bids when leg 1 fills;
take-profit = limit sell at +tp over the AVERAGE filled price, re-placed at the next hourly check after each extra
fill; unfilled lower bids cancelled when the take-profit fills or the 96h time stop (from first fill) hits; 3h
cooldown; 35% crash guard; H=1 coin at a time; no max-loss, no break-even drop). With one leg it equals 90% mode
(m13_split_check.py). Wrapper adds 'start 2 weeks later' (entry mask zero before t0) and the stress/worse switches.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_m13/split"))
import engine as E  # noqa: E402
import m10_ladder_lib as M  # noqa: E402

T21, T24, T25 = M.T_D0, M.T_T0, M.T_S0
H18, H21 = pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")
LATE = pd.Timestamp("2021-01-15")
PEN = 0.002
PER = {"d": (T21, T24), "t": (T24, None), "s": (T25, None)}
HPER = {"h": (H18, H21)}


def srun(P, idx, legs, tp=0.025, hm=96, cd=3, guard=0.35, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False,
         t0=None):
    legs = tuple(legs)
    if t0 is None:
        return M.lrun(P, idx, legs=legs, H=1, N=1, fmax=1.0, tp=tp, hm=hm, cd=cd + cd_add, guard=guard, pen=pen,
                      pens=pens, mtp_now=mtp_now, liquidate=liquidate)
    dl = np.array([d for d, _ in legs]); wl = np.array([w for _, w in legs], float); wl /= wl.sum()
    O = P["o"].values; n, K = O.shape
    sl = np.array([E.SLIP_MAJOR if s in E.H19 else E.SLIP_OTHER for s in P["o"].columns]) if liquidate \
        else np.full(K, E.SLIP_MAJOR)
    gm = np.ascontiguousarray(M.guard_mask(P, guard)).copy()
    gm[:int(np.searchsorted(idx.values, np.datetime64(t0)))] = 0.0
    eq, lg = M.lkernel(O, P["h"].values, P["l"].values, P["cf"].values, P["ref"].values, gm, dl, wl, 1, 1, 1.0,
                       float(tp), int(hm), int(cd + cd_add), float(E.EPS), float(E.FEE), sl, float(pen), float(pens), 1,
                       bool(mtp_now), 0, 0.0, 0, 0, 0.003, M.last_bars(P, liquidate))
    return pd.Series(eq, idx), pd.DataFrame(lg, columns=M.LOGC)


def legs_of(d1, d2, f, d3=None, w3=0.0):
    if d3 is None:
        return ((d1, f), (d2, 1 - f))
    return ((d1, f * (1 - w3)), (d2, (1 - f) * (1 - w3)), (d3, w3))


def summ(e, lg, idx, a, b):
    r = M.stats(e, lg, idx, a, b)
    return dict(wr=r["wr_pos"], mo=r["mo"], dd=r["dd"], wt=r["wt"], tpm=r["tpm"], n=r["n"], mult=r["mult"])


VERS = {"n": {}, "cm": dict(cd_add=-1), "cp": dict(cd_add=1), "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=LATE)}


def design_scores(legs, tp, cd=3, hm=96):
    """2021-23 only (panel cut at 2024-01-01): 5 nearby versions x (normal fills, every fill 0.2% worse)."""
    syms, idx, P = M.market(False, end=T24)
    out = {}
    for tag, kw in VERS.items():
        for fl, fk in (("", {}), ("w", dict(pen=PEN, pens=PEN))):
            e, lg = srun(P, idx, legs, tp=tp, hm=hm, cd=cd, **kw, **fk)
            a = LATE if tag == "ls" else T21
            r = summ(e, lg, idx, a, T24)
            for k in ("wr", "mo", "dd"):
                out[f"{fl}{tag}_{k}"] = r[k]
    vs = list(VERS)
    for fl in ("", "w"):
        out[f"{fl}mo"] = float(np.mean([out[f"{fl}{t}_mo"] for t in vs]))
        out[f"{fl}wr"] = float(np.mean([out[f"{fl}{t}_wr"] for t in vs]))
        out[f"{fl}dd"] = float(np.median([out[f"{fl}{t}_dd"] for t in vs]))
    return out


def full(legs, tp, cd=3, hm=96):
    """Final look: dict[(run, period)] -> metrics. Runs: normal, worse, stress; periods d,t,s (main) and h (holdout)."""
    res = {}
    for run in ("normal", "worse", "stress"):
        kw = {}
        if run == "worse":
            kw = dict(pen=PEN, pens=PEN)
        if run == "stress":
            kw = dict(pen=PEN, pens=PEN, mtp_now=True, cd_add=1, liquidate=True)
        syms, idx, P = M.market(run == "stress")
        e, lg = srun(P, idx, legs, tp=tp, hm=hm, cd=cd, **kw)
        for p, (a, b) in PER.items():
            res[(run, p)] = summ(e, lg, idx, a, b)
        if run != "stress":
            syms, idx, P = M.holdout_market(False)
            e, lg = srun(P, idx, legs, tp=tp, hm=hm, cd=cd, **kw)
            res[(run, "h")] = summ(e, lg, idx, H18, H21)
    return res


def passes(res):
    """Pass-rule items 1-3 (4 needs stress; reported separately)."""
    n = lambda p: res[("normal", p)]; w = lambda p: res[("worse", p)]
    i1 = n("t")["mo"] * 100 > 6.89 and n("t")["wr"] >= .925 and n("t")["dd"] >= -.35
    i2 = n("h")["mo"] * 100 > 10.82 and n("h")["wr"] >= .925 and n("h")["dd"] >= -.663
    i3 = w("t")["mo"] * 100 > 5.31 and w("h")["mo"] * 100 > 7.41 and w("t")["wr"] >= .92 and w("h")["wr"] >= .92
    return i1, i2, i3
