"""m17 bigtp: bigger take-profits on deep dips (10% mode family). Single-step exit via m11_fusion_lib.fk (fixed or
range-scaled tp, calm-coin filter, guard); two-step exit (sell `part` at tp, rest at tp2) via ens_lib.ekernel on the
same V.market panel (vol filter passed as entry mask). Everything else as 90% mode (1 coin x 100%, 24h-high bid)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402
import m10_velocity_lib as V  # noqa: E402
import ens_lib as EN  # noqa: E402

PEN = F.PEN


def D(depth=0.14, tp=0.025, hm=96, cd=3, guard=0.35, vol=0.0, part=1.0, tp2=0.0):
    return dict(depth=depth, tp=tp, hm=hm, cd=cd, guard=guard, vol=vol, part=part, tp2=tp2)


def run(P, idx, d, pen=0.0, pens=0.0, mtp_now=False, cd_add=0, liquidate=False, t0=None):
    """-> (equity, log with V.LOGC columns, inv)"""
    if d["part"] >= 1.0:
        fd = dict(tiers=[dict(ref=24, depth=d["depth"], tp=d["tp"], hm=d["hm"], cdw=d["cd"], cdl=d["cd"], vol=d["vol"])],
                  guard=d["guard"])
        return F.frun(P, idx, fd, pen=pen, pens=pens, mtp_now=mtp_now, cd_add=cd_add, liquidate=liquidate, t0=t0)
    sv = dict(depth=d["depth"], N=1, fmax=1.0, tp=d["tp"], part=d["part"], tp2=d["tp2"], hm=d["hm"], cd=d["cd"])
    sm = F.vol_mask(P, d["vol"]) if d["vol"] > 0 else None
    e, lg, _, _ = EN.erun(P, idx, [sv], guard=d["guard"], pen=pen, pens=pens, mtp_now=mtp_now, cd_add=cd_add,
                          liquidate=liquidate, smult=sm)
    lg = lg.rename(columns={"sl": "tier"})
    return e, lg, pd.Series(np.zeros(len(idx)), idx)


def scores(d):
    """2021-23 only: normal, worse fills, cooldown -1/+1, bot timing, start 2 weeks later (two-step: later start is
    measured on the same run from 2021-01-15, equity re-based)."""
    syms, idx, P = V.market(False)
    out = {}
    runs = {"n": {}, "w": dict(pen=PEN, pens=PEN), "cm": dict(cd_add=-1), "cp": dict(cd_add=1),
            "bt": dict(mtp_now=True, cd_add=1), "ls": dict(t0=F.LATE)}
    for tag, kw in runs.items():
        e, lg, inv = run(P, idx, d, **kw)
        per = (("d", F.LATE, V.T24),) if tag == "ls" else F.DPER
        r = V.summarize(e, lg, inv, idx, per)
        out[f"{tag}_wr"], out[f"{tag}_mo"], out[f"{tag}_dd"] = r["d_wr"], r["d_mo"], r["d_dd"]
        if tag == "n":
            out["n_tpm"] = r["d_tpm"]
    out["robust"] = float(np.mean([out[f"{t}_mo"] for t in ("n", "cm", "cp", "bt", "ls")]))
    out["ddmin"] = float(min(out[f"{t}_dd"] for t in ("n", "w", "cm", "cp", "bt", "ls")))
    return out


def full(d, worse=False, stress=False, holdout=False):
    dead = stress
    syms, idx, P = V.market_holdout(dead) if holdout else V.market(dead)
    kw = {}
    if worse or stress:
        kw.update(pen=PEN, pens=PEN)
    if stress:
        kw.update(mtp_now=True, cd_add=1)
    if dead:
        kw["liquidate"] = True
    e, lg, inv = run(P, idx, d, **kw)
    return V.summarize(e, lg, inv, idx, V.HOLD if holdout else V.MAIN), e, lg
