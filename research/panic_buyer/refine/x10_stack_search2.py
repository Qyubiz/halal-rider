"""Lever "stack", two follow-ups to x10_stack_ceiling.py (same factor space, same design encoding).

(A) HONEST WALK-FORWARD OF THE SEARCH ITSELF (pre-registered here, before running): the same coordinate ascent,
    but scored on 2021-23 only. Score = stressed 2021-23 %/month (+LUNA/FTT/SRM, every fill 0.2% worse), designs
    with normal 2021-23 drop < -50% or stressed 2021-23 drop < -55% are infeasible (the main grid's eligibility).
    Same three starts; the best end point by that score is the pick; 2024-26 and since-2025 printed once.
    This shows what an automated stack search would have delivered without hindsight.
(B) HINDSIGHT PAIR SEARCH (not a pick): every pair of factors x every pair of values around the ceiling winner
    (N1x100 cd24 tp3 rest3 lad off tpf0.2 btc0 hm48 d12), scored on normal 2024-26 %/month, to see whether two
    simultaneous changes reach 10%/month that one-at-a-time ascent missed.
Logs: $PB_OUT/search2.log; tables search2_walkforward.csv, search2_pairs.csv
"""
import itertools
import json
import os

import numpy as np
import pandas as pd

import x10_stack_lib as X
from x10_stack_ceiling import FACT, name, to_spec

_c = {}


def ev2(c):
    k = json.dumps(c, sort_keys=True)
    if k not in _c:
        r, _, _ = X.evaluate(to_spec(c))
        rs, _, _ = X.evaluate(to_spec(c), stress=True)
        _c[k] = (r, rs)
    return _c[k]


def score_design(c):
    r, rs = ev2(c)
    if r["d_dd"] < -0.50 or rs["d_dd"] < -0.55:
        return -np.inf
    return rs["d_mo"]


def ascend_design(c):
    best = dict(c); bv = score_design(best)
    for p in range(6):
        changed = False
        for f, vals in FACT.items():
            for v in vals:
                t = dict(best); t[f] = v
                s = score_design(t)
                if s > bv + 1e-12:
                    best, bv, changed = t, s, True
        print(f"    pass {p+1}: {name(best)} -> stressed 2021-23 {bv*100:+.2f}%/mo ({len(_c)} designs)", flush=True)
        if not changed:
            break
    return best, bv


if __name__ == "__main__":
    base = dict(siz="N1x100", cd=24, tp=0.03, exit="rest1", lad="off", tpf=0.0, btc=0.0, hm=48, depth=0.12)
    starts = {"N1 x 100% +3% cd24": base, "N2 x 50% +3% cd24": {**base, "siz": "N2x50"},
              "grid pick": {**base, "siz": "N2x50", "cd": 6, "tp": 0.05, "lad": "50/18", "tpf": 0.2}}
    print("######## (A) honest walk-forward of the coordinate ascent (scored on 2021-23 only)")
    ends = []
    for sn, c in starts.items():
        print(f"== start {sn}: score {score_design(c)*100 if np.isfinite(score_design(c)) else -np.inf:+.2f}")
        ends.append(ascend_design(c))
    pick, pv = max(ends, key=lambda x: x[1])
    r, rs = ev2(pick)
    print(f"PICK (2021-23 only): {name(pick)}")
    print(X.fmt4(name(pick), r, rs))
    W = pd.DataFrame([{"design": k, **{f"n_{a}": b for a, b in v[0].items()}, **{f"s_{a}": b for a, b in v[1].items()}}
                      for k, v in _c.items()])
    W.to_csv(os.path.join(X.OUT, "search2_walkforward.csv"), index=False)

    print("\n######## (B) HINDSIGHT pair search around the ceiling winner (normal 2024-26 %/month)")
    win = dict(siz="N1x100", cd=24, tp=0.03, exit="rest3", lad="off", tpf=0.2, btc=0.0, hm=48, depth=0.12)
    rows = []
    for f1, f2 in itertools.combinations(FACT, 2):
        for v1 in FACT[f1]:
            for v2 in FACT[f2]:
                if v1 == win[f1] or v2 == win[f2]:
                    continue
                t = dict(win); t[f1] = v1; t[f2] = v2
                r0, _, _ = X.evaluate(to_spec(t))
                rows.append({"design": name(t), "f1": f1, "v1": v1, "f2": f2, "v2": v2, **r0})
    B = pd.DataFrame(rows)
    B.to_csv(os.path.join(X.OUT, "search2_pairs.csv"), index=False)
    print(f"{len(B)} pair designs: 2024-26 %/month max {B.t_mo.max()*100:.2f}, median {B.t_mo.median()*100:.2f}, "
          f">= 10%/mo: {int((B.t_mo >= 0.10).sum())}, >= 9%/mo: {int((B.t_mo >= 0.09).sum())}")
    for _, q in B.sort_values("t_mo", ascending=False).head(8).iterrows():
        print(f"  {q.design:62s} 21-23 {q.d_mo*100:+6.2f}%/mo DD {q.d_dd*100:5.1f}% | 24-26 {q.t_mo*100:+6.2f}%/mo DD "
              f"{q.t_dd*100:5.1f}% | 25-26 {q.s_mo*100:+6.2f}%/mo DD {q.s_dd*100:5.1f}%")
    best = B.sort_values("t_mo", ascending=False).iloc[0]
    t = dict(win); t[best.f1] = best.v1; t[best.f2] = best.v2
    for k in ("cd", "hm"):
        t[k] = int(t[k])
    r, rs = ev2(t)
    print(X.fmt4(f"HINDSIGHT pair-search best: {name(t)}", r, rs))
