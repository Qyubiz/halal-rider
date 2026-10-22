"""Lever "stack", HINDSIGHT CEILING: how far can stacking push 2024-26 if the stack is tuned ON 2024-26 itself?

This is deliberately overfit (it chooses on the test period) and is NOT a design to trade. It answers one question:
is 10%/month in 2024-26 reachable by any combination of the stack's boosters on the 19 coins, even with hindsight?

Coordinate ascent on normal 2024-26 %/month from three starts (N1 x 100% +3% cd24; N2 x 50% +3% cd24; the honest
pick of x10_stack_grid.py), over:
    sizing (N, cap) {(1,100%), (2,50%), (2,100%), (3,50%), (3,33%)}; cd {0,2,6,12,24,36,48}; tp {2,2.5,3,3.5,4,5,6%};
    exit {rest tpwait 1,2,3,4; no rest tpwait 1,2,3}; ladder {off, 50/18, 67/16, 67/18, 67/20, 75/20};
    tpf on market buys {0, .1, .2, .3}; BTC rule {none, -3%, -5%, -8%} -> 1 bid x 100%; hm {36,48,60,72,96};
    depth {11,12,13,14,15%}.
Each pass tries every value of every factor (one at a time) and keeps the best; stops when a pass changes nothing.
Then the four reports for each end point, and a +-1 step jitter of the winner (how much of it is noise).
Log: $PB_OUT/ceiling_search.log, table ceiling_search.csv
"""
import json
import os

import numpy as np
import pandas as pd

import x10_stack_lib as X

SIZ = {"N1x100": dict(N=1, fmax=1.0), "N2x50": dict(N=2, fmax=0.5), "N2x100": dict(N=2, fmax=1.0),
       "N3x50": dict(N=3, fmax=0.5), "N3x33": dict(N=3, fmax=1 / 3)}
EXIT = {"rest1": dict(tprest=True, tpwait=1), "rest2": dict(tprest=True, tpwait=2), "rest3": dict(tprest=True, tpwait=3),
        "rest4": dict(tprest=True, tpwait=4), "norest1": dict(tprest=False, tpwait=1),
        "norest2": dict(tprest=False, tpwait=2), "norest3": dict(tprest=False, tpwait=3)}
LAD = {"off": dict(w=1.0, d2=0.0), "50/18": dict(w=0.5, d2=0.18), "67/16": dict(w=2 / 3, d2=0.16),
       "67/18": dict(w=2 / 3, d2=0.18), "67/20": dict(w=2 / 3, d2=0.20), "75/20": dict(w=0.75, d2=0.20)}
FACT = {"siz": list(SIZ), "cd": [0, 2, 6, 12, 24, 36, 48], "tp": [0.02, 0.025, 0.03, 0.035, 0.04, 0.05, 0.06],
        "exit": list(EXIT), "lad": list(LAD), "tpf": [0.0, 0.1, 0.2, 0.3], "btc": [0.0, 0.03, 0.05, 0.08],
        "hm": [36, 48, 60, 72, 96], "depth": [0.11, 0.12, 0.13, 0.14, 0.15]}
_cache = {}


def to_spec(c):
    s = dict(depth=c["depth"], tp=c["tp"], hm=c["hm"], cd=c["cd"], **SIZ[c["siz"]], **EXIT[c["exit"]], **LAD[c["lad"]])
    if c["tpf"] > 0:
        s.update(tpf=c["tpf"], tpfm=1)
    if c["btc"] > 0:
        s.update(bx=c["btc"], bN=1, bf=1.0)
    if s["w"] == 1.0:
        s.pop("w"); s.pop("d2")
    return s


def key(c):
    return json.dumps(c, sort_keys=True)


def ev(c):
    k = key(c)
    if k not in _cache:
        r, _, _ = X.evaluate(to_spec(c))
        _cache[k] = r
    return _cache[k]


def name(c):
    return (f"{c['siz']} cd{c['cd']} tp{c['tp']*100:g} {c['exit']} lad {c['lad']} tpf{c['tpf']:g} btc{c['btc']:g} "
            f"hm{c['hm']} d{c['depth']*100:g}")


def ascend(c):
    best = dict(c); bv = ev(best)["t_mo"]
    for p in range(6):
        changed = False
        for f, vals in FACT.items():
            for v in vals:
                t = dict(best); t[f] = v
                r = ev(t)
                if r["t_mo"] > bv + 1e-12:
                    best, bv, changed = t, r["t_mo"], True
        print(f"    pass {p+1}: {name(best)} -> 2024-26 {bv*100:+.2f}%/mo  (designs evaluated so far {len(_cache)})",
              flush=True)
        if not changed:
            break
    return best


def jitter(c):
    out = []
    for f, vals in FACT.items():
        i = vals.index(c[f])
        for j in (i - 1, i + 1):
            if 0 <= j < len(vals):
                t = dict(c); t[f] = vals[j]
                out.append((f"{f}={vals[j]}", ev(t)))
    return out


if __name__ == "__main__":
    base = dict(siz="N1x100", cd=24, tp=0.03, exit="rest1", lad="off", tpf=0.0, btc=0.0, hm=48, depth=0.12)
    starts = {"N1 x 100% +3% cd24": base,
              "N2 x 50% +3% cd24": {**base, "siz": "N2x50"},
              "honest pick (N2 x 50%, cd6, ladder 50/18, +5%, tpf .2 mkt)": {**base, "siz": "N2x50", "cd": 6, "tp": 0.05,
                                                                            "lad": "50/18", "tpf": 0.2}}
    ends = {}
    for sn, c in starts.items():
        print(f"\n== start: {sn}: 2024-26 {ev(c)['t_mo']*100:+.2f}%/mo")
        ends[sn] = ascend(c)
    print("\n######## END POINTS (hindsight: chosen on 2024-26)")
    for sn, c in ends.items():
        r, _, _ = X.evaluate(to_spec(c))
        rs, _, _ = X.evaluate(to_spec(c), stress=True)
        print(X.fmt4(f"from '{sn}': {name(c)}", r, rs))
        for y in range(2021, 2027):
            print(f"    {y}: normal {r[f'y{y}']*100:+7.1f}%  stressed {rs[f'y{y}']*100:+7.1f}%")
    win = max(ends.values(), key=lambda c: ev(c)["t_mo"])
    print(f"\n######## JITTER of the winner ({name(win)}, 2024-26 {ev(win)['t_mo']*100:+.2f}%/mo): one step per factor")
    J = jitter(win)
    for lab, r in J:
        print(f"  {lab:14s} 21-23 {r['d_mo']*100:+6.2f}%/mo DD {r['d_dd']*100:5.1f}% | 24-26 {r['t_mo']*100:+6.2f}%/mo DD "
              f"{r['t_dd']*100:5.1f}% | 25-26 {r['s_mo']*100:+6.2f}%/mo")
    tj = np.array([r["t_mo"] for _, r in J])
    print(f"  neighbours 2024-26 %/month: median {np.median(tj)*100:.2f}, min {tj.min()*100:.2f}, max {tj.max()*100:.2f}")
    T = pd.DataFrame([{"design": k, **v} for k, v in _cache.items()])
    T.to_csv(os.path.join(X.OUT, "ceiling_search.csv"), index=False)
    print(f"\nall {len(T)} designs evaluated: 2024-26 %/month max {T.t_mo.max()*100:.2f}, >= 10%/mo: {int((T.t_mo >= 0.1).sum())}, "
          f"since-2025 max {T.s_mo.max()*100:.2f}")
