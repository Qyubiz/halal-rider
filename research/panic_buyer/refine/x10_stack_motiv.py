"""Lever "stack", motivation on 2021-23 ONLY (prints no 2024+ number; the kernel is causal, so 2021-23 results do
not depend on later candles). One booster at a time on two concentrated bases, then a few pairs, to set the
grid ranges in x10_stack_grid.py. Columns: normal run (19 coins) and stressed run (+LUNA/FTT/SRM, every fill 0.2%
worse on buys and sells), 2021-23 %/month, worst drop, 2022, and the worst single trade as % of the account.
Log: $PB_OUT/motiv.log
"""
import numpy as np

import x10_stack_lib as X

BASES = {"B2: N2 cap50 tp3%": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=24),
         "B1: N1 cap100 tp3%": dict(depth=0.12, N=1, fmax=1.0, tp=0.03, hm=48, cd=24)}
MODS = {"(base)": {},
        "cd 0": dict(cd=0), "cd 2": dict(cd=2), "cd 6": dict(cd=6), "cd 12": dict(cd=12),
        "tpwait 2": dict(tpwait=2), "tpwait 4": dict(tpwait=4), "tpwait 6": dict(tpwait=6),
        "no resting tp": dict(tprest=False), "no resting tp, tpwait 3": dict(tprest=False, tpwait=3),
        "ladder 50% / 16%": dict(w=0.5, d2=0.16), "ladder 50% / 18%": dict(w=0.5, d2=0.18),
        "ladder 50% / 20%": dict(w=0.5, d2=0.20), "ladder 67% / 18%": dict(w=2 / 3, d2=0.18),
        "tpf .10 market buys": dict(tpf=0.10, tpfm=1), "tpf .20 market buys": dict(tpf=0.20, tpfm=1),
        "tpf .30 market buys": dict(tpf=0.30, tpfm=1),
        "BTC -3%: 1 bid x 100%": dict(bx=0.03, bN=1, bf=1.0), "BTC -5%: 1 bid x 100%": dict(bx=0.05, bN=1, bf=1.0),
        "tp 2%": dict(tp=0.02), "tp 5%": dict(tp=0.05), "hm 96": dict(hm=96), "hm 24": dict(hm=24),
        "depth 10%": dict(depth=0.10), "depth 15%": dict(depth=0.15)}
PAIRS = {"cd 2 + tpwait 4": dict(cd=2, tpwait=4), "cd 2 + ladder 50/18": dict(cd=2, w=0.5, d2=0.18),
         "tpwait 4 + ladder 50/18": dict(tpwait=4, w=0.5, d2=0.18),
         "cd 2 + tpwait 4 + ladder 50/18": dict(cd=2, tpwait=4, w=0.5, d2=0.18),
         "cd 2 + tpwait 4 + ladder 50/18 + tpf .2 mkt": dict(cd=2, tpwait=4, w=0.5, d2=0.18, tpf=0.2, tpfm=1),
         "cd 2 + no rest tpw3 + ladder 50/18": dict(cd=2, tprest=False, tpwait=3, w=0.5, d2=0.18)}


def dl(r):
    return (f"{r['d_mo']*100:+6.2f}%/mo x{r['d_mult']:8.2f} DD {r['d_dd']*100:5.1f}% 2022 {r['y2022']*100:+5.0f}% "
            f"wt {r['worst_acct_d']*100:+5.1f}%")


if __name__ == "__main__":
    print("2021-23 ONLY. normal (19 coins) | stressed (+LUNA/FTT/SRM, fills 0.2% worse both sides)")
    r, _, _ = X.evaluate({})
    rs, _, _ = X.evaluate({}, stress=True)
    print(f"{'LIVE settings':52s} {dl(r)} | {dl(rs)}")
    for bn, b in BASES.items():
        print(f"\n== {bn}")
        for mn, m in {**MODS, **PAIRS}.items():
            s = {**b, **m}
            r, _, _ = X.evaluate(s)
            rs, _, _ = X.evaluate(s, stress=True)
            print(f"  {mn:50s} {dl(r)} | {dl(rs)}", flush=True)
