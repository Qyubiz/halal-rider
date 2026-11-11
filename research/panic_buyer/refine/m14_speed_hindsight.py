"""HINDSIGHT (no pick was made; nothing passed the pre-registered bars). One look at the best-robust design that failed
only the worse-fill drop bar: A=8h, fast depth 12%, slow depth 14%."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m14_speed_lib as L
def table(c):
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, e, lg = L.evaluate(c, **kw)
        if lab != "stressed": r.update(L.evaluate(c, holdout=True, **kw)[0])
        for p, pl in (("d", "2021-23"), ("t", "2024-26"), ("s", "since25"), ("h", "2018-20")):
            if f"{p}_wr" in r and not (lab == "stressed" and p in ("s", "h")):
                print(f"  {lab:9s} {pl:8s} {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f} {r[f'{p}_dd']*100:6.1f} {r[f'{p}_tpm']:5.1f}")
        if lab == "stressed": print("  stressed d_mult", r["d_mult"])
for c in (L.OFF, dict(A=8, df=.12, ds=.14)):
    print(c); table(c)
