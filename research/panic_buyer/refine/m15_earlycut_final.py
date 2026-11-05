"""m15 earlycut ONE look. The pre-written rule gave NO PICK (0 of 132 pass 2021-23 bar 1). Labelled FALLBACK, chosen
after the grid on 2021-23 only: the best 5-version score among designs passing bar 2 (all versions win rate >= 92.5%
and drop no deeper than -64.45%) = A, BTC, coin -10% and BTC -6% within 6h. Not a pick under the rule."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import m15_earlycut_lib as L, m15_earlycut_grid as G, m10_velocity_lib as V
FB = dict(fam=1, mi=0, aN=6, ax=0.10, ay=0.06)
PER = (("d", "2021-23", V.T21, V.T24), ("t", "2024-26", V.T24, None), ("s", "since25", V.T25, None), ("h", "2018-20", V.H18, V.H21))


def table(c, show=True):
    R = {}
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, e, lg = L.evaluate(c, **kw); _, idx, _ = V.market(kw.get("stress", False))
        for p, pl, a, b in PER[:3]:
            r[f"{p}_al"], r[f"{p}_wt2"], _ = L.loss_stats(lg, idx, a, b)
        if lab != "stressed":
            rh, eh, lgh = L.evaluate(c, holdout=True, **kw); _, idxh, _ = V.market_holdout(False)
            rh["h_al"], rh["h_wt2"], _ = L.loss_stats(lgh, idxh, V.H18, V.H21); r.update(rh)
        R[lab] = r
        if show:
            for p, pl, a, b in PER:
                if f"{p}_wr" in r and not (lab == "stressed" and p in ("s", "h")):
                    print(f"  {lab:9s} {pl:8s} WR {r[f'{p}_wr']*100:5.1f} {r[f'{p}_mo']*100:+6.2f}%/mo drop {r[f'{p}_dd']*100:6.1f} "
                          f"{r[f'{p}_tpm']:5.1f} tr/mo | avg loser {r[f'{p}_al']*100:+6.2f}% worst {r[f'{p}_wt2']*100:+6.2f}%")
    return R


def items(R):
    n, w, s = R["normal"], R["worse"], R["stressed"]
    return (n["t_mo"] > .0689 and n["t_wr"] >= .925 and n["t_dd"] >= -.35,
            n["h_mo"] > .1082 and n["h_wr"] >= .925 and n["h_dd"] >= -.663,
            w["t_mo"] > .0531 and w["h_mo"] > .0741 and w["t_wr"] >= .92 and w["h_wr"] >= .92,
            s["t_mo"] > .0549 and s["d_dd"] > -.95)


print("90% mode"); table(L.OFF)
print("FALLBACK (no pick under the rule)", json.dumps(FB)); R = table(FB)
for i, ok in enumerate(items(R), 1):
    print("item", i, "PASS" if ok else "FAIL")
for tag, h in (("main", False), ("hold", True)):
    r, e, lg = L.evaluate(FB, holdout=h)
    e.rename("equity").rename_axis("time").to_csv(os.path.join(L.OUT, f"pick_eq_{tag}.csv"), header=True)
ok = 0; nbs = G.neighbours(FB)
for x in nbs:
    i = items(table(x, False))[:3]; ok += all(i); print(" nb", json.dumps(x), i)
print("neighbours passing 1-3:", ok, "/", len(nbs))
