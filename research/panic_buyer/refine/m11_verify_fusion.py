"""Independent check of the m11 "fusion" pick, run after its one look (so everything here is HINDSIGHT and is used
only to judge robustness, never to re-pick).

The pick used a 40% crash guard and lost the whole account on LUNA in the stressed 2021-23 run. This script runs it
with the 35% guard of 90% mode, then every nearby version of its second trigger (6h high +-2h, 9-11% dip,
+1.5-2.5%) and of its calm-coin filter (6-8%), and counts how many beat 90% mode on win rate and on %/month in
each period, with normal fills, every fill 0.2% worse and the stressed run (bot timing, worse fills, LUNA/FTT/SRM).

Usage (from research/panic_buyer/refine): PB_DATA=... PB_HOLDOUT=... PB_OUT=.../pb_out_m11/fusion python3 m11_verify_fusion.py
"""
import copy
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_fusion_lib as F  # noqa: E402

RNG = ("rng", 0.03, 0.5, 0.01, 0.06)
A = dict(ref=24, depth=0.14, tp=RNG, hm=96, cdw=0, cdl=3, fs=1.0, vol=0.07)
B = dict(ref=6, depth=0.10, tp=0.02, hm=48, cdw=0, cdl=3, fs=1.0, vol=0.0)
PICK35 = dict(F.OFFRISK, guard=0.35, tiers=[A, B])
PERIODS = ("t", "s", "h")
LAB = {"d": "2021-23", "t": "2024-26", "s": "since 2025", "h": "2018-20"}


def runs(d):
    out = {}
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
        r, _, _ = F.evaluate(d, **kw)
        rh, _, _ = F.evaluate(d, holdout=True, **kw)
        r.update(rh)
        out[lab] = r
    return out


def cell(r, p):
    return f"{r[f'{p}_wr']*100:4.1f} {r[f'{p}_mo']*100:+5.1f} {r[f'{p}_dd']*100:4.0f}"


def show(name, R):
    print(f"| {name} | " + " | ".join(cell(R['normal'], p) for p in "dtsh") + " | "
          + " | ".join(cell(R['worse'], p) for p in "tsh") + " | " + " | ".join(cell(R['stressed'], p) for p in "dt")
          + " |", flush=True)


def main():
    print("| design | 2021-23 | 2024-26 | since 2025 | 2018-20 | worse 2024-26 | worse since 2025 | worse 2018-20 "
          "| stressed 2021-23 | stressed 2024-26 |\n|---|---|---|---|---|---|---|---|---|---|")
    base = runs(F.MODE90)
    show("90% mode", base)
    show("fusion pick, guard 35%", runs(PICK35))
    res = []
    for ref, dep, tp, vol in itertools.product((4, 6, 8), (0.09, 0.10, 0.11), (0.015, 0.02, 0.025), (0.06, 0.07, 0.08)):
        d = copy.deepcopy(PICK35)
        d["tiers"][0]["vol"] = vol
        d["tiers"][1].update(ref=ref, depth=dep, tp=tp)
        R = runs(d)
        res.append(R)
        show(f"B {ref}h -{dep*100:.0f}% +{tp*100:.1f}%, calm {vol*100:.0f}%", R)
    n = len(res)
    print(f"\nShare of the {n} neighbours that beat 90% mode (win rate / %/month / both):")
    for lab in ("normal", "worse", "stressed"):
        for p in (("d", "t") if lab == "stressed" else "dtsh"):
            w = np.mean([r[lab][f"{p}_wr"] > base[lab][f"{p}_wr"] for r in res])
            m = np.mean([r[lab][f"{p}_mo"] > base[lab][f"{p}_mo"] for r in res])
            both = np.mean([(r[lab][f"{p}_wr"] > base[lab][f"{p}_wr"]) and (r[lab][f"{p}_mo"] > base[lab][f"{p}_mo"])
                            for r in res])
            med = np.median([r[lab][f"{p}_mo"] for r in res]) * 100
            print(f"  {lab:8s} {LAB[p]:10s} {w*100:3.0f}% / {m*100:3.0f}% / {both*100:3.0f}%   median %/month {med:+.1f}"
                  f" (90% mode {base[lab][f'{p}_mo']*100:+.1f})")
    wipe = sum(r["stressed"]["d_dd"] < -0.95 for r in res)
    print(f"\nNeighbours that lose 95%+ in the stressed 2021-23 run (LUNA/FTT/SRM): {wipe} of {n}")


if __name__ == "__main__":
    main()
