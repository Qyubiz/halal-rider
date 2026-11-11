"""Independent check of the "velocity" finalist (m10_velocity_final.py) before anything goes near the bot.

1. Ablation: which part of the pick makes the gain, the second (12h) trigger or re-arming a coin right after a win?
2. Neighbours: every nearby version of the second trigger (12h/8h/16h high, 12-14% dip, +2.5-3.5%), and nearby
   cooldowns. A real edge should hold for most neighbours, not only the picked one.
Each row: win rate / %/month / worst drop, normal fills and every fill 0.2% worse, for 2021-23, 2024-26,
since 2025 and the 2018-20 holdout, plus the stressed 2024-26 run (bot timing, worse fills, LUNA/FTT/SRM).

Usage (from research/panic_buyer/refine): PB_DATA=... PB_HOLDOUT=... python3 m10_verify_velocity.py
"""
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402

A = dict(ref=24, depth=0.14, tp=0.025, hm=96)


def design(b=None, cdw=3, cdl=3):
    tiers = [dict(A, cdw=cdw, cdl=cdl)]
    if b:
        tiers.append(dict(b, cdw=cdw, cdl=cdl))
    return dict(tiers=tiers, nslots=1, excl=1, dyn=0, guard=0.35)


def row(kw):
    n, _, _, _ = V.evaluate(kw)
    nh, _, _, _ = V.evaluate(kw, holdout=True)
    w, _, _, _ = V.evaluate(kw, worse=True)
    wh, _, _, _ = V.evaluate(kw, holdout=True, worse=True)
    st, _, _, _ = V.evaluate(kw, stress=True)
    n.update(nh)
    w.update(wh)
    return n, w, st


def cell(r, p):
    return f"{r[f'{p}_wr']*100:4.1f} {r[f'{p}_mo']*100:+5.1f} {r[f'{p}_dd']*100:4.0f}"


def show(name, n, w, st):
    print(f"| {name} | " + " | ".join(cell(n, p) for p in "dtsh") + " | "
          + " | ".join(cell(w, p) for p in "tsh") + f" | {cell(st, 't')} |", flush=True)


HEAD = ("| design | 2021-23 | 2024-26 | since 2025 | 2018-20 | worse 2024-26 | worse since 2025 | worse 2018-20 "
        "| stressed 2024-26 |\n|---|---|---|---|---|---|---|---|---|")


def main():
    B = dict(ref=12, depth=0.13, tp=0.03, hm=48)
    print("## 1. Ablation (win % / %/month / worst drop %)\n")
    print(HEAD)
    for name, kw in [("90% mode (cooldown 3h)", design()),
                     ("90% mode, re-arm after a win (cdw 0)", design(cdw=0)),
                     ("+ 12h trigger, cooldown 3h", design(B)),
                     ("PICK: + 12h trigger, re-arm after a win", design(B, cdw=0))]:
        show(name, *row(kw))

    print("\n## 2. Cooldown neighbours of 90% mode and of the pick\n")
    print(HEAD)
    for cdw, cdl in [(0, 2), (0, 4), (1, 3), (2, 3), (4, 4), (6, 6)]:
        show(f"90% mode cdw {cdw} cdl {cdl}", *row(design(cdw=cdw, cdl=cdl)))
        show(f"pick cdw {cdw} cdl {cdl}", *row(design(B, cdw=cdw, cdl=cdl)))

    print("\n## 3. Second-trigger neighbours (re-arm after a win, cdl 3)\n")
    print(HEAD)
    res = []
    for ref, dep, tp, hm in itertools.product((8, 12, 16), (0.12, 0.13, 0.14), (0.025, 0.03, 0.035), (48,)):
        n, w, st = row(design(dict(ref=ref, depth=dep, tp=tp, hm=hm), cdw=0))
        show(f"{ref}h -{dep*100:.0f}% +{tp*100:.1f}% {hm}h", n, w, st)
        res.append((n, w, st))
    base_n, base_w, base_st = row(design())
    print("\nShare of the 27 neighbours that beat 90% mode on %/month (normal / worse fills / stressed):")
    for p, lab in zip("dtsh", ("2021-23", "2024-26", "since 2025", "2018-20")):
        bn = np.mean([r[0][f"{p}_mo"] > base_n[f"{p}_mo"] for r in res])
        bw = np.mean([r[1][f"{p}_mo"] > base_w[f"{p}_mo"] for r in res])
        wr = np.mean([r[0][f"{p}_wr"] >= base_n[f"{p}_wr"] for r in res])
        print(f"  {lab}: {bn*100:.0f}% / {bw*100:.0f}%   (win rate >= 90% mode: {wr*100:.0f}%)")
    bs = np.mean([r[2]["t_mo"] > base_st["t_mo"] for r in res])
    print(f"  stressed 2024-26: {bs*100:.0f}%")


if __name__ == "__main__":
    main()
