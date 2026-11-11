"""Adversarial check, part 2: does the SAME pre-registered rule (2021-23 only: 2022 >= 0%, 2021-23 drop >= -50%,
highest 2021-23 CAGR) still choose a 5%/month setting when the 2021-23 selection data is made more realistic?
  a. coin list without survivorship: LUNA/FTT/SRM included in 2021-23 (a 2021 trader would have traded them)
  b. 2021-23 simulated with the real bot's timing (mtp_now, cooldown + 1h)
  c. 2021-23 with fills 0.2% worse
  d. all three
  e. the broader 1,056-setting universe (pick grid + 960 grid) instead of the hindsight-placed 192
For each: the pick, its 2024-26 (on the normal 19-coin data, backtest timing, and with bot timing).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hr_pick as H  # noqa: E402
import study  # noqa: E402

OUT = study.OUT


def rule(D, max_dd=-0.50):
    return D[(D.y2022 >= 0.0) & (D.d_dd >= max_dd)].sort_values(["d_cagr", "d_dd"], ascending=[False, False])


def show(tag, grid, dead, over, cache):
    rows = []
    for s in grid:
        r = H.run(s, dead=dead, **over(s))[0]
        rows.append(r)
    R = pd.DataFrame(rows)
    D = R[H.DESIGN_COLS]
    print(f"\n######## {tag}: {len(R)} settings")
    for lim in (-0.50, -0.45, -0.40):
        ok = rule(D, lim)
        if not len(ok):
            print(f"  limit {lim*100:.0f}%: nothing qualifies"); continue
        p = ok.iloc[0]
        s = (float(p.depth), int(p.N), float(p.fmax), float(p.tp), int(p.hm), int(p.cd))
        if s not in cache:
            a = H.run(s)[0]
            b = H.run(s, mtp_now=True, cd=s[5] + 1)[0]
            cache[s] = (a, b)
        a, b = cache[s]
        top = ok.head(10)
        hits = []
        for _, q in top.iterrows():
            t = (float(q.depth), int(q.N), float(q.fmax), float(q.tp), int(q.hm), int(q.cd))
            if t not in cache:
                cache[t] = (H.run(t)[0], H.run(t, mtp_now=True, cd=t[5] + 1)[0])
            hits.append(cache[t][0]["t_mo"])
        hits = np.array(hits)
        print(f"  limit {lim*100:.0f}%: {len(ok)} qualify; pick {H.name(s)} (its 21-23 here: x{p.d_mult:.2f} DD {p.d_dd*100:.1f}% "
              f"2022 {p.y2022*100:+.0f}%) -> 24-26 normal data: {a['t_mo']*100:+.2f}%/mo DD {a['t_dd']*100:.0f}%; with bot timing "
              f"{b['t_mo']*100:+.2f}%/mo DD {b['t_dd']*100:.0f}%  {'>=5%' if a['t_mo'] >= 0.05 else 'MISSES 5%'}; "
              f"top-10 reaching 5%: {int((hits >= 0.05).sum())}/{len(hits)}, median {np.median(hits)*100:.2f}%/mo")
    return R


def main():
    cache = {}
    none = lambda s: {}  # noqa: E731
    show("a. +LUNA/FTT/SRM in the 2021-23 data", H.PICK_GRID, True, none, cache)
    show("b. real-bot timing in 2021-23 (mtp_now, cd+1)", H.PICK_GRID, False, lambda s: dict(mtp_now=True, cd=s[5] + 1), cache)
    show("c. fills 0.2% worse in 2021-23", H.PICK_GRID, False, lambda s: dict(pen=0.002), cache)
    show("d. all three in 2021-23", H.PICK_GRID, True, lambda s: dict(pen=0.002, mtp_now=True, cd=s[5] + 1), cache)
    uni = list(dict.fromkeys(H.PICK_GRID + H.G960))
    show("e. broader universe (pick grid + 960 grid), normal data", uni, False, none, cache)
    show("f. broader universe, +LUNA/FTT/SRM and real-bot timing in 2021-23", uni, True,
         lambda s: dict(mtp_now=True, cd=s[5] + 1), cache)


if __name__ == "__main__":
    main()
