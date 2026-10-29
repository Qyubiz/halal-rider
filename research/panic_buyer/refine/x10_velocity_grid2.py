"""Lever "velocity", round 2, pre-registered grid (x10 study). Written after the round-2 development pass
(x10_velocity_dev2.py, 2021-23 only) and before any 2024-26 number of a round-2 mechanic was seen. Round 2 itself was
started because round 1 missed 10%/month in 2024-26, so it is a second look (a hindsight-informed decision to keep
searching), not a fresh experiment; the choice INSIDE this grid is still made on 2021-23 only. Nothing is changed after the run.

GRID (768 designs; depth 12%, volume-rank slippage, 1-minute engine x10_velocity_lib.vkernel; contains the round-1 pick):
  sizing   N1 fmax 1.00 | N2 fmax 0.50 | N3 fmax 0.33
  cycle    15 min + re-arm after a sale | every minute
  exit     trailing arm 3% trail 0 cap 8% (1 min) | trailing arm 5% trail 0 cap 8% (1 min) | tp 3% next hour | tp 3% after 30 min
  time stop 6h | 12h | 24h | 48h
  flash    off | look 1h depth 8% | look 3h depth 8% | look 3h depth 6%   (flash buys: tp 3% for fixed exits, same trailing)
  re-entry off (cooldown 24h) | cooldown 0 + re-entry at exit -3% for 60 min after a profitable exit
SELECTION RULE (same as round 1; reads 2021-23 columns only): eligible if 19-coin 2021-23 worst drop >= -50%,
22-coin 2021-23 worst drop >= -60% and 19-coin 2022 >= 0%; highest min(19-coin, 22-coin 2021-23 %/month) wins
(tie: shallower 19-coin drop). Then the four reports for the pick, the hindsight best (by 19-coin 2024-26 %/month,
labelled hindsight) and the hindsight best with a 2024-26 drop no deeper than -50%.
"""
import hashlib
import itertools
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_velocity_lib as V  # noqa: E402
import x10_velocity_grid as G1  # noqa: E402  (data, one(), stress())

SIZING = {"N1": dict(N=1, fmax=1.0), "N2": dict(N=2, fmax=0.5), "N3": dict(N=3, fmax=0.33)}
CYCLE = {"15min": dict(cyc=15, rearm=1), "1min": dict(cyc=1, rearm=1)}
EXIT = {"trail a3": dict(xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1),
        "trail a5": dict(xmode=1, arm=0.05, trail=0.0, cap=0.08, delay=1),
        "tp3": dict(tp=0.03, tpf=0.03), "tp3 d30": dict(tp=0.03, tpf=0.03, delay=30)}
HM = {f"hm{h}": dict(hm=h) for h in (6, 12, 24, 48)}
FLASH = {"noflash": dict(), "flash1h8": dict(look2=1, depth2=0.08), "flash3h8": dict(look2=3, depth2=0.08),
         "flash3h6": dict(look2=3, depth2=0.06)}
REENTRY = {"cd24h": dict(cdm=-1, cd=24), "cd0+redip3/60m": dict(cdm=0, redip=0.03, rewin=60)}
GRID = {" | ".join(c): {**SIZING[c[0]], **CYCLE[c[1]], **EXIT[c[2]], **HM[c[3]], **FLASH[c[4]], **REENTRY[c[5]]}
        for c in itertools.product(SIZING, CYCLE, EXIT, HM, FLASH, REENTRY)}
R1PICK = "N2 | 1min | trail a3 | hm48 | noflash | cd24h"
DD19, DD22, MIN2022 = -0.50, -0.60, 0.0


def main():
    sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"x10_velocity_grid2.py sha256 {sha}\n{len(GRID)} designs x 2 panels", flush=True)
    assert R1PICK in GRID
    t0 = time.time()
    with mp.get_context("fork").Pool(2) as pool:
        rows = []
        for i, r in enumerate(pool.imap(G1.one, list(GRID.items()), chunksize=4)):
            rows.append(r)
            if i % 100 == 0:
                print(f"  {i+1}/{len(GRID)} done, {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(V.OUT, "grid2.csv"), index=False)
    print(f"grid done in {time.time()-t0:.0f}s", flush=True)

    D = R[["name", "d_pm", "d_dd", "y2022", "x_d_pm", "x_d_dd"]].copy()       # 2021-23 columns only
    D["score"] = np.minimum(D.d_pm, D.x_d_pm)
    ok = D[(D.d_dd >= DD19) & (D.x_d_dd >= DD22) & (D.y2022 >= MIN2022)].sort_values(["score", "d_dd"], ascending=[False, False])
    print(f"\n######## SELECTION (2021-23 only): {len(ok)} of {len(D)} designs eligible")
    for _, r in ok.head(12).iterrows():
        print(f"  {r['name']:52s} 21-23 {r.d_pm*100:+6.2f}%/mo DD {r.d_dd*100:4.0f}% | +LUNA/FTT/SRM {r.x_d_pm*100:+6.2f}%/mo "
              f"DD {r.x_d_dd*100:4.0f}% | 2022 {r.y2022*100:+4.0f}%")
    r1 = D[D.name == R1PICK].iloc[0]
    print(f"  (round-1 pick in this grid: score {r1.score*100:.2f}%/mo, rank {int((D.score > r1.score).sum()) + 1} of {len(D)} overall)")
    pick = ok.iloc[0]["name"] if len(ok) else None
    print(f"PICK: {pick}")
    out = []
    if pick:
        print(f"\n######## ROUND-2 PICK (2021-23 rule): {pick}")
        out += G1.stress(pick, GRID[pick])
    H = R.sort_values("t_pm", ascending=False)
    print("\n######## HINDSIGHT: top 10 of the round-2 grid by 19-coin 2024-26 %/month")
    for _, r in H.head(10).iterrows():
        print(f"  {r['name']:52s} {V.fmt(r)} | +LUNA/FTT/SRM 21-23 {r.x_d_pm*100:+6.2f}%/mo DD {r.x_d_dd*100:4.0f}%")
    hb = H.iloc[0]["name"]
    print(f"\n######## HINDSIGHT BEST (round 2): {hb}")
    out += G1.stress(hb, GRID[hb])
    H50 = R[R.t_dd >= -0.50].sort_values("t_pm", ascending=False)
    if len(H50) and H50.iloc[0]["name"] != hb:
        print(f"\n######## HINDSIGHT BEST with 2024-26 drop >= -50% (round 2): {H50.iloc[0]['name']}")
        out += G1.stress(H50.iloc[0]["name"], GRID[H50.iloc[0]["name"]])
    pd.DataFrame(out).to_csv(os.path.join(V.OUT, "grid2_stress.csv"), index=False)
    print(f"\nround-2 grid: 2024-26 %/month max {R.t_pm.max()*100:.2f}, median {R.t_pm.median()*100:.2f}; >= 10%: "
          f"{int((R.t_pm >= 0.10).sum())}; >= 8%: {int((R.t_pm >= 0.08).sum())}; since 2025 max {R.s_pm.max()*100:.2f}%/month")
    for ax, d in (("sizing", SIZING), ("cycle", CYCLE), ("exit", EXIT), ("time stop", HM), ("flash", FLASH), ("re-entry", REENTRY)):
        print(f"  by {ax}: " + " | ".join(
            f"{k} 24-26 med {R[R.name.str.split(' [|] ').apply(lambda x: k in x)].t_pm.median()*100:+.2f} "
            f"max {R[R.name.str.split(' [|] ').apply(lambda x: k in x)].t_pm.max()*100:+.2f}" for k in d))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
