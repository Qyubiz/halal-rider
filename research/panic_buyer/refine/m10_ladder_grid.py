"""m10 ladder: design grids on 2021-23 ONLY (the panel is cut at 2024-01-01; nothing later is loaded or printed).

Every design runs twice: normal, and stressed (bot timing: market-buy take-profit at once and cooldown +1h; every buy
and sell fill 0.2% worse). Win rate is strict (a win = a take-profit exit above cost, or a position still open at the
cut marked above cost; time stops, max-loss exits are losses).

Stage 1 (structure, 384 designs): legs x holdings x take-profit x exit, cooldown 3h, crash guard 35%, reserve mode.
Stages 2-3 (robust selection over six perturbed versions of 2021-23) are in m10_ladder_grid2.py.

Usage: PB_DATA=... PB_OUT=... python3 m10_ladder_grid.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import m10_ladder_lib as M

LEGS1 = {
    "1x14": ((0.14, 1),),
    "2x12/20": ((0.12, 1), (0.20, 1)),
    "2x14/22": ((0.14, 1), (0.22, 1)),
    "2x12/20 w1:2": ((0.12, 1), (0.20, 2)),
    "3x12/18/24": ((0.12, 1), (0.18, 1), (0.24, 1)),
    "3x12/18/25 w2:3:5": ((0.12, 2), (0.18, 3), (0.25, 5)),
    "4x10/15/20/25": ((0.10, 1), (0.15, 1), (0.20, 1), (0.25, 1)),
    "5x10/15/20/25/30": ((0.10, 1), (0.15, 1), (0.20, 1), (0.25, 1), (0.30, 1)),
}
HOLD1 = ((1, 1.0), (2, 0.5), (3, 1 / 3))
TP1 = (0.01, 0.015, 0.02, 0.03)
EXIT1 = ((96, 0.0), (336, 0.0), (720, 0.0), (720, 0.25))


def stage1():
    G = []
    for lk, legs in LEGS1.items():
        for H, f in HOLD1:
            for tp in TP1:
                for hm, ml in EXIT1:
                    G.append(dict(legs=legs, H=H, fmax=f, tp=tp, hm=hm, ml=ml, cd=3, guard=0.35))
    return G


def run_grid(G, tag):
    rows = []
    t0 = time.time()
    for i, spec in enumerate(G):
        r = M.evaluate_design(spec)
        rs = M.evaluate_design(spec, stress=True)
        row = {"name": M.name(spec), **{k: (str(v) if k == "legs" else v) for k, v in spec.items()}, **r,
               **{"x_" + k: v for k, v in rs.items()}}
        rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(G)} designs, {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(M.OUT, f"grid{tag}.csv"), index=False)
    return R


def show(R, title, n=25, sort="d_mo"):
    print(f"\n{title}")
    print(f"{'design':72s} {'WR':>5s} {'WR+':>5s} {'%/mo':>6s} {'drop':>5s} {'wt%':>5s} {'tr/mo':>5s} {'hold':>5s} "
          f"{'hmax':>5s} {'ts':>3s} {'ml':>3s} {'2022':>5s} | stressed WR %/mo drop")
    for _, r in R.sort_values(sort, ascending=False).head(n).iterrows():
        print(f"{r['name'][:72]:72s} {r.d_wr*100:5.1f} {r.d_wr_pos*100:5.1f} {r.d_mo*100:+6.2f} {r.d_dd*100:5.0f} "
              f"{r.d_wt*100:+5.0f} {r.d_tpm:5.1f} {r.d_hold:5.1f} {r.d_hold_max:5.0f} {r.d_ts:3d} {r.d_mlx:3d} "
              f"{r.y2022*100:+5.0f} | {r.x_d_wr*100:5.1f} {r.x_d_mo*100:+6.2f} {r.x_d_dd*100:4.0f}")


def frontier(R):
    print("\nFRONTIER 2021-23: best %/month at each strict win-rate floor (normal) | same floor on stressed WR too")
    for wr in (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99):
        for cap in (-1.0, -0.65, -0.50):
            Q = R[(R.d_wr >= wr) & (R.d_dd >= cap)]
            Qs = Q[Q.x_d_wr >= wr - 0.02]
            a = Q.sort_values("d_mo", ascending=False).head(1)
            b = Qs.sort_values("d_mo", ascending=False).head(1)
            fa = (f"{a.iloc[0]['name'][:60]} {a.iloc[0].d_mo*100:+.2f}%/mo dd {a.iloc[0].d_dd*100:.0f}% wr "
                  f"{a.iloc[0].d_wr*100:.1f}" if len(a) else "none")
            fb = (f"{b.iloc[0].d_mo*100:+.2f}%/mo (stressed {b.iloc[0].x_d_mo*100:+.2f}, wr {b.iloc[0].x_d_wr*100:.1f})"
                  if len(b) else "none")
            print(f"  WR>={wr*100:.0f}% drop>={cap*100:.0f}%: {fa} | {fb}")


if __name__ == "__main__":
    st = "1"
    R = run_grid(stage1(), "1")
    show(R, f"STAGE {st}: top 25 by 2021-23 %/month")
    show(R[R.d_wr >= 0.96], f"STAGE {st}: top 25 by 2021-23 %/month with strict WR >= 96%")
    show(R[R.d_wr >= 0.98], f"STAGE {st}: top 15 with strict WR >= 98%", 15)
    frontier(R)
