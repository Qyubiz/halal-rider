"""wr90 frontier over every grid run so far (grid_*.csv in $PB_OUT): for each win-rate level (2021-23 AND 2024-26 at or
above it), the best design by the fixed selection rule (highest min of the three periods' %/month, ties -> shallower
worst drop), with its stressed version; and the same ranked by the STRESSED min %/month (robust view).
Also: the best design for each worst-drop cap at win rate >= 90%."""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wr90_grid as G  # noqa: E402
import w90_wr90_lib as W  # noqa: E402


def load(tags=None):
    fs = sorted(glob.glob(os.path.join(W.OUT, "grid_*.csv")))
    R = []
    for f in fs:
        t = os.path.basename(f)[5:-4]
        if tags and t not in tags:
            continue
        x = pd.read_csv(f); x["grid"] = t
        R.append(x)
    R = pd.concat(R, ignore_index=True)
    R["fmax"] = R.fmax.round(4)
    R = R.drop_duplicates(subset=G.KEYS)
    R["x_minmo"] = R[["x_d_mo", "x_t_mo", "x_s_mo"]].min(axis=1)
    R["x_mindd"] = R[["x_d_dd", "x_t_dd", "x_s_dd"]].min(axis=1)
    R["x_minwr"] = R[["x_d_wr", "x_t_wr"]].min(axis=1)
    return R


def name(r):
    s = f"N{int(r.N)}x{r.fmax*100:.0f}% d{r.depth*100:g} tp{r.tp*100:g} hm{int(r.hm)} cd{int(r.cd)} g{r.guard*100:g}"
    if r.pmode == 1:
        s += f" PO(be{r.be*100:g} hs{r.hs*100:g} hmax{int(r.hmax)})"
    elif r.hs > 0:
        s += f" stop{r.hs*100:g}"
    return s


def line(r):
    return (f"{name(r):52s} [{r.grid}] WR {r.d_wr*100:4.1f}/{r.t_wr*100:4.1f}/{r.s_wr*100:4.1f}  "
            f"mo {r.d_mo*100:+5.1f}/{r.t_mo*100:+5.1f}/{r.s_mo*100:+5.1f}  dd {r.d_dd*100:3.0f}/{r.t_dd*100:3.0f}/{r.s_dd*100:3.0f}"
            f" | STRESS WR {r.x_d_wr*100:4.1f}/{r.x_t_wr*100:4.1f} mo {r.x_d_mo*100:+5.1f}/{r.x_t_mo*100:+5.1f}/{r.x_s_mo*100:+5.1f}"
            f" dd {r.x_mindd*100:3.0f}")


def main(tags=None):
    R = load(tags)
    print(f"{len(R)} distinct designs from grids {sorted(R.grid.unique())}")
    print("\nFRONTIER by the selection rule (normal runs): best min(%/mo) at each win-rate level (2021-23 and 2024-26)")
    for wr in (0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98):
        Q = G.rank(G.qualify(R, wr))
        if len(Q):
            print(f"WR>={wr*100:4.1f}% ({len(Q):4d}) " + line(Q.iloc[0]))
    print("\nROBUST VIEW: best STRESSED min(%/mo) among designs with normal AND stressed WR >= level")
    for wr in (0.85, 0.88, 0.90, 0.92, 0.94):
        Q = R[(R.minwr >= wr) & (R.x_minwr >= wr)].sort_values(["x_minmo", "x_mindd"], ascending=False)
        if len(Q):
            print(f"WR>={wr*100:4.1f}% ({len(Q):4d}) " + line(Q.iloc[0]))
    print("\nDROP CAPS at WR>=90% (normal): best min(%/mo) with worst drop of the three periods above the cap")
    Q = G.qualify(R, 0.90)
    for cap in (-0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50, -0.60, -1.0):
        x = G.rank(Q[Q.mindd >= cap])
        if len(x):
            print(f"drop >= {cap*100:4.0f}% ({len(x):4d}) " + line(x.iloc[0]))
    print("\nDROP CAPS in BOTH normal and stressed runs, WR>=90% normal: best min(%/mo) (normal) with normal worst drop >= cap"
          " and stressed worst drop >= cap - 5 points")
    for cap in (-0.20, -0.25, -0.30, -0.35, -0.40, -0.45, -0.50, -0.60):
        x = G.rank(Q[(Q.mindd >= cap) & (Q.x_mindd >= cap - 0.05)])
        if len(x):
            print(f"drop >= {cap*100:4.0f}% ({len(x):4d}) " + line(x.iloc[0]))
    print("\nTOP 15 qualifiers (WR>=90%) by the rule")
    for _, r in G.rank(Q).head(15).iterrows():
        print("   " + line(r))
    print("\nTOP 10 qualifiers (WR>=90% normal and stressed) by stressed min %/mo")
    for _, r in R[(R.minwr >= 0.9) & (R.x_minwr >= 0.9)].sort_values(["x_minmo", "x_mindd"], ascending=False).head(10).iterrows():
        print("   " + line(r))
    return R


if __name__ == "__main__":
    main(sys.argv[1:] or None)
