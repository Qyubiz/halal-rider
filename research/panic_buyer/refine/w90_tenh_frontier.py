"""The full ten-hourly frontier over grids 1-6 (all designs found by looking at 2024-26 = HINDSIGHT):
  1. best design (task preference: highest worst-of-three %/month, ties -> shallower drop) at each win-rate level
     (win rate = lower of 2021-23 and 2024-26), with its stressed version;
  2. among designs with >= 90% win rate in both periods: the best at each worst-drop level;
  3. the stressed version of the 40 best designs by worst-of-three: which ones hold up best under stress.
Duplicates (identical results under different names) are counted once."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402
import w90_tenh_report as R  # noqa: E402
import w90_tenh_grid1 as G1  # noqa: E402
import w90_tenh_grid2 as G2  # noqa: E402
import w90_tenh_grid3 as G3  # noqa: E402
import w90_tenh_grid4 as G4  # noqa: E402
import w90_tenh_grid5 as G5  # noqa: E402
import w90_tenh_grid6 as G6  # noqa: E402

out = open(os.path.join(T.OUT, "frontier.log"), "w")


def say(s=""):
    print(s, flush=True)
    out.write(s + "\n"); out.flush()


D, F = {}, []
for tag, mod in (("g1", G1), ("g2", G2), ("g3", G3), ("g4", G4), ("g5", G5), ("g6", G6)):
    d = mod.designs()
    g = pd.read_csv(os.path.join(T.OUT, f"grid{tag[1]}.csv"))
    g["name"] = tag + " " + g["name"]
    D.update({tag + " " + k: v for k, v in d.items()})
    F.append(g)
A = pd.concat(F, ignore_index=True)
n_all = len(A)
A["key"] = A[["d_mo", "t_mo", "s_mo", "d_win", "t_win"]].round(10).astype(str).agg("|".join, axis=1)
A = A.drop_duplicates("key").reset_index(drop=True)
A = R.qualify(A)
say(f"{n_all} designs run in grids 1-6, {len(A)} with distinct results | qualify ten: {int(A.q_ten.sum())} | qualify wr90: "
    f"{int(A.q_wr90.sum())} | max 2021-23 {A.d_mo.max()*100:.2f}%/mo | max 2024-26 {A.t_mo.max()*100:.2f}%/mo | max since 2025 "
    f"{A.s_mo.max()*100:.2f}%/mo | max worst-of-three {A.min_mo.max()*100:.2f}%/mo")
say(f"designs >= 10%/mo in 2024-26: {int((A.t_mo >= 0.10).sum())}; >= 9%: {int((A.t_mo >= 0.09).sum())}; >= 8%: "
    f"{int((A.t_mo >= 0.08).sum())} | since 2025 >= 10%: {int((A.s_mo >= 0.10).sum())}; >= 8%: {int((A.s_mo >= 0.08).sum())}; "
    f">= 7%: {int((A.s_mo >= 0.07).sum())}")

_st = {}


def stressed(name):
    if name not in _st:
        _st[name] = T.evaluate(D[name], stress=True)[0]
    return _st[name]


def sline(rs):
    return (f"      stressed: 21-23 {rs['d_mo']*100:+5.1f}% dd{rs['d_dd']*100:4.0f} | 24-26 {rs['t_mo']*100:+5.1f}% dd"
            f"{rs['t_dd']*100:4.0f} | 25- {rs['s_mo']*100:+5.1f}% dd{rs['s_dd']*100:4.0f} | win {rs['d_win']*100:.1f}/"
            f"{rs['t_win']*100:.1f}/{rs['s_win']*100:.1f}% | worst trade {min(rs['d_wacct'], rs['t_wacct'])*100:+.0f}%")


LEV = [(0.95, 1.01), (0.90, 0.95), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80), (0.70, 0.75), (0.0, 0.70)]
say("\n1. Best design at each win-rate level (win rate = lower of 2021-23 and 2024-26):")
for lo, hi in LEV:
    g = A[(A.min_win >= lo) & (A.min_win < hi)]
    if len(g):
        r = R.rank(g).iloc[0]
        say(f"  win {lo*100:.0f}-{min(hi, 1)*100:.0f}%: " + R.line(r))
        say(sline(stressed(r["name"])))

say("\n2. Win rate >= 90% in 2021-23 and 2024-26: best design at each worst-drop level (worst of the three periods):")
W = A[A.q_wr90]
for cap in (-0.30, -0.40, -0.50, -0.60, -1.0):
    g = W[W.min_dd >= cap]
    if len(g):
        r = R.rank(g).iloc[0]
        say(f"  worst drop >= {cap*100:.0f}%: " + R.line(r) + f" | worst drop {r.min_dd*100:.0f}%")
        say(sline(stressed(r["name"])))

say("\n   same, win rate >= 90% in all three periods (2021-23, 2024-26 and since 2025) AND in the stressed version:")
for cap in (-0.30, -0.40, -0.50, -0.60, -1.0):
    g = R.rank(W[(W.min_dd >= cap) & (W.s_win >= 0.90)])
    for _, r in g.head(12).iterrows():
        rs = stressed(r["name"])
        if min(rs["d_win"], rs["t_win"], rs["s_win"]) >= 0.90:
            say(f"  worst drop >= {cap*100:.0f}%: " + R.line(r) + f" | worst drop {r.min_dd*100:.0f}%")
            say(sline(rs))
            break

say("\n3. Stressed version of the 40 best designs by worst-of-three (normal); sorted by stressed worst-of-three:")
top = R.rank(A).head(40)
rows = []
for _, r in top.iterrows():
    rs = stressed(r["name"])
    rows.append(dict(name=r["name"], n_min=r.min_mo, n_t=r.t_mo, n_s=r.s_mo, s_d=rs["d_mo"], s_t=rs["t_mo"], s_s=rs["s_mo"],
                     s_min=min(rs["d_mo"], rs["t_mo"], rs["s_mo"]), s_dd=min(rs["d_dd"], rs["t_dd"], rs["s_dd"]),
                     win=r.min_win, s_win=min(rs["d_win"], rs["t_win"])))
S = pd.DataFrame(rows).sort_values("s_min", ascending=False)
S.to_csv(os.path.join(T.OUT, "frontier_stressed_top40.csv"), index=False)
for _, x in S.iterrows():
    say(f"  {x['name']:58s} normal min {x.n_min*100:+5.2f} (24-26 {x.n_t*100:+5.2f}, 25- {x.n_s*100:+5.2f}) | stressed "
        f"21-23 {x.s_d*100:+5.2f} 24-26 {x.s_t*100:+5.2f} 25- {x.s_s*100:+5.2f} min {x.s_min*100:+5.2f} dd {x.s_dd*100:4.0f}% | "
        f"win {x.win*100:.0f}% (stressed {x.s_win*100:.0f}%)")
A.to_csv(os.path.join(T.OUT, "all_designs.csv"), index=False)
out.close()
