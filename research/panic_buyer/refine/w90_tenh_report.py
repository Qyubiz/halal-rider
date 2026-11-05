"""Shared reporting for the ten-hourly grids: qualifiers under the fixed rule, top designs, the win-rate frontier,
and the stressed version of the leaders."""
import os

import numpy as np
import pandas as pd

import w90_tenh_lib as T

LEVELS = [(0.90, 1.01), (0.85, 0.90), (0.80, 0.85), (0.75, 0.80), (0.70, 0.75), (0.0, 0.70)]


def qualify(G):
    G = G.copy()
    G["q_ten"] = (G.d_mo >= 0.10) & (G.t_mo >= 0.10)
    G["q_wr90"] = (G.d_win >= 0.90) & (G.t_win >= 0.90)
    return G


def rank(G):
    """The task's preference: highest worst-of-three %/month, ties -> shallower worst drop."""
    return G.sort_values(["min_mo", "min_dd"], ascending=[False, False])


def line(r):
    return (f"{r['name']:44s} 21-23 {r.d_mo*100:+5.1f}% dd{r.d_dd*100:4.0f} w{r.d_win*100:3.0f} | "
            f"24-26 {r.t_mo*100:+5.1f}% dd{r.t_dd*100:4.0f} w{r.t_win*100:3.0f} n{int(r.t_n):3d} | "
            f"25- {r.s_mo*100:+5.1f}% dd{r.s_dd*100:4.0f} w{r.s_win*100:3.0f} | min {r.min_mo*100:+5.2f}% "
            f"wtrade {min(r.d_wacct, r.t_wacct)*100:+.0f}%")


def frontier(G):
    """Best design (by the task's preference) at each win-rate level, win rate = the lower of 2021-23 and 2024-26."""
    out = []
    for lo, hi in LEVELS:
        g = G[(G.min_win >= lo) & (G.min_win < hi)]
        if len(g):
            out.append((lo, hi, rank(g).iloc[0]))
    return out


def pareto(G):
    """Designs not beaten on all three of: lower win rate (min of periods), worst-of-three %/month, worst drop."""
    a = G[["min_win", "min_mo", "min_dd"]].values
    keep = []
    for i in range(len(a)):
        dom = np.all(a >= a[i], axis=1) & np.any(a > a[i], axis=1)
        if not dom.any():
            keep.append(i)
    return G.iloc[keep]


def report(G, tag, D=None, stress_top=6, extra=()):
    G = qualify(G)
    G.to_csv(os.path.join(T.OUT, f"{tag}.csv"), index=False)
    f = open(os.path.join(T.OUT, f"{tag}.log"), "w")

    def say(s=""):
        print(s, flush=True)
        f.write(s + "\n"); f.flush()

    say(f"#### {tag}: {len(G)} designs | qualify ten (>=10%/mo in 2021-23 AND 2024-26): {int(G.q_ten.sum())} | "
        f"qualify wr90: {int(G.q_wr90.sum())} | max 2024-26 {G.t_mo.max()*100:.2f}%/mo | max since-2025 "
        f"{G.s_mo.max()*100:.2f}%/mo | max worst-of-three {G.min_mo.max()*100:.2f}%/mo")
    say("\nTop 15 by worst-of-three %/month (all designs):")
    for _, r in rank(G).head(15).iterrows():
        say("  " + line(r) + ("  [ten]" if r.q_ten else ""))
    say("\nTop 10 by 2024-26 %/month:")
    for _, r in G.sort_values("t_mo", ascending=False).head(10).iterrows():
        say("  " + line(r))
    say("\nTop 10 by since-2025 %/month:")
    for _, r in G.sort_values("s_mo", ascending=False).head(10).iterrows():
        say("  " + line(r))
    for q in ("q_ten", "q_wr90"):
        g = G[G[q]]
        say(f"\nQualifiers {q} ({len(g)}), best 8 by the rule:")
        for _, r in rank(g).head(8).iterrows():
            say("  " + line(r))
    say("\nFrontier: best design at each win-rate level (win rate = lower of 2021-23 and 2024-26):")
    for lo, hi, r in frontier(G):
        say(f"  win {lo*100:.0f}-{min(hi,1)*100:.0f}%: " + line(r))
    P = pareto(G)
    say(f"\nPareto set (win rate, worst-of-three %/mo, worst drop): {len(P)} designs")
    for _, r in P.sort_values("min_win").iterrows():
        say("  " + line(r))
    if D is not None and stress_top:
        names = list(rank(G).head(stress_top).name) + [n for n in extra if n in D]
        say(f"\nStressed version (bot timing cd+1, buys and sells 0.2% worse, +LUNA/FTT/SRM sold at their last hour):")
        for n in dict.fromkeys(names):
            s = D[n]
            r, _, _ = T.evaluate(s)
            rs, _, _ = T.evaluate(s, stress=True)
            say(T.fmt(n, r, rs))
    f.close()
    return G
