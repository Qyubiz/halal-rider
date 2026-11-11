"""w90 ten-watch, step 5: summary tables from the grid CSVs (w90_watch_grid.py A-D; watch bot, wicks 1%, 19 coins).
Prints the frontier (best worst-of-3 %/month per win-rate floor), the live-like rows, and how far %/month gets."""
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_watch_lib as W  # noqa: E402

G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(W.OUT, "grid[ABCD]_*.csv")))], ignore_index=True)
G = G.drop_duplicates(subset=["p_N", "p_depth", "p_tp", "p_guard", "p_hm"]).reset_index(drop=True)
G["w3"] = G[["d_pm", "t_pm", "s_pm"]].min(axis=1)
G["wdd"] = G[["d_dd", "t_dd", "s_dd"]].min(axis=1)
G["wmin"] = G[["d_win", "t_win"]].min(axis=1)


def row(b):
    return (f"{b['name']:26s} win {b.d_win*100:4.1f}/{b.t_win*100:4.1f}/{b.s_win*100:4.1f}% | %/mo {b.d_pm*100:+6.2f}/"
            f"{b.t_pm*100:+5.2f}/{b.s_pm*100:+5.2f} | drop {b.d_dd*100:3.0f}/{b.t_dd*100:3.0f}/{b.s_dd*100:3.0f}% "
            f"(24-26 minute marks {b.t_dd_min*100:3.0f}%) | trades {b.d_trades:.0f}/{b.t_trades:.0f}")


print(f"{len(G)} distinct designs; columns: 2021-23/2024-26/since 2025")
print(f"2024-26 %/month: max {G.t_pm.max()*100:.2f}, designs >= 8%: {int((G.t_pm >= .08).sum())}, >= 10%: "
      f"{int((G.t_pm >= .10).sum())}; since 2025 max {G.s_pm.max()*100:.2f}; designs with >= 10% in BOTH 2021-23 and "
      f"2024-26: {int(((G.d_pm >= .1) & (G.t_pm >= .1)).sum())}")
print(f"designs with win rate >= 90% in BOTH 2021-23 and 2024-26: {int((G.wmin >= .9).sum())}")
for cap, lab in ((None, "no drop cap"), (-0.40, "worst drop in every period shallower than -40%"),
                 (-0.30, "worst drop in every period shallower than -30%")):
    print(f"\nFRONTIER ({lab}): best worst-of-3 %/month at each both-period win-rate floor")
    H = G if cap is None else G[G.wdd >= cap]
    last = None
    for lvl in (0.60, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96, 0.97):
        q = H[H.wmin >= lvl].sort_values(["w3", "wdd"], ascending=[False, False])
        if not len(q):
            continue
        b = q.iloc[0]
        tag = "  (same)" if b["name"] == last else ""
        print(f"  win >= {lvl*100:3.0f}%: {row(b)}{tag}")
        last = b["name"]
print("\nreference rows (same watch bot):")
for n in ("B N8 d12 tp1.5 g0 hm48", "B N8 d12 tp1.5 g35 hm48", "A N1 d14 tp5 g35 hm48", "A N1 d14 tp5 g0 hm48"):
    q = G[G.name == n]
    if len(q):
        print("  " + row(q.iloc[0]))
