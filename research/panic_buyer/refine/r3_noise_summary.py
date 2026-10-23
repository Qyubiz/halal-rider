"""Round 3, task "noise": summary of r3_noise.py's saved runs (reads $PB_OUT/r3_noise_runs.csv; runs nothing).

For each near miss, setting and period: the paired difference (near miss minus live under the same perturbation)
over the 100 random combined draws and over the 15 named perturbations: mean, sd, min, max and how often the
multiple is higher; the same for the drop (in points). Also the live band's 5th-95th percentiles over the draws.
"""
import os

import numpy as np
import pandas as pd

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out"))
R = pd.read_csv(os.path.join(OUT, "r3_noise_runs.csv"))

print("live, 100 random draws: 5th / 50th / 95th percentiles")
for st, g in R[(R.strat == "LIVE") & (R.kind == "random")].groupby("setting", sort=False):
    q = g[["d_mult", "d_dd", "t_mult", "t_dd"]].quantile([0.05, 0.5, 0.95])
    print(f"  {st:17s} 2021-23 x{q.d_mult.iloc[0]:.3f} / x{q.d_mult.iloc[1]:.3f} / x{q.d_mult.iloc[2]:.3f}, "
          f"drop {q.d_dd.iloc[0] * 100:.2f}% / {q.d_dd.iloc[1] * 100:.2f}% / {q.d_dd.iloc[2] * 100:.2f}% | 2024-26 "
          f"x{q.t_mult.iloc[0]:.3f} / x{q.t_mult.iloc[1]:.3f} / x{q.t_mult.iloc[2]:.3f}, drop {q.t_dd.iloc[0] * 100:.2f}% / "
          f"{q.t_dd.iloc[1] * 100:.2f}% / {q.t_dd.iloc[2] * 100:.2f}%")

rows = []
for st in R.setting.unique():
    L = R[(R.strat == "LIVE") & (R.setting == st)].set_index(["kind", "label"])
    for name in R.strat.unique():
        if name == "LIVE":
            continue
        V = R[(R.strat == name) & (R.setting == st)].set_index(["kind", "label"])
        for sel, kinds in (("100 random", ["random"]), ("15 named", ["param", "start"])):
            lv = L[L.index.get_level_values(0).isin(kinds)]
            vv = V.loc[lv.index]
            for per in ("d", "t"):
                dm = vv[f"{per}_mult"] - lv[f"{per}_mult"]
                dd = (vv[f"{per}_dd"] - lv[f"{per}_dd"]) * 100
                rows.append(dict(setting=st, strat=name, draws=sel, period="2021-23" if per == "d" else "2024-26",
                                 mult_gap_mean=dm.mean(), mult_gap_sd=dm.std(ddof=1), mult_gap_min=dm.min(),
                                 mult_gap_max=dm.max(), mult_higher=f"{int((dm > 0).sum())}/{len(dm)}",
                                 drop_gap_pt_mean=dd.mean(), drop_gap_pt_sd=dd.std(ddof=1), drop_gap_pt_min=dd.min(),
                                 drop_gap_pt_max=dd.max(), drop_within_1pt=f"{int((dd >= -1.0).sum())}/{len(dd)}"))
S = pd.DataFrame(rows)
S.to_csv(os.path.join(OUT, "r3_noise_paired.csv"), index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
S["strat"] = S.strat.str.slice(0, 44)
for st in R.setting.unique():
    print(f"\n--- paired differences (near miss minus live, same perturbation), {st}")
    print(S[S.setting == st].drop(columns="setting").to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
