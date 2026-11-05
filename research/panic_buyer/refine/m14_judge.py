"""m14 judge: the ONE look for every candidate (M14_PREREG.md pass rule, items 1-4 and 6; item 5 is reported by
the grid neighbours / the agents).

  python3 m14_judge.py grid                      grid finalists ($PB_OUT/finalists.json): items 1-4 + luck check
  python3 m14_judge.py curves NAME MAIN HOLD     an agent's pick from its saved hourly equity CSVs (time,equity):
                                                 luck check only (items 1-4 come from the agent's own table)

Luck check: monthly returns (calendar months, normal fills) of candidate minus 90% mode, for 2024-26 and for the
2018-20 holdout; stationary block bootstrap of the mean difference (mean block 3 months, 20,000 resamples, seed 0);
one-sided p must be below 0.05/30 in BOTH periods.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402
import m11_fusion_lib as F  # noqa: E402
import m14_grid as G  # noqa: E402

ALPHA = 0.05 / 30
T24, H18, H21 = pd.Timestamp("2024-01-01"), pd.Timestamp("2018-01-01"), pd.Timestamp("2021-01-01")


def monthly(eq, a, b):
    m = eq.resample("1ME").last()
    prev = eq[eq.index < a]
    start = prev.iloc[-1] if len(prev) else eq.iloc[0]
    m = m[(m.index >= a) & ((m.index < b) if b is not None else True)]
    return pd.concat([pd.Series([start]), m.reset_index(drop=True)]).pct_change().dropna().values


def boot_p(d, n=20000, block=3.0, seed=0):
    """One-sided p-value for mean(d) > 0, stationary bootstrap of the centred series."""
    rng = np.random.default_rng(seed)
    T = len(d)
    obs = d.mean()
    x = d - obs
    p_new = 1.0 / block
    cnt = 0
    for _ in range(n):
        i = rng.integers(T)
        s = 0.0
        for _t in range(T):
            s += x[i]
            i = rng.integers(T) if rng.random() < p_new else (i + 1) % T
        cnt += s / T >= obs
    return (cnt + 1) / (n + 1), obs


_base = {}


def base_curves():
    if not _base:
        _, e, _ = F.evaluate(F.MODE90)
        _, eh, _ = F.evaluate(F.MODE90, holdout=True)
        _base["m"], _base["h"] = e, eh
    return _base["m"], _base["h"]


def luck(e_main, e_hold):
    bm, bh = base_curves()
    out = {}
    for tag, e, b, a, z in (("t", e_main, bm, T24, None), ("h", e_hold, bh, H18, H21)):
        d = monthly(e, a, z) - monthly(b, a, z)
        p, obs = boot_p(d)
        out[f"{tag}_p"], out[f"{tag}_diff"] = p, obs
    out["luck_ok"] = out["t_p"] < ALPHA and out["h_p"] < ALPHA
    return out


B = {"t_mo": 0.0689, "h_mo": 0.1082, "wt_mo": 0.0531, "wh_mo": 0.0741, "st_mo": 0.0549}


def items(d):
    rn, en, _ = F.evaluate(d)
    rh, eh, _ = F.evaluate(d, holdout=True)
    rw, _, _ = F.evaluate(d, worse=True)
    rwh, _, _ = F.evaluate(d, worse=True, holdout=True)
    rs, _, _ = F.evaluate(d, stress=True)
    i1 = rn["t_mo"] > B["t_mo"] and rn["t_wr"] >= 0.925 and rn["t_dd"] >= -0.35
    i2 = rh["h_mo"] > B["h_mo"] and rh["h_wr"] >= 0.925 and rh["h_dd"] >= -0.663
    i3 = (rw["t_mo"] > B["wt_mo"] and rwh["h_mo"] > B["wh_mo"] and rw["t_wr"] >= 0.92 and rwh["h_wr"] >= 0.92)
    i4 = rs["t_mo"] > B["st_mo"] and rs["d_dd"] > -0.95
    row = dict(t=(rn["t_wr"], rn["t_mo"], rn["t_dd"]), s=(rn["s_wr"], rn["s_mo"], rn["s_dd"]),
               h=(rh["h_wr"], rh["h_mo"], rh["h_dd"]), wt=rw["t_mo"], wh=rwh["h_mo"], st=rs["t_mo"], sd=rs["d_dd"],
               i1=i1, i2=i2, i3=i3, i4=i4)
    return row, en, eh


def fmt(x):
    return f"{x[0]*100:.1f} / {x[1]*100:+.2f} / {x[2]*100:.1f}"


def main():
    mode = sys.argv[1]
    if mode == "grid":
        fin = json.load(open(os.path.join(F.OUT, "finalists.json")))
        print("| # | setting | 2024-26 | since 2025 | 2018-20 | worse 24-26 / 18-20 | stressed 24-26 / 21-23 drop "
              "| items 1-4 | luck p 24-26 / 18-20 |\n|---|---|---|---|---|---|---|---|---|")
        for k, ix in enumerate(fin, 1):
            d = G.design(tuple(ix))
            r, en, eh = items(d)
            lk = luck(en, eh)
            print(f"| {k} | {F.describe(d)} | {fmt(r['t'])} | {fmt(r['s'])} | {fmt(r['h'])} | {r['wt']*100:+.2f} / "
                  f"{r['wh']*100:+.2f} | {r['st']*100:+.2f} / {r['sd']*100:.0f} | "
                  f"{''.join('Y' if r[f'i{j}'] else 'n' for j in range(1, 5))} | {lk['t_p']:.4f} / {lk['h_p']:.4f} |",
                  flush=True)
    else:
        name, fm, fh = sys.argv[2:5]
        em = pd.read_csv(fm, parse_dates=["time"]).set_index("time")["equity"]
        eh = pd.read_csv(fh, parse_dates=["time"]).set_index("time")["equity"]
        lk = luck(em, eh)
        print(f"{name}: mean monthly edge vs 90% mode 2024-26 {lk['t_diff']*100:+.2f} pts (p {lk['t_p']:.4f}), "
              f"2018-20 {lk['h_diff']*100:+.2f} pts (p {lk['h_p']:.4f}); luck check {'PASS' if lk['luck_ok'] else 'fail'}"
              f" (needs p < {ALPHA:.4f} in both)")


if __name__ == "__main__":
    main()
