"""Lever "cycle", phase 1 pick: apply the pre-registered rules of x10_cycle_grid.py, then report each pick once.

The rule functions receive only the 2021-23 columns (d_*, y2021..y2023) plus the settings. 2024-26 and
since-2025 are looked up afterwards. Every pick and every hindsight setting is re-run here to print the four
reports (normal run; stressed run = every buy 0.2% worse + LUNA/FTT/SRM), the worst trade as % of the account,
per-year returns, and two extra stresses: fills 0.2% worse on BOTH sides (buys and sells), and both-sides plus
LUNA/FTT/SRM.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_lib as L  # noqa: E402
import x10_cycle_grid as G  # noqa: E402

KEYS = ["C", "W", "depth", "tp", "hm", "cd", "N", "fmax"]
DCOLS = KEYS + ["d_mult", "d_mo", "d_dd", "y2021", "y2022", "y2023"]
_bars = {}


def bars(C, dead):
    if (C, dead) not in _bars:
        _bars.clear()                 # keep one bar set in memory
        B = L.load_bars(C, dead)
        _bars[(C, dead)] = (B, L.ffill_close(B), L.Clock(B["t"]))
    return _bars[(C, dead)]


def rule(design, max_dd=-0.50, min_2022=0.0):
    """design: DataFrame with DCOLS only. Returns the picked row's settings (dict) and the qualifier count."""
    assert set(design.columns) <= set(DCOLS)
    ok = design[(design.y2022 >= min_2022) & (design.d_dd >= max_dd)]
    if len(ok):
        best = ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).iloc[0]
    else:
        best = design.assign(calmar=design.d_mo / design.d_dd.abs()).sort_values("calmar", ascending=False).iloc[0]
    return {k: best[k] for k in KEYS}, len(ok)


def name(s):
    return (f"check {int(s['C'])}m, trigger {int(s['W'])}h-high -{s['depth']*100:.0f}%, tp +{s['tp']*100:.1f}%, "
            f"stop {int(s['hm'])}h, cooldown {int(s['cd'])}h, {int(s['N'])} coin(s) x {s['fmax']*100:.0f}%")


def run(s, dead=False, pen=0.0, pens=0.0):
    C = int(s["C"]); bph = 60 // C
    B, cf, clk = bars(C, dead)
    ref = L.make_ref(B, int(s["W"]) * bph)
    eq, lg = L.crun(B, ref, cf, s["depth"], int(s["N"]), s["fmax"], s["tp"], int(s["hm"]) * bph,
                    int(s["cd"]) * bph, pen=pen, pens=pens)
    return L.summarize(clk, eq, lg), eq, lg, clk


def full_report(label, s, out):
    r, eq, lg, clk = run(s)
    rs = run(s, dead=True, pen=G.STRESS_PEN)[0]
    rb = run(s, pen=0.002, pens=0.002)[0]
    rbd = run(s, dead=True, pen=0.002, pens=0.002)[0]
    txt = L.fmt4(f"{label}: {name(s)}", r, rs)
    txt += (f"\n  extra: fills 0.2% worse on buys AND sells: 2024-26 {rb['t_mo']*100:+.2f}%/mo drop {rb['t_dd']*100:.1f}% | "
            f"2025-26 {rb['s_mo']*100:+.2f}%/mo | 2021-23 {rb['d_mo']*100:+.2f}%/mo drop {rb['d_dd']*100:.1f}%"
            f"\n  extra: both sides + LUNA/FTT/SRM: 2024-26 {rbd['t_mo']*100:+.2f}%/mo drop {rbd['t_dd']*100:.1f}% | "
            f"2025-26 {rbd['s_mo']*100:+.2f}%/mo | 2021-23 {rbd['d_mo']*100:+.2f}%/mo drop {rbd['d_dd']*100:.1f}%")
    txt += "\n  years: " + " ".join(f"{y} {r[f'y{y}']*100:+.0f}%" for y in range(2021, 2027))
    m = clk.months(eq)
    mt = m[m.index >= L.T_TEST0]
    txt += (f"\n  2024-26 months: up {(mt > 0).mean()*100:.0f}%, worst {mt.min()*100:+.1f}% ({mt.idxmin():%Y-%m}), "
            f"best {mt.max()*100:+.1f}% ({mt.idxmax():%Y-%m}), median {mt.median()*100:+.1f}%")
    print(txt + "\n", file=out, flush=True)
    print(txt + "\n", flush=True)
    return r, rs


def main():
    A = pd.read_csv(os.path.join(L.OUT, "grid_normal.csv"))
    S = pd.read_csv(os.path.join(L.OUT, "grid_stressed.csv"))
    out = open(os.path.join(L.OUT, "pick_report.txt"), "w")
    print(f"grid: {len(A)} settings (normal), {len(S)} (stressed)\n", file=out)

    sa, na = rule(A[DCOLS])
    sb, nb = rule(S[DCOLS])
    print(f"RULE A: {na} settings qualify on 2021-23 (2022 >= 0, drop >= -50%)", file=out)
    print(f"RULE A: {na} settings qualify; RULE B (stressed 2021-23): {nb} qualify")
    full_report("RULE A PICK (chosen on 2021-23 only)", sa, out)
    print(f"RULE B: {nb} settings qualify on stressed 2021-23", file=out)
    full_report("RULE B PICK (chosen on stressed 2021-23 only)", sb, out)

    # how the rule-A qualifiers did afterwards
    ok = A[(A.y2022 >= 0) & (A.d_dd >= -0.50)]
    if len(ok):
        print(f"Rule-A qualifiers ({len(ok)}): 2024-26 %/mo median {ok.t_mo.median()*100:+.2f}, max {ok.t_mo.max()*100:+.2f}, "
              f"share >= 10%/mo {(ok.t_mo >= 0.10).mean()*100:.1f}%, share >= 5%/mo {(ok.t_mo >= 0.05).mean()*100:.1f}%",
              file=out)
        top = ok.sort_values("d_mo", ascending=False).head(15)
        print("Top 15 rule-A qualifiers by 2021-23 %/mo (what they did later):", file=out)
        for _, r in top.iterrows():
            print(f"  {name(r):95s} 21-23 {r.d_mo*100:+6.2f}%/mo dd {r.d_dd*100:4.0f}% | 24-26 {r.t_mo*100:+6.2f}%/mo "
                  f"dd {r.t_dd*100:4.0f}% | 25-26 {r.s_mo*100:+6.2f}%/mo", file=out)

    # hindsight
    print("\nHINDSIGHT (chosen by looking at 2024-26; NOT an honest pick)", file=out)
    print(f"all {len(A)}: 2024-26 %/mo median {A.t_mo.median()*100:+.2f}, max {A.t_mo.max()*100:+.2f}; "
          f">= 10%/mo: {(A.t_mo >= 0.10).sum()}; >= 7%/mo: {(A.t_mo >= 0.07).sum()}; max since-2025 "
          f"{A.s_mo.max()*100:+.2f}%/mo", file=out)
    hb = A.sort_values("t_mo", ascending=False).iloc[0]
    full_report("HINDSIGHT best 2024-26", {k: hb[k] for k in KEYS}, out)
    h50 = A[A.t_dd >= -0.50].sort_values("t_mo", ascending=False).iloc[0]
    full_report("HINDSIGHT best 2024-26 with 2024-26 drop >= -50%", {k: h50[k] for k in KEYS}, out)
    hs = A.sort_values("s_mo", ascending=False).iloc[0]
    full_report("HINDSIGHT best since 2025", {k: hs[k] for k in KEYS}, out)
    print("\nTop 15 by 2024-26 %/mo (hindsight):", file=out)
    for _, r in A.sort_values("t_mo", ascending=False).head(15).iterrows():
        print(f"  {name(r):95s} 24-26 {r.t_mo*100:+6.2f}%/mo dd {r.t_dd*100:4.0f}% | 25-26 {r.s_mo*100:+6.2f}%/mo | "
              f"21-23 {r.d_mo*100:+6.2f}%/mo dd {r.d_dd*100:4.0f}%", file=out)
    # marginal effects (median 2024-26 and 2021-23 %/mo per parameter value)
    print("\nMedian %/month by parameter value (all settings): 2021-23 | 2024-26 | since 2025 | 2024-26 drop", file=out)
    for k in KEYS:
        if k == "fmax":
            continue
        g = A.groupby(k)[["d_mo", "t_mo", "s_mo", "t_dd"]].median()
        for v, r in g.iterrows():
            print(f"  {k:6s} {v:>7}: {r.d_mo*100:+6.2f} | {r.t_mo*100:+6.2f} | {r.s_mo*100:+6.2f} | {r.t_dd*100:5.1f}",
                  file=out)
    out.close()


if __name__ == "__main__":
    main()
