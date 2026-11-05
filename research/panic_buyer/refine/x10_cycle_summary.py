"""Lever "cycle": one table over every phase, read from the saved grids (no new simulation).

Phases: 1 plain fast bot (x10_cycle_grid.py), 2 guards/exits/slow trigger (x10_cycle_grid2.py), 4 wick layer
(x10_cycle_wick.py). Stressed columns use the liquidation-corrected re-runs (x10_cycle_restress.py). Phase 3 (1-minute
checks) is two runs, in pick2_liq_report.txt. 10%/month over 2024-01-01..2026-10-22 = x24.9.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_lib as L  # noqa: E402

K1 = ["C", "W", "depth", "tp", "hm", "cd", "N", "fmax"]
K2 = K1 + ["lcd", "stop", "skip", "exit", "dual"]


def load():
    A1 = pd.read_csv(os.path.join(L.OUT, "grid_normal.csv"))
    S1 = pd.read_csv(os.path.join(L.OUT, "grid_stressed_liq.csv"))
    J1 = A1.merge(S1.rename(columns={c: c + "_s" for c in S1.columns if c not in K1}), on=K1)
    A2 = pd.read_csv(os.path.join(L.OUT, "grid2_normal.csv"))
    S2 = pd.read_csv(os.path.join(L.OUT, "grid2_stressed_liq.csv"))
    J2 = A2.merge(S2.rename(columns={c: c + "_s" for c in S2.columns if c not in K2}), on=K2)
    W4 = pd.read_csv(os.path.join(L.OUT, "wick_grid.csv"))
    return J1, J2, W4


def rule(df, tag):
    ok = df[(df["y2022" + tag] >= 0) & (df["d_dd" + tag] >= -0.5)]
    return ok, ok.sort_values(["d_mo" + tag, "d_dd" + tag], ascending=[False, False]).iloc[0]


def four(r):
    return (f"(1) 24-26 {r.t_mo*100:+.2f}%/mo dd {r.t_dd*100:.1f}% | (2) 25-26 {r.s_mo*100:+.2f}%/mo dd {r.s_dd*100:.1f}% | "
            f"(3) 21-23 {r.d_mo*100:+.2f}%/mo dd {r.d_dd*100:.1f}% | (4) stressed 24-26 {r.t_mo_s*100:+.2f}%/mo dd "
            f"{r.t_dd_s*100:.1f}%, 25-26 {r.s_mo_s*100:+.2f}%/mo dd {r.s_dd_s*100:.1f}%, 21-23 {r.d_mo_s*100:+.2f}%/mo dd "
            f"{r.d_dd_s*100:.1f}%, whole-run dd {r.run_dd_s*100:.1f}%")


def main():
    J1, J2, W4 = load()
    out = []
    p = lambda s="": out.append(s)  # noqa: E731
    p(f"variants: phase 1 {len(J1)}, phase 2 {len(J2)}, phase 4 {len(W4)} combinations, phase 3 2 runs "
      f"(each normal + stressed) -> {len(J1) + len(J2) + len(W4) + 2} designs")
    for nm, J, keys in (("PHASE 1", J1, K1), ("PHASE 2", J2, K2)):
        p(f"\n{nm}")
        for lab, tag in (("rule A (normal 2021-23)", ""), ("rule B (stressed 2021-23)", "_s")):
            ok, r = rule(J, tag)
            p(f"  {lab}: {len(ok)} qualify; pick {dict((k, r[k]) for k in keys)}")
            p(f"    {four(r)}")
            p(f"    worst trade {r.worst_trade_acct*100:+.1f}% of account (stressed {r.worst_trade_acct_s*100:+.1f}%)")
            p(f"    qualifiers' normal 2024-26: median {ok.t_mo.median()*100:+.2f}%/mo, max {ok.t_mo.max()*100:+.2f}%/mo, "
              f">= 10%/mo: {(ok.t_mo >= 0.10).sum()}")
        h = J.sort_values("t_mo", ascending=False).iloc[0]
        p(f"  HINDSIGHT best normal 2024-26: {dict((k, h[k]) for k in keys)}")
        p(f"    {four(h)}")
        p(f"    settings >= 10%/mo (normal 2024-26): {(J.t_mo >= 0.10).sum()}; of those with stressed whole-run drop "
          f"above -99%: {((J.t_mo >= 0.10) & (J.run_dd_s > -0.99)).sum()}; with stressed 2024-26 >= 10%/mo: "
          f"{(J.t_mo_s >= 0.10).sum()}; best stressed 2024-26 {J.t_mo_s.max()*100:+.2f}%/mo; best since 2025 "
          f"{J.s_mo.max()*100:+.2f}%/mo")
        both = J[(J.t_mo >= 0.10)]
        if len(both):
            p(f"    the >=10% settings: 2021-23 %/mo {both.d_mo.min()*100:+.1f}..{both.d_mo.max()*100:+.1f}, 2021-23 drop "
              f"{both.d_dd.min()*100:.0f}..{both.d_dd.max()*100:.0f}%, sizing N = {sorted(both.N.unique().tolist())}")
    p("\nPHASE 4 (wick layer)")
    ok = W4[(W4.y2022_s >= 0) & (W4.d_dd_s >= -0.5)]
    r = ok.sort_values(["d_mo_s", "d_dd_s"], ascending=[False, False]).iloc[0]
    p(f"  rule B4: {len(ok)} qualify; pick main={r['main']} a={r['a']} (a=0: the main bot alone)")
    h = W4.sort_values("t_mo", ascending=False).iloc[0]
    p(f"  HINDSIGHT best normal 2024-26: main={h['main']} a={h['a']} wick C{h.w_C:.0f} W{h.w_W:.0f} depth {h.w_depth} "
      f"tp {h.w_tp} hm {h.w_hm:.0f} N{h.w_N:.0f} lcd {h.w_lcd:.0f}: 24-26 {h.t_mo*100:+.2f}%/mo dd {h.t_dd*100:.1f}%, "
      f"21-23 {h.d_mo*100:+.2f}%/mo dd {h.d_dd*100:.1f}%")
    allmax = max(J1.t_mo.max(), J2.t_mo.max(), W4.t_mo.max())
    p(f"\nHIGHEST 2024-26 %/month of any variant (hindsight, normal runs): {allmax*100:+.2f}%")
    txt = "\n".join(out)
    print(txt)
    open(os.path.join(L.OUT, "summary.txt"), "w").write(txt + "\n")


if __name__ == "__main__":
    main()
