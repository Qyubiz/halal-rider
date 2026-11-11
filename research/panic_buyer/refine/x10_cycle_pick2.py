"""Lever "cycle", phase 2 + 3 report: apply the pre-registered rules of x10_cycle_grid2.py, report each pick once
(all four reports), the hindsight bests (labelled), what each guard/exit did, and phase 3 (the picks with
1-minute checks).

Usage: python x10_cycle_pick2.py [phase3] [liq]
  phase3: the 1-minute versions of the picks (needs ~2.5 GB).
  liq:    use the stressed grids re-run with collapse-coin liquidation (x10_cycle_restress.py) and liquidation in
          every stressed run here; also re-applies phase-1 rule B to the corrected phase-1 stressed grid.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_1m as M1  # noqa: E402
import x10_cycle_grid2 as G  # noqa: E402
import x10_cycle_lib as L  # noqa: E402

KEYS = G.KEYS
DCOLS = KEYS + ["d_mult", "d_mo", "d_dd", "y2021", "y2022", "y2023"]
P1 = dict(lcd=0, stop=0.0, skip=0.0, exit="plain", dual=0.0)
LIQ = "liq" in sys.argv[1:]
_cache = {}


def env(C, dead):
    """Bars, ffilled close, clock and the trigger refs for one cycle length (one set kept in memory)."""
    if (C, dead) not in _cache:
        _cache.clear()
        if C == 1:
            B = M1.load(dead); cf = M1.cf32(B); mk = M1.ref32
        else:
            B = L.load_bars(C, dead); cf = L.ffill_close(B); mk = L.make_ref
        _cache[(C, dead)] = dict(B=B, cf=cf, clk=L.Clock(B["t"]), bph=60 // C, mk=mk, refs={})
    e = _cache[(C, dead)]
    return e


def run(s, dead=False, pen=0.0, pens=0.0):
    e = env(int(s["C"]), dead)
    bph = e["bph"]
    for w in (int(s["W"]), 24, 168):
        if w not in e["refs"]:
            e["refs"][w] = e["mk"](e["B"], w * bph)
    s = dict(s); s["W"] = int(s["W"]); s["N"] = int(s["N"]); s["hm"] = int(s["hm"]); s["lcd"] = int(s["lcd"])
    s["cd"] = int(s["cd"])
    if s["N"] == 3:
        s["fmax"] = 1.0 / 3.0                     # the csv stores 0.3333
    eq, lg = G.run_one(e["B"], e["refs"], e["cf"], bph, s, pen=pen, pens=pens, liquidate=LIQ and dead)
    r = L.summarize(e["clk"], eq, lg)
    r["run_dd"] = float((eq / np.maximum.accumulate(eq) - 1).min())
    return r, eq, lg, e["clk"]


def rule(design, max_dd=-0.50, min_2022=0.0):
    assert set(design.columns) <= set(DCOLS)
    ok = design[(design.y2022 >= min_2022) & (design.d_dd >= max_dd)]
    if len(ok):
        best = ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).iloc[0]
    else:
        best = design.assign(calmar=design.d_mo / design.d_dd.abs()).sort_values("calmar", ascending=False).iloc[0]
    return {k: best[k] for k in KEYS}, len(ok)


def name(s):
    g = []
    if s["lcd"]:
        g.append(f"{int(s['lcd'])}h block after a loss")
    if s["stop"]:
        g.append(f"exit at -{s['stop']*100:.0f}%")
    if s["skip"]:
        g.append(f"skip if >{s['skip']*100:.0f}% under 7d high")
    if s["dual"]:
        g.append(f"+ 24h-high -{s['dual']*100:.0f}% trigger")
    return (f"check {int(s['C'])}m, trigger {int(s['W'])}h-high -{s['depth']*100:.0f}%, tp +{s['tp']*100:.1f}% "
            f"[{s['exit']}], stop {int(s['hm'])}h, cd {int(s['cd'])}h, {int(s['N'])} x {s['fmax']*100:.0f}%; "
            f"guards: {', '.join(g) if g else 'none'}")


def wiped(r):
    return " ACCOUNT WIPED OUT (lost >99% at some point; later periods meaningless)" if r["run_dd"] <= -0.99 else ""


def full_report(label, s, out):
    r, eq, lg, clk = run(s)
    rs = run(s, dead=True, pen=G.STRESS_PEN)[0]
    rb = run(s, pen=0.002, pens=0.002)[0]
    rbd = run(s, dead=True, pen=0.002, pens=0.002)[0]
    txt = L.fmt4(f"{label}: {name(s)}", r, rs) + wiped(rs)
    txt += (f"\n  extra: fills 0.2% worse on buys AND sells: 2024-26 {rb['t_mo']*100:+.2f}%/mo drop {rb['t_dd']*100:.1f}% | "
            f"2025-26 {rb['s_mo']*100:+.2f}%/mo | 2021-23 {rb['d_mo']*100:+.2f}%/mo drop {rb['d_dd']*100:.1f}%"
            f"\n  extra: both sides + LUNA/FTT/SRM: 2024-26 {rbd['t_mo']*100:+.2f}%/mo drop {rbd['t_dd']*100:.1f}% | "
            f"2025-26 {rbd['s_mo']*100:+.2f}%/mo | 2021-23 {rbd['d_mo']*100:+.2f}%/mo drop {rbd['d_dd']*100:.1f}%"
            + wiped(rbd))
    txt += "\n  years: " + " ".join(f"{y} {r[f'y{y}']*100:+.0f}%" for y in range(2021, 2027))
    txt += "\n  stressed years: " + " ".join(f"{y} {rs[f'y{y}']*100:+.0f}%" for y in range(2021, 2027))
    m = clk.months(eq)
    mt = m[m.index >= L.T_TEST0]
    txt += (f"\n  2024-26 months: up {(mt > 0).mean()*100:.0f}%, worst {mt.min()*100:+.1f}% ({mt.idxmin():%Y-%m}), "
            f"best {mt.max()*100:+.1f}% ({mt.idxmax():%Y-%m}), median {mt.median()*100:+.1f}%")
    print(txt + "\n", file=out, flush=True)
    print(txt + "\n", flush=True)
    return r, rs


def phase1_liq(out):
    """Phase-1 rule B re-applied to the phase-1 stressed grid re-run with liquidation (rule unchanged)."""
    A1 = pd.read_csv(os.path.join(L.OUT, "grid_normal.csv"))
    S1 = pd.read_csv(os.path.join(L.OUT, "grid_stressed_liq.csv"))
    k1 = ["C", "W", "depth", "tp", "hm", "cd", "N", "fmax"]
    assert (A1[k1].values == S1[k1].values).all()
    d1 = k1 + ["d_mult", "d_mo", "d_dd", "y2021", "y2022", "y2023"]
    ok = S1[(S1.y2022 >= 0) & (S1.d_dd >= -0.5)]
    best = ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).iloc[0] if len(ok) else None
    print(f"PHASE-1 RULE B on the liquidation-corrected stressed grid: {len(ok)} settings qualify", file=out)
    if best is not None:
        s = {k: best[k] for k in k1}; s.update(P1)
        full_report("PHASE-1 RULE B PICK, corrected stress (chosen on stressed 2021-23 only)", s, out)
    J = A1.merge(S1.rename(columns={c: c + "_s" for c in S1.columns if c not in k1}), on=k1)
    surv = J[J.run_dd_s >= -0.5]
    print(f"phase 1, corrected stress: {len(surv)} settings whose stressed run never drops >50%; "
          f"their best normal 2024-26 {surv.t_mo.max()*100:+.2f}%/mo; best stressed 2024-26 overall "
          f"{J.t_mo_s.max()*100:+.2f}%/mo", file=out)


def main(phase3=False):
    A = pd.read_csv(os.path.join(L.OUT, "grid2_normal.csv"))
    S = pd.read_csv(os.path.join(L.OUT, "grid2_stressed_liq.csv" if LIQ else "grid2_stressed.csv"))
    assert (A[KEYS].values == S[KEYS].values).all()
    rep = "pick2_liq_report.txt" if LIQ else "pick2_report.txt"
    out = open(os.path.join(L.OUT, rep), "a" if phase3 else "w")
    if not phase3:
        print(f"phase 2 grid: {len(A)} settings (normal), {len(S)} (stressed{', liquidation on' if LIQ else ''})\n",
              file=out)
        if LIQ:
            phase1_liq(out)
    sb, nb = rule(S[DCOLS])
    sa, na = rule(A[DCOLS])
    if phase3:
        print("\n######## PHASE 3: the phase-2 picks with 1-minute checks (C=1, same hours)\n", file=out)
        for lab, s in (("RULE B2 PICK, 1-minute checks", sb), ("RULE A2 PICK, 1-minute checks", sa)):
            s = dict(s); s["C"] = 1
            full_report(lab, s, out)
        out.close()
        return
    print(f"RULE B2 (stressed 2021-23): {nb} settings qualify", file=out)
    full_report("RULE B2 PICK (chosen on stressed 2021-23 only)", sb, out)
    print(f"RULE A2 (normal 2021-23): {na} settings qualify", file=out)
    full_report("RULE A2 PICK (chosen on normal 2021-23 only)", sa, out)

    J = A.merge(S, on=KEYS, suffixes=("", "_s"))
    okb = J[(J.y2022_s >= 0) & (J.d_dd_s >= -0.5)]
    print(f"Rule-B2 qualifiers ({len(okb)}): 2024-26 %/mo median {okb.t_mo.median()*100:+.2f}, max {okb.t_mo.max()*100:+.2f}; "
          f"stressed 2024-26 median {okb.t_mo_s.median()*100:+.2f}, max {okb.t_mo_s.max()*100:+.2f}", file=out)
    print("Top 12 rule-B2 qualifiers by stressed 2021-23 %/mo (what they did later):", file=out)
    for _, r in okb.sort_values("d_mo_s", ascending=False).head(12).iterrows():
        print(f"  {name(r)}\n      stressed 21-23 {r.d_mo_s*100:+6.2f}%/mo dd {r.d_dd_s*100:4.0f}% | normal 24-26 {r.t_mo*100:+6.2f}%/mo "
              f"dd {r.t_dd*100:4.0f}% | stressed 24-26 {r.t_mo_s*100:+6.2f}%/mo | 25-26 {r.s_mo*100:+6.2f}%/mo", file=out)

    print("\nHINDSIGHT (chosen by looking at 2024-26; NOT an honest pick)", file=out)
    print(f"all {len(A)}: normal 2024-26 %/mo median {A.t_mo.median()*100:+.2f}, max {A.t_mo.max()*100:+.2f}; "
          f">= 10%/mo: {(A.t_mo >= 0.10).sum()}; >= 8%/mo: {(A.t_mo >= 0.08).sum()}; max since 2025 "
          f"{A.s_mo.max()*100:+.2f}%/mo; stressed 2024-26 max {S.t_mo.max()*100:+.2f}%/mo", file=out)
    hb = A.sort_values("t_mo", ascending=False).iloc[0]
    full_report("HINDSIGHT best 2024-26 (normal)", {k: hb[k] for k in KEYS}, out)
    surv = J[J.run_dd_s >= -0.5]
    if len(surv):
        hs = surv.sort_values("t_mo", ascending=False).iloc[0]
        full_report("HINDSIGHT best 2024-26 among settings whose stressed run never drops >50%", {k: hs[k] for k in KEYS}, out)
    hss = J.sort_values("t_mo_s", ascending=False).iloc[0]
    full_report("HINDSIGHT best stressed 2024-26", {k: hss[k] for k in KEYS}, out)
    print("\nTop 12 by normal 2024-26 %/mo (hindsight):", file=out)
    for _, r in J.sort_values("t_mo", ascending=False).head(12).iterrows():
        print(f"  {name(r)}\n      24-26 {r.t_mo*100:+6.2f}%/mo dd {r.t_dd*100:4.0f}% | 25-26 {r.s_mo*100:+6.2f}%/mo | 21-23 "
              f"{r.d_mo*100:+6.2f}%/mo dd {r.d_dd*100:4.0f}% | stressed: 24-26 {r.t_mo_s*100:+6.2f}%/mo run-dd {r.run_dd_s*100:4.0f}%",
              file=out)
    print("\nMedian by mechanic: normal 21-23 %/mo | normal 24-26 %/mo | since 2025 | stressed 21-23 %/mo | "
          "stressed 24-26 %/mo | stressed run drop | share of stressed runs wiped (>99% loss)", file=out)
    for k in ["C", "W", "depth", "tp", "hm", "N", "lcd", "stop", "skip", "exit", "dual"]:
        g = J.groupby(k)
        for v, r in g[["d_mo", "t_mo", "s_mo", "d_mo_s", "t_mo_s", "run_dd_s"]].median().iterrows():
            wp = (g.get_group(v).run_dd_s <= -0.99).mean()
            print(f"  {k:6s} {str(v):>6}: {r.d_mo*100:+6.2f} | {r.t_mo*100:+6.2f} | {r.s_mo*100:+6.2f} | {r.d_mo_s*100:+6.2f} | "
                  f"{r.t_mo_s*100:+6.2f} | {r.run_dd_s*100:5.1f} | {wp*100:4.1f}%", file=out)
    out.close()


if __name__ == "__main__":
    main("phase3" in sys.argv[1:])
