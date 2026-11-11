"""Lever "velocity", pre-registered grid (x10 study). Written after the 2021-23-only development pass
(x10_velocity_dev.py) and BEFORE any 2024-26 number of a velocity design was seen. Nothing below is changed after the run.

GRID (420 designs; depth 12%, time stop 48h, volume-rank slippage, 1-minute engine x10_velocity_lib.vkernel):
  sizing   S1 N1 fmax 1.00 | S2 N2 fmax 0.50 | S3 N3 fmax 0.33 | S4 N2 fmax 1.00 near 1% | S5 N3 fmax 1.00 near 1%
  cycle    C1 hourly (live) | C2 hourly + re-arm after a sale | C3 every 15 min + re-arm | C4 every minute
  cooldown 24h (live) | 60 min | 0
  exit     E1 tp 1.5% next hour | E2 tp 3% next hour | E3 tp 5% next hour | E4 tp 3% active 30 min after the fill |
           E5 trailing arm 3% trail 0 (first lower minute close) cap 8% | E6 trailing arm 5% trail 0 cap 8% |
           E7 trailing arm 3% trail 0.5% cap 8%   (trailing exits active 1 minute after the fill)
  Every design runs on the 19 live coins and on 22 coins (+ LUNA/FTT/SRM, cut as in engine.DEAD).

SELECTION RULE (2021-23 only; the selection code reads only 2021-23 columns):
  eligible: 19-coin 2021-23 worst drop >= -50%, 22-coin 2021-23 worst drop >= -60%, 19-coin calendar 2022 >= 0%.
  score: min(19-coin 2021-23 %/month, 22-coin 2021-23 %/month); highest wins (tie: shallower 19-coin 2021-23 drop).
  If nothing is eligible: no pick.
THEN, once for the pick: 2024-26 and since-2025 (19 and 22 coins), every fill 0.2% worse (buys AND sells) with
LUNA/FTT/SRM, worst single trade as % of the account. The hindsight best of the grid (highest 19-coin 2024-26 %/month)
is reported separately and labelled hindsight, with the same stress rows.
References (not candidates): live settings and the earlier 2-coin pick (N2 tp 5% cd 24h) on the same engine and costs.
%/month = multiple ** (1 / months) - 1, months from the period's span (2024-26: 33.7, since 2025: 21.7, 2021-23: 36).
"""
import hashlib
import itertools
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_velocity_lib as V  # noqa: E402

SIZING = {"S1 N1": dict(N=1, fmax=1.0), "S2 N2": dict(N=2, fmax=0.5), "S3 N3": dict(N=3, fmax=0.33),
          "S4 N2 near1%": dict(N=2, fmax=1.0, near=0.01), "S5 N3 near1%": dict(N=3, fmax=1.0, near=0.01)}
CYCLE = {"C1 hourly": dict(cyc=60, rearm=0), "C2 hourly+rearm": dict(cyc=60, rearm=1),
         "C3 15min+rearm": dict(cyc=15, rearm=1), "C4 1min": dict(cyc=1, rearm=1)}
COOL = {"cd24h": dict(cdm=-1, cd=24), "cd60m": dict(cdm=60), "cd0": dict(cdm=0)}
EXIT = {"E1 tp1.5": dict(tp=0.015), "E2 tp3": dict(tp=0.03), "E3 tp5": dict(tp=0.05), "E4 tp3 d30": dict(tp=0.03, delay=30),
        "E5 trail a3 t0 c8": dict(xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1),
        "E6 trail a5 t0 c8": dict(xmode=1, arm=0.05, trail=0.0, cap=0.08, delay=1),
        "E7 trail a3 t.5 c8": dict(xmode=1, arm=0.03, trail=0.005, cap=0.08, delay=1)}
GRID = {f"{s} | {c} | {d} | {x}": {**SIZING[s], **CYCLE[c], **COOL[d], **EXIT[x]}
        for s, c, d, x in itertools.product(SIZING, CYCLE, COOL, EXIT)}
REFS = {"REF live (N8 f0.2 tp1.5 hourly cd24h)": dict(),
        "REF earlier 2-coin pick (N2 tp5 hourly cd24h)": dict(N=2, fmax=0.5, tp=0.05)}
DD19, DD22, MIN2022 = -0.50, -0.60, 0.0

A, hidx, Ph, SLIP = V.load()


def one(item):
    name, kw = item
    r = {"name": name, **{k: kw.get(k, V.DEFAULTS[k]) for k in V.DEFAULTS}}
    for K, tag in ((19, ""), (22, "x_")):
        e, lg, inv = V.vrun(A, hidx, Ph, SLIP, K=K, **kw)
        rr = V.report(e, lg, hidx)
        r.update({tag + k: v for k, v in rr.items()})
        r[tag + "inv"] = float(inv.mean())
    return r


def stress(name, kw):
    """The four reports for one design: 19 coins; 22 coins; fills 0.2% worse (buys and sells) on 19 and on 22 coins."""
    rows = []
    for lab, K, over in (("19 coins", 19, {}), ("+LUNA/FTT/SRM", 22, {}), ("fills 0.2% worse", 19, dict(pen=0.002, spen=0.002)),
                         ("fills 0.2% worse +LUNA/FTT/SRM", 22, dict(pen=0.002, spen=0.002))):
        e, lg, inv = V.vrun(A, hidx, Ph, SLIP, K=K, **{**kw, **over})
        r = V.report(e, lg, hidx)
        r.update(design=name, row=lab, inv=float(inv.mean()))
        rows.append(r)
        print(f"    {lab:32s} {V.fmt(r)} | 2022 {r['y2022']*100:+.0f}%", flush=True)
        print(f"    {'':32s} worst trade (% of account): 24-26 {r['t_worst_trade']*100:+.1f}% ({r['t_worst_trade_coin']}), "
              f"21-23 {r['d_worst_trade']*100:+.1f}% ({r['d_worst_trade_coin']}); trades 24-26 {r['t_trades']}, "
              f"win {r['t_win']*100:.0f}%, avg {r['t_avg']*100:+.2f}%; invested {r['inv']*100:.0f}% of the time-weighted equity", flush=True)
    return rows


def main():
    sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"x10_velocity_grid.py sha256 {sha}\n{len(GRID)} designs x 2 panels", flush=True)
    t0 = time.time()
    items = list(GRID.items()) + list(REFS.items())
    with mp.get_context("fork").Pool(2) as pool:
        rows = []
        for i, r in enumerate(pool.imap(one, items, chunksize=4)):
            rows.append(r)
            if i % 40 == 0:
                print(f"  {i+1}/{len(items)} done, {time.time()-t0:.0f}s", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(V.OUT, "grid.csv"), index=False)
    G = R[~R.name.str.startswith("REF")]
    print(f"grid done in {time.time()-t0:.0f}s", flush=True)

    # ---------------------------------------------------------------- selection on 2021-23 only
    D = G[["name", "d_pm", "d_dd", "y2022", "x_d_pm", "x_d_dd"]].copy()       # no 2024-26 column is visible here
    D["score"] = np.minimum(D.d_pm, D.x_d_pm)
    ok = D[(D.d_dd >= DD19) & (D.x_d_dd >= DD22) & (D.y2022 >= MIN2022)].sort_values(["score", "d_dd"], ascending=[False, False])
    print(f"\n######## SELECTION (2021-23 only): {len(ok)} of {len(D)} designs eligible")
    for _, r in ok.head(12).iterrows():
        print(f"  {r['name']:58s} 21-23 {r.d_pm*100:+6.2f}%/mo DD {r.d_dd*100:4.0f}% | +LUNA/FTT/SRM {r.x_d_pm*100:+6.2f}%/mo "
              f"DD {r.x_d_dd*100:4.0f}% | 2022 {r.y2022*100:+4.0f}%")
    pick = ok.iloc[0]["name"] if len(ok) else None
    print(f"PICK: {pick}")

    # ---------------------------------------------------------------- one look: pick, references, hindsight
    out = []
    for label, name in (("PICK (2021-23 rule)", pick), ("REF live", list(REFS)[0]), ("REF 2-coin pick", list(REFS)[1])):
        if name is None:
            continue
        print(f"\n######## {label}: {name}")
        out += stress(name, (GRID | REFS)[name])
    H = G.sort_values("t_pm", ascending=False)
    print("\n######## HINDSIGHT: top 10 of the grid by 19-coin 2024-26 %/month (chosen by looking at 2024-26)")
    for _, r in H.head(10).iterrows():
        print(f"  {r['name']:58s} {V.fmt(r)} | +LUNA/FTT/SRM 24-26 {r.x_t_pm*100:+5.2f}%/mo DD {r.x_t_dd*100:4.0f}%")
    hb = H.iloc[0]["name"]
    print(f"\n######## HINDSIGHT BEST: {hb}")
    out += stress(hb, GRID[hb])
    H50 = G[G.t_dd >= -0.50].sort_values("t_pm", ascending=False)
    if len(H50) and H50.iloc[0]["name"] != hb:
        print(f"\n######## HINDSIGHT BEST with 2024-26 drop >= -50%: {H50.iloc[0]['name']}")
        out += stress(H50.iloc[0]["name"], GRID[H50.iloc[0]["name"]])
    pd.DataFrame(out).to_csv(os.path.join(V.OUT, "grid_stress.csv"), index=False)
    print(f"\ngrid: 2024-26 %/month max {G.t_pm.max()*100:.2f}, median {G.t_pm.median()*100:.2f}; designs >= 10%/month in 2024-26: "
          f"{int((G.t_pm >= 0.10).sum())}; >= 7%: {int((G.t_pm >= 0.07).sum())}; since 2025 max {G.s_pm.max()*100:.2f}%/month")
    for ax, d in (("sizing", SIZING), ("cycle", CYCLE), ("cooldown", COOL), ("exit", EXIT)):
        print(f"  by {ax}: " + " | ".join(f"{k} 24-26 med {G[G.name.str.contains(k, regex=False)].t_pm.median()*100:+.2f} "
                                            f"max {G[G.name.str.contains(k, regex=False)].t_pm.max()*100:+.2f}" for k in d))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
