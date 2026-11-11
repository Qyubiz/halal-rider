"""Lever "velocity", round 3 (x10 study). HINDSIGHT-LOCATED: this grid is centred on the round-2 designs that made
>= 10%/month in 2024-26 (one coin, 1-minute cycle, 3h/8% flash trigger, re-entry after a profitable exit), which
round 2 showed lose ~100% once LUNA/FTT/SRM are added. Round 3 adds a cooldown after LOSING exits (cdl) and a 70% cap.
The choice inside the grid uses the same 2021-23 rule as rounds 1-2, but the grid's location was chosen by looking at
2024-26, so no number from this round is an honest out-of-sample result. Written before the round-3 run.

GRID (192 designs; depth 12%, 1-minute cycle with re-arm, volume-rank slippage):
  sizing     N1 fmax 1.00 | N1 fmax 0.70 | N2 fmax 0.50
  exit       trailing arm 3% trail 0 cap 8% | tp 3% next hour
  time stop  12h | 24h
  flash      look 3h depth 8% | off
  re-entry   cooldown 0 + re-entry at exit -3% for 60 min | the same at exit -2%
  loss cool  off | 24h | 72h | 168h after a losing exit
SELECTION RULE (2021-23 columns only): 19-coin 2021-23 worst drop >= -50%, 22-coin >= -60%, 19-coin 2022 >= 0%;
highest min(19-coin, 22-coin 2021-23 %/month). Reported for the pick and the hindsight best (19-coin 2024-26):
the four reports, plus fills 0.1% worse, per-year returns and the worst single trade.
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
import engine as E  # noqa: E402
import x10_velocity_lib as V  # noqa: E402
import x10_velocity_grid as G1  # noqa: E402

SIZING = {"N1 f1": dict(N=1, fmax=1.0), "N1 f0.7": dict(N=1, fmax=0.7), "N2": dict(N=2, fmax=0.5)}
EXIT = {"trail a3": dict(xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1), "tp3": dict(tp=0.03, tpf=0.03)}
HM = {"hm12": dict(hm=12), "hm24": dict(hm=24)}
FLASH = {"flash3h8": dict(look2=3, depth2=0.08), "noflash": dict()}
REENTRY = {"redip3/60m": dict(cdm=0, redip=0.03, rewin=60), "redip2/60m": dict(cdm=0, redip=0.02, rewin=60)}
LOSSCD = {"lossCD off": dict(cdl=-1), "lossCD 24h": dict(cdl=1440), "lossCD 72h": dict(cdl=4320), "lossCD 168h": dict(cdl=10080)}
BASE = dict(cyc=1, rearm=1)
GRID = {" | ".join(c): {**BASE, **SIZING[c[0]], **EXIT[c[1]], **HM[c[2]], **FLASH[c[3]], **REENTRY[c[4]], **LOSSCD[c[5]]}
        for c in itertools.product(SIZING, EXIT, HM, FLASH, REENTRY, LOSSCD)}
DD19, DD22, MIN2022 = -0.50, -0.60, 0.0


def stress(name, kw):
    rows = []
    for lab, K, over in (("19 coins", 19, {}), ("+LUNA/FTT/SRM", 22, {}), ("fills 0.1% worse", 19, dict(pen=0.001, spen=0.001)),
                         ("fills 0.2% worse", 19, dict(pen=0.002, spen=0.002)),
                         ("fills 0.2% worse +LUNA/FTT/SRM", 22, dict(pen=0.002, spen=0.002))):
        e, lg, inv = V.vrun(G1.A, G1.hidx, G1.Ph, G1.SLIP, K=K, **{**kw, **over})
        r = V.report(e, lg, G1.hidx)
        ys = E.years(e)
        r.update(design=name, row=lab, inv=float(inv.mean()), **{f"y{y}": ys.get(y, np.nan) for y in range(2021, 2027)})
        rows.append(r)
        wt = f"{r['t_worst_trade']*100:+.1f}% ({r['t_worst_trade_coin']})" if r.get("t_trades", 0) else "no trades (account wiped/stuck)"
        print(f"    {lab:32s} {V.fmt(r)} | 2022 {r['y2022']*100:+.0f}% | worst trade 24-26 {wt}, "
              f"21-23 {r['d_worst_trade']*100:+.1f}% ({r['d_worst_trade_coin']})", flush=True)
        print(f"    {'':32s} per year " + " ".join(f"{y} {ys.get(y, np.nan)*100:+.0f}%" for y in range(2021, 2027))
              + f" | trades 24-26 {r.get('t_trades', 0)}", flush=True)
    return rows


def main():
    sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"x10_velocity_grid3.py sha256 {sha}\n{len(GRID)} designs x 2 panels", flush=True)
    t0 = time.time()
    with mp.get_context("fork").Pool(2) as pool:
        rows = list(pool.imap(G1.one, list(GRID.items()), chunksize=4))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(V.OUT, "grid3.csv"), index=False)
    print(f"grid done in {time.time()-t0:.0f}s", flush=True)
    D = R[["name", "d_pm", "d_dd", "y2022", "x_d_pm", "x_d_dd"]].copy()
    D["score"] = np.minimum(D.d_pm, D.x_d_pm)
    ok = D[(D.d_dd >= DD19) & (D.x_d_dd >= DD22) & (D.y2022 >= MIN2022)].sort_values(["score", "d_dd"], ascending=[False, False])
    print(f"\n######## SELECTION (2021-23 only): {len(ok)} of {len(D)} designs eligible")
    for _, r in ok.head(8).iterrows():
        print(f"  {r['name']:64s} 21-23 {r.d_pm*100:+6.2f}%/mo DD {r.d_dd*100:4.0f}% | +LUNA/FTT/SRM {r.x_d_pm*100:+6.2f}%/mo "
              f"DD {r.x_d_dd*100:4.0f}% | 2022 {r.y2022*100:+4.0f}%")
    out = []
    if len(ok):
        pick = ok.iloc[0]["name"]
        print(f"\n######## ROUND-3 PICK (2021-23 rule inside a hindsight-located grid): {pick}")
        out += stress(pick, GRID[pick])
    print("\n######## HINDSIGHT: top 12 of the round-3 grid by 19-coin 2024-26 %/month")
    for _, r in R.sort_values("t_pm", ascending=False).head(12).iterrows():
        print(f"  {r['name']:64s} {V.fmt(r)} | +LUNA/FTT/SRM 21-23 {r.x_d_pm*100:+6.2f}%/mo DD {r.x_d_dd*100:4.0f}%")
    hb = R.sort_values("t_pm", ascending=False).iloc[0]["name"]
    print(f"\n######## HINDSIGHT BEST (round 3): {hb}")
    out += stress(hb, GRID[hb])
    S = R[(R.x_d_dd > -0.95)].sort_values("t_pm", ascending=False)
    if len(S) and S.iloc[0]["name"] != hb:
        print(f"\n######## HINDSIGHT BEST that survives LUNA/FTT/SRM (22-coin 2021-23 drop shallower than -95%): {S.iloc[0]['name']}")
        out += stress(S.iloc[0]["name"], GRID[S.iloc[0]["name"]])
    pd.DataFrame(out).to_csv(os.path.join(V.OUT, "grid3_stress.csv"), index=False)
    print(f"\nround-3 grid: 2024-26 %/month max {R.t_pm.max()*100:.2f}, median {R.t_pm.median()*100:.2f}; >= 10%: "
          f"{int((R.t_pm >= 0.10).sum())}; of those surviving LUNA/FTT/SRM (22-coin 21-23 drop > -95%): "
          f"{int(((R.t_pm >= 0.10) & (R.x_d_dd > -0.95)).sum())}; since 2025 max {R.s_pm.max()*100:.2f}%/month")
    for ax, d in (("sizing", SIZING), ("exit", EXIT), ("time stop", HM), ("flash", FLASH), ("re-entry", REENTRY), ("loss cooldown", LOSSCD)):
        print(f"  by {ax}: " + " | ".join(
            f"{k}: 24-26 med {R[R.name.str.split(' [|] ').apply(lambda x: k in x)].t_pm.median()*100:+.2f} max "
            f"{R[R.name.str.split(' [|] ').apply(lambda x: k in x)].t_pm.max()*100:+.2f}, 22-coin 21-23 DD med "
            f"{R[R.name.str.split(' [|] ').apply(lambda x: k in x)].x_d_dd.median()*100:.0f}%" for k in d))
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
