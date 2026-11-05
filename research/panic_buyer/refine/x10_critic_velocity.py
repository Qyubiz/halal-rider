"""x10 critic: re-run the velocity designs, then stress the 1-minute fill model and the universe agent's hourly watch mode.

A. Reproduce the four reports (19 coins; +LUNA/FTT/SRM; fills 0.2% worse on buys and sells with LUNA/FTT/SRM) for the
   reference, the velocity picks and the hindsight designs (numbers in x10_velocity_summary.log).
B. Execution realism on 1-minute data (x10_critic_lib.xkernel):
   lat  - a new bid cannot fill in the minute whose open was used to place it;
   mlat - market buys fill at the next minute's open, trailing sales one minute later;
   both, and both + every fill 0.1% worse.
C. 10 Oct 2025 flash crash: trades entered that day and a re-run with no new entries 20:00-23:59 UTC that day.
D. Universe agent's H4 core (19 coins, depth 14%, +5%, 48h, 24h cooldown, 35% crash guard, one coin x 100%): hourly
   watch (their kernel) vs the same bot at 1-minute resolution (watch with hourly cycle; watch every minute; one
   resting bid re-pointed every minute) and the live bot's mechanism (hourly resting bid).
E. How fragile the round-1 honest pick is: 2024-26 result of the next-best designs under the same 2021-23 rule.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import x10_critic_lib as X  # noqa: E402
import x10_velocity_lib as V  # noqa: E402
import x10_universe_lib as U  # noqa: E402

R1 = dict(N=2, fmax=0.5, cyc=1, rearm=1, cdm=-1, cd=24, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1)
R2 = dict(N=2, fmax=0.5, cyc=15, rearm=1, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1, hm=48, look2=3, depth2=0.08,
          cdm=-1, cd=24)
R3 = dict(cyc=1, rearm=1, N=1, fmax=0.7, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1, hm=24, look2=3, depth2=0.08,
          cdm=0, redip=0.03, rewin=60, cdl=4320)
HB = dict(cyc=1, rearm=1, N=1, fmax=1.0, xmode=1, arm=0.03, trail=0.0, cap=0.08, delay=1, hm=24, look2=3, depth2=0.08,
          cdm=0, redip=0.03, rewin=60, cdl=1440)
HB168 = dict(HB, hm=12, cdl=10080)
DESIGNS = [("REF live", {}), ("REF 2-coin hourly pick (N2 tp5)", dict(N=2, fmax=0.5, tp=0.05)),
           ("R1 PICK honest", R1), ("R2 pick", R2), ("R3 pick (hindsight-located grid)", R3),
           ("HINDSIGHT best 11.71", HB), ("HINDSIGHT LUNA-survivor 10.51", HB168)]
STRESS = [("base", {}), ("lat (no fill in placement minute)", dict(lat=1)), ("mlat (market orders 1 min late)", dict(mlat=1)),
          ("lat+mlat", dict(lat=1, mlat=1)), ("lat+mlat + fills 0.1% worse", dict(lat=1, mlat=1, pen=0.001, spen=0.001)),
          ("lat+mlat + fills 0.2% worse", dict(lat=1, mlat=1, pen=0.002, spen=0.002))]


def rep(e, lg):
    return V.report(e, lg, X_hidx)


def line(r):
    s = V.fmt(r)
    wt = f"{r['t_worst_trade']*100:+.1f}%" if r.get("t_trades", 0) else "n/a"
    return (f"{s} | 2022 {r['y2022']*100:+.0f}% | worst trade %acct 24-26 {wt}, 21-23 {r.get('d_worst_trade', np.nan)*100:+.1f}%"
            f" ({r.get('d_worst_trade_coin', '')}) | 24-26 x{r['t_mult']:.2f} trades {r.get('t_trades', 0)}")


def main():
    global X_hidx
    t0 = time.time()
    A, hidx, Ph, SLIP = V.load()
    X_hidx = hidx
    rows = []

    print("######## A. reproduce the four reports", flush=True)
    for name, kw in DESIGNS:
        print(f"\n{name}: {kw}")
        for lab, K, over in (("(1-3) 19 coins", 19, {}), ("+LUNA/FTT/SRM", 22, {}),
                             ("(4) fills 0.2% worse +LUNA/FTT/SRM", 22, dict(pen=0.002, spen=0.002))):
            e, lg, _ = X.xrun(A, hidx, Ph, SLIP, K=K, **{**kw, **over})
            r = rep(e, lg)
            r.update(design=name, row=lab)
            rows.append(r)
            print(f"  {lab:36s} {line(r)}", flush=True)

    print("\n######## B. execution realism (19 coins)", flush=True)
    for name, kw in DESIGNS:
        print(f"\n{name}")
        for lab, over in STRESS:
            e, lg, _ = X.xrun(A, hidx, Ph, SLIP, K=19, **{**kw, **over})
            r = rep(e, lg)
            r.update(design=name, row="B " + lab)
            rows.append(r)
            print(f"  {lab:36s} {line(r)}", flush=True)
        e, lg, _ = X.xrun(A, hidx, Ph, SLIP, K=22, **{**kw, "lat": 1, "mlat": 1, "pen": 0.001, "spen": 0.001})
        r = rep(e, lg); r.update(design=name, row="B lat+mlat+0.1% worse +LUNA/FTT/SRM"); rows.append(r)
        print(f"  {'lat+mlat+0.1% worse +LUNA/FTT/SRM':36s} {line(r)}", flush=True)

    print("\n######## C. 10 Oct 2025 flash crash (19 coins, normal costs)", flush=True)
    h0, h1 = hidx.searchsorted(pd.Timestamp("2025-10-10 20:00")), hidx.searchsorted(pd.Timestamp("2025-10-11 00:00"))
    refc = Ph["ref"].values[:, :19].copy(); refc[h0:h1] = np.nan
    for name, kw in DESIGNS:
        e, lg, _ = X.xrun(A, hidx, Ph, SLIP, K=19, **kw)
        r = rep(e, lg)
        fm = pd.to_datetime(hidx[0]) + pd.to_timedelta(lg.fm.astype(int).values, unit="min")
        sel = (fm >= pd.Timestamp("2025-10-10 00:00")) & (fm < pd.Timestamp("2025-10-12 00:00"))
        g = lg[sel]
        tr = "; ".join(f"{V.SYMS[int(x.k)]} in {pd.Timestamp(hidx[0]) + pd.Timedelta(minutes=int(x.fm))} @ {x.entry:.4g} -> "
                       f"{x.ret*100:+.1f}% ({x.ret*x.amt/x.eqb*100:+.1f}% of acct)" for x in g.itertuples())
        e2, lg2, _ = X.xrun(A, hidx, Ph, SLIP, K=19, ref=refc, **kw)
        r2 = rep(e2, lg2)
        print(f"  {name:34s} base 24-26 {r['t_pm']*100:+.2f}%/mo, since 2025 {r['s_pm']*100:+.2f}% | no entries 10 Oct 20-24h: "
              f"24-26 {r2['t_pm']*100:+.2f}%/mo DD {r2['t_dd']*100:.0f}%, since 2025 {r2['s_pm']*100:+.2f}%\n     trades 10-11 Oct: {tr or 'none'}",
              flush=True)
        r2.update(design=name, row="C no entries 10 Oct 2025 20-24h"); rows.append(r2)

    print("\n######## D. universe H4 core on the 19 coins: hourly watch vs 1-minute execution", flush=True)
    H4 = dict(depth=0.14, tp=0.05, hm=48, cd=24)
    for K in (19, 22):
        ref = X.guard_ref(Ph, 0.35, K)
        sm = (~np.isnan(ref)).astype(float)
        P = {f: v.iloc[:, :K] for f, v in Ph.items()}
        print(f"  --- {K} coins")
        for pen in (0.0, 0.002):
            tag = "" if pen == 0 else " fills 0.2% worse"
            ue, ul = U.urun(P, hidx, SLIP[:, :K], N=1, fmax=1.0, smult=sm, watch=True, pen=pen, **H4)
            ul = ul.rename(columns={"eq0": "eqb"})
            if pen:      # universe kernel has no sell penalty: emulate the buy side only and say so
                tag += " (buys only in their kernel)"
            r = rep(ue, ul); r.update(design="H4 core 19", row=f"D{K} hourly watch (universe kernel){tag}"); rows.append(r)
            print(f"  hourly watch, universe kernel{tag:30s} {line(r)}", flush=True)
            for lab, kw in (("1-min watch, hourly cycle", dict(watch=1, cyc=60)),
                            ("1-min watch, every minute + re-arm", dict(watch=1, cyc=1, rearm=1)),
                            ("1-min watch, every minute, lat+mlat", dict(watch=1, cyc=1, rearm=1, lat=1, mlat=1)),
                            ("1 resting bid re-pointed every minute", dict(cyc=1, rearm=1)),
                            ("hourly resting bid (live mechanism)", dict(cyc=60))):
                over = dict(pen=pen, spen=pen) if pen else {}
                e, lg, _ = X.xrun(A, hidx, Ph, SLIP, K=K, ref=ref, N=1, fmax=1.0, **H4, **kw, **over)
                r = rep(e, lg); r.update(design="H4 core 19", row=f"D{K} {lab}{tag}"); rows.append(r)
                print(f"  {lab + tag:60s} {line(r)}", flush=True)

    print("\n######## E. round-1 selection fragility (velocity grid.csv, 2021-23 rule, top 8 eligible)", flush=True)
    G = pd.read_csv(os.path.join(X.VEL_OUT, "grid.csv"))
    G = G[~G.name.str.startswith("REF")].copy()
    G["score"] = np.minimum(G.d_pm, G.x_d_pm)
    ok = G[(G.d_dd >= -0.50) & (G.x_d_dd >= -0.60) & (G.y2022 >= 0)].sort_values(["score", "d_dd"], ascending=[False, False])
    for _, r in ok.head(8).iterrows():
        print(f"  {r['name']:52s} score {r.score*100:6.2f} | 24-26 {r.t_pm*100:+5.2f}%/mo DD {r.t_dd*100:4.0f}% | since 2025 "
              f"{r.s_pm*100:+5.2f}%/mo")
    print(f"  eligible {len(ok)}; their 2024-26 %/month: median {ok.t_pm.median()*100:.2f}, 25th pct {ok.t_pm.quantile(.25)*100:.2f}, "
          f"75th pct {ok.t_pm.quantile(.75)*100:.2f}, max {ok.t_pm.max()*100:.2f}")

    pd.DataFrame(rows).to_csv(os.path.join(X.OUT, "critic_velocity.csv"), index=False)
    print(f"\ndone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
