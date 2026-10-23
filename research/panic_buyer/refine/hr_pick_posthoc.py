"""POST-HOC checks on the hr_pick.py pick (written AFTER hr_pick.py was run; none of this changes the pick).

1. Rule sensitivity: what the same rule picks with other drop limits / without the 2022 filter, and how the top 10
   2021-23 settings did in 2024-26 (is reaching 5%/month typical of the rule's choices, or luck of one pick?).
2. The fmax 1.0 twins of N >= 2 that hr_pick.py skipped as duplicates turned out NOT identical in 2021-23 (36 of 96
   pairs differ); re-apply the rule with them included.
3. Jitter: depth +-0.1/0.2 point, tp +-0.1 point, fee +-0.005%, eps 0.09%/0.11% (changes that should not matter).
4. Rolling windows: return over 1/3/6/12 months from every start day (what an investor starting on a random day saw).
5. Stricter 1-minute fills (round 2 "realistic-strict"): buys 0.3% through, take-profits 0.2% through, 1-minute
   latency; and the same with the live bot's market-buy timing.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import hr_pick as H  # noqa: E402
import r2_realism_lib as RL  # noqa: E402
import study  # noqa: E402

OUT = study.OUT
PICK = (0.12, 2, 0.5, 0.05, 48, 24)


def rule(D, max_dd=-0.50, min22=0.0):
    ok = D[(D.d_dd >= max_dd) & (D.y2022 >= min22)].sort_values(["d_cagr", "d_dd"], ascending=[False, False])
    return ok


def mo(m, months):
    return m ** (1 / months) - 1


def main():
    R = pd.read_csv(os.path.join(OUT, "pick_grid.csv"))
    print("######## 1. rule sensitivity (pick grid, 192 settings)")
    for max_dd, min22 in ((-0.50, 0.0), (-0.45, 0.0), (-0.40, 0.0), (-0.35, 0.0), (-0.50, -1.0), (-0.40, -1.0)):
        ok = rule(R, max_dd, min22)
        if not len(ok):
            print(f"  drop >= {max_dd*100:.0f}%, 2022 >= {min22*100:.0f}%: nothing"); continue
        p = ok.iloc[0]
        top = ok.head(10)
        print(f"  drop >= {max_dd*100:.0f}%, 2022 >= {min22*100:.0f}%: {len(ok)} qualify; pick {H.name(tuple(p[k] for k in H.KEYS))} "
              f"21-23 {p.d_mo*100:+.2f}%/mo DD {p.d_dd*100:.0f}% -> 24-26 {p.t_mo*100:+.2f}%/mo DD {p.t_dd*100:.0f}%; "
              f"top-10: {int((top.t_mo >= 0.05).sum())}/10 reach 5%/mo, 24-26 %/mo median {top.t_mo.median()*100:.2f} "
              f"[{top.t_mo.min()*100:.2f}..{top.t_mo.max()*100:.2f}], drop median {top.t_dd.median()*100:.0f}% worst {top.t_dd.min()*100:.0f}%")
    ok = rule(R)
    print("  top 10 under the pre-registered rule, with their 2024-26:")
    for _, r in ok.head(10).iterrows():
        print(f"    {H.name(tuple(r[k] for k in H.KEYS)):38s} {H.dline(r)} | {H.tline(r)}")

    print("\n######## 2. rule re-applied with the skipped fmax 1.0 twins (N>=2) included")
    rows = []
    for s in H.PICK_GRID:
        if s[1] >= 2:
            rows.append(H.run((s[0], s[1], 1.0) + s[3:])[0])
    T = pd.concat([R, pd.DataFrame(rows)], ignore_index=True)
    ok2 = rule(T)
    p = ok2.iloc[0]
    print(f"  {len(ok2)} of {len(T)} qualify; pick {H.name(tuple(p[k] for k in H.KEYS))} {H.dline(p)} | {H.tline(p)}")
    for _, r in ok2.head(5).iterrows():
        print(f"    {H.name(tuple(r[k] for k in H.KEYS)):38s} {H.dline(r)} | {H.tline(r)}")

    print("\n######## 3. jitter around the pick")
    jit = [("as picked", {}), ("depth 11.8%", dict(depth=0.118)), ("depth 11.9%", dict(depth=0.119)),
           ("depth 12.1%", dict(depth=0.121)), ("depth 12.2%", dict(depth=0.122)), ("tp 4.9%", dict(tp=0.049)),
           ("tp 5.1%", dict(tp=0.051)), ("fee 0.095%", dict(fee=0.00095)), ("fee 0.105%", dict(fee=0.00105)),
           ("eps 0.09%", dict(eps=0.0009)), ("eps 0.11%", dict(eps=0.0011))]
    J = []
    for lab_, over in jit:
        r = H.run(PICK, **over)[0]
        r["jitter"] = lab_
        J.append(r)
        print(f"  {lab_:12s} {H.dline(r)} | {H.tline(r)}")
    J = pd.DataFrame(J)
    J.to_csv(os.path.join(OUT, "posthoc_jitter.csv"), index=False)
    print(f"  jitter band 2024-26: {J.t_mo.min()*100:.2f} to {J.t_mo.max()*100:.2f} %/mo (x{J.t_mult.min():.2f} to x{J.t_mult.max():.2f}), "
          f"drop {J.t_dd.min()*100:.0f}% to {J.t_dd.max()*100:.0f}%; 2021-23 drop {J.d_dd.min()*100:.0f}% to {J.d_dd.max()*100:.0f}%")

    print("\n######## 4. rolling windows from every start day (daily closes of the equity curve)")
    W = []
    for lab_, s, over in (("PICK", PICK, {}), ("PICK bot timing", PICK, dict(mtp_now=True, cd=25)),
                          ("LIVE", H.LIVE, {})):
        e = H.run(s, **over)[1].resample("1D").last()
        for per, ee in (("2021-26", e), ("2024-26", e[e.index >= E.DESIGN_END])):
            for days in (30, 91, 182, 365):
                r = (ee.shift(-days) / ee - 1).dropna()
                W.append(dict(who=lab_, period=per, days=days, worst=r.min(), p05=r.quantile(0.05), p50=r.median(),
                              share_loss=(r < 0).mean(), share_below_m10=(r < -0.10).mean(), n=len(r)))
                print(f"  {lab_:16s} {per} {days:3d}d: worst {r.min()*100:+6.1f}%  5th pct {r.quantile(0.05)*100:+6.1f}%  "
                      f"median {r.median()*100:+6.1f}%  start days with a loss {(r < 0).mean()*100:4.1f}%  "
                      f"with < -10% {(r < -0.10).mean()*100:4.1f}%  (n={len(r)})")
    pd.DataFrame(W).to_csv(os.path.join(OUT, "posthoc_windows.csv"), index=False)

    print("\n######## 5. stricter fills on 1-minute candles (round-2 realistic-strict), pick vs live")
    syms, _, _ = study.market(False)
    _, A, hidx, Ph = E.minute_panel(syms)
    for lab_, s in (("PICK", PICK), ("LIVE", H.LIVE)):
        kw = H.kw(s)
        for sc, over in (("1-min engine, live fills", {}),
                         ("strict 0.3%+tp0.2%+1min", dict(beps=0.003, seps=0.002, lat=1, mpx=1)),
                         ("strict + bot mkt timing", dict(beps=0.003, seps=0.002, lat=1, mpx=1, mdelay=1, mtsh=1))):
            e, lg = RL.rrun(A, hidx, Ph, **kw, **over)
            r = E.score(e, lg, hidx)
            r["d_mo"] = (1 + r["d_cagr"]) ** (1 / 12) - 1; r["t_mo"] = (1 + r["t_cagr"]) ** (1 / 12) - 1
            print(f"  {lab_} {sc:26s} {H.dline(r)} | {H.tline(r)}")


if __name__ == "__main__":
    main()
