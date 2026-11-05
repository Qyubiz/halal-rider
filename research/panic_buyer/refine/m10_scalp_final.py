"""m10 scalp, step 3: frontier (2021-23 only), the pick, and its one-time report on 2024-26, since 2025 and the untouched
2018-20 holdout, with every fill 0.05% and 0.1% worse.

Pick rule (fixed before the 2024-26 / holdout runs): among the "conc" designs (m10_scalp_explore.py, 2021-23 only) with
a 2021-23 win rate >= 95% and a worst drop (1-minute marks) no deeper than -40%, the highest compounded %/month.
Fill model: limit buys fill at the limit only if the low trades 0.1% through, take-profits only at the limit when the high
trades 0.1% through, active from the minute after the buy; wicks beyond 0.5% of the candle body removed; market orders
0.02% (top-50 volume) / 0.1% slippage; fees 0.1% per side. "x% worse": every buy fill x higher and every sell x lower."""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_scalp_lib as S  # noqa: E402

pd.set_option("display.width", 250)


def frontier():
    d = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(S.OUT, "explore_conc*.csv")))], ignore_index=True)
    d = d.drop_duplicates(["W", "depth", "tp", "hmm", "N", "fmax", "guard"])
    print(f"######## 2021-23 frontier over {len(d)} concentrated designs (crash guard 35%, 19 coins)")
    cols = ["W", "depth", "tp", "hmm", "N", "fmax", "win", "pm", "dd", "dd_min", "worst_trade", "tpm", "avg"]
    for wr in (0.90, 0.93, 0.95, 0.97, 0.98, 0.99):
        x = d[d.win >= wr]
        if not len(x):
            print(f"  win >= {wr*100:.0f}%: none"); continue
        a = x.loc[x.pm.idxmax()]
        y = x[x.dd_min >= -0.40]
        b = y.loc[y.pm.idxmax()] if len(y) else None
        f = lambda r: (f"W{r.W:.0f} d{r.depth*100:.0f}% tp{r.tp*100:.1f}% hold{r.hmm/60:.0f}h N{r.N:.0f}: win {r.win*100:.1f}% "
                       f"{r.pm*100:+.2f}%/mo DD {r.dd_min*100:.0f}% worst trade {r.worst_trade*100:+.1f}% {r.tpm:.1f} tr/mo")
        print(f"  win >= {wr*100:.0f}%  best: {f(a)}\n               best with DD >= -40%: {f(b) if b is not None else 'none'}")
    ok = d[(d.win >= 0.95) & (d.dd_min >= -0.40)]
    pick = ok.loc[ok.pm.idxmax()]
    print("\nPICK (rule: 2021-23 win >= 95%, 1-min DD >= -40%, max %/month):")
    print(pick[cols].to_string())
    top = ok.sort_values("pm", ascending=False).head(8)[cols]
    print("\nnext best under the same rule (2021-23):\n" + top.round(4).to_string(index=False))
    return dict(W=int(pick.W), depth=float(pick.depth), tp=float(pick.tp), hmm=int(pick.hmm), N=int(pick.N),
                fmax=float(pick.fmax), guard=float(pick.guard))


def report(spec):
    out = []
    A, hidx, Ph, SLIP = S.load(19)
    g = S.guard_ref(Ph)
    HA, hhidx, HPh, HSLIP, hsyms = S.load_holdout()
    hg = S.guard_ref(HPh)
    for worse in (0.0, 0.0005, 0.001):
        for wick in (0.005,) if worse else (0.005, 0.01, 0.0):
            kw = S.scalp(**spec, wick=wick, pen=worse, spen=worse)
            e, lg, inv, lo, ntr = S.srun(A, hidx, Ph, SLIP, gref=g, **kw)
            he, hlg, hinv, hlo, hntr = S.srun(HA, hhidx, HPh, HSLIP, gref=hg, **kw)
            tag = f"fills {worse*100:.2f}% worse, wicks>{wick*100:.1f}% trimmed" if wick else f"fills {worse*100:.2f}% worse, raw wicks"
            print(f"\n######## {tag}")
            for p in ("h", "d", "t", "s"):
                a, b = S.PERIODS[p]
                r = S.period_stats(he, hlg, hlo, hhidx, a, b) if p == "h" else S.period_stats(e, lg, lo, hidx, a, b)
                print("  " + S.line(r, S.PLAB[p]))
                out.append(dict(worse=worse, wick=wick, period=S.PLAB[p], **{k: v for k, v in r.items()}))
            if worse == 0 and wick == 0.005:
                for y in range(2018, 2027):
                    src = (he, hlg, hlo, hhidx) if y < 2021 else (e, lg, lo, hidx)
                    r = S.period_stats(src[0], src[1], src[2], src[3], pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01"))
                    print("    " + S.line(r, str(y)))
                lg2 = lg.copy(); lg2["t_in"] = hidx[lg.eb.astype(int).values]
                lg2["acct"] = lg2.ret * lg2.amt / lg2.eqb
                w = lg2.nsmallest(6, "acct")
                print("    worst trades 2021-26: " + "; ".join(f"{S.SYMS[int(r.k)][:-4]} {r.t_in:%Y-%m-%d} {r.ret*100:+.1f}% "
                                                          f"({r.acct*100:+.1f}% of acct)" for r in w.itertuples()))
                hl2 = hlg.copy(); hl2["t_in"] = hhidx[hlg.eb.astype(int).values]
                hl2["acct"] = hl2.ret * hl2.amt / hl2.eqb
                w = hl2.nsmallest(4, "acct")
                print("    worst trades 2018-20: " + "; ".join(f"{hsyms[int(r.k)][:-4]} {r.t_in:%Y-%m-%d} {r.ret*100:+.1f}% "
                                                          f"({r.acct*100:+.1f}% of acct)" for r in w.itertuples()))
    # collapse coins (LUNA/FTT/SRM added, 22 coins), normal fills
    A22, hidx22, Ph22, SLIP22 = S.load(22)
    e, lg, inv, lo, ntr = S.srun(A22, hidx22, Ph22, SLIP22, gref=S.guard_ref(Ph22), **S.scalp(**spec))
    print("\n######## with LUNA/FTT/SRM added (22 coins), normal fills")
    for p in ("d", "t"):
        print("  " + S.line(S.period_stats(e, lg, lo, hidx22, *S.PERIODS[p]), S.PLAB[p]))
    # reference: 90% mode through the same kernel (hourly bids on the 24h reference, engine fill rules, normal fills)
    ref90 = dict(depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=3, guard=0.35)
    e, lg, inv, lo, ntr = S.srun(A, hidx, Ph, SLIP, gref=g, **ref90)
    he, hlg, hinv, hlo, hntr = S.srun(HA, hhidx, HPh, HSLIP, gref=hg, **ref90)
    print("\n######## reference: 90% mode (hourly bids, 24h reference, engine fill rules) through the same kernel")
    for p in ("h", "d", "t", "s"):
        a, b = S.PERIODS[p]
        r = S.period_stats(he, hlg, hlo, hhidx, a, b) if p == "h" else S.period_stats(e, lg, lo, hidx, a, b)
        print("  " + S.line(r, S.PLAB[p]))
    pd.DataFrame(out).to_csv(os.path.join(S.OUT, "final_report.csv"), index=False)


if __name__ == "__main__":
    spec = frontier()
    print(f"\nspec: {spec}")
    report(spec)
