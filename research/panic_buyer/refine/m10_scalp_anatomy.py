"""m10 scalp: anatomy of a design on the DESIGN period (2021-23) only - per year, entry type (resting limit vs market at
or under the trigger), exit type, and the state of the market at entry for winners and losers:
  dd7    the coin's price at the fill vs its highest hourly close of the previous 168 hours (crash-guard measure)
  btcw   BTC's last close before the fill minute vs its max close of the previous W minutes (market-wide dip)
  breadth  share of the 19 coins whose last close is under their own W-minute trigger level x (1 + depth/2)
  tr24   the coin's last close vs its close 24h earlier
Everything is measured from data before the fill minute. Usage: python m10_scalp_anatomy.py W depth tp hmm N fmax"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_scalp_lib as S  # noqa: E402


def anatomy(W, depth, tp, hmm, N, fmax, **extra):
    A, hidx, Ph, SLIP = S.load(19)
    i1 = hidx.searchsorted(S.T24)
    g = S.guard_ref(Ph)
    kw = S.scalp(W=W, depth=depth, tp=tp, hmm=hmm, N=N, fmax=fmax, **extra)
    e, lg, inv, lo, ntr = S.srun(A, hidx, Ph, SLIP, i0=0, i1=i1, gref=g, **kw)
    hx = hidx[:i1]
    print(f"design {kw}")
    print(S.line(S.period_stats(e, lg, lo, hx, S.T21, S.T24), "21-23"))
    for y in (2021, 2022, 2023):
        r = S.period_stats(e, lg, lo, hx, pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01"))
        print(S.line(r, str(y)))
    c = np.asarray(A["c"][:60 * i1]); o = np.asarray(A["o"][:60 * i1])
    fm = lg.fm.astype(int).values; k = lg.k.astype(int).values
    cprev = pd.DataFrame(c).shift(1).values
    rmax = pd.DataFrame(c).rolling(W, min_periods=1).max().shift(1).values
    sl = SLIP[:i1][fm // 60, k]
    mkt = np.isclose(lg.entry.values, o[fm, k].astype(np.float64) * (1 + sl), rtol=1e-12)
    d = pd.DataFrame({"ret": lg.ret.values, "why": lg.why.values, "mkt": mkt, "k": k,
                      "hold_h": (lg.xm.values - fm) / 60.0,
                      "dd7": lg.entry.values / g[fm // 60, k] - 1,
                      "btcw": cprev[fm, 0] / rmax[fm, 0] - 1,
                      "breadth": (cprev[fm] < rmax[fm] * (1 - depth / 2)).mean(axis=1),
                      "tr24": cprev[fm, k] / c[np.maximum(fm - 1441, 0), k] - 1})
    d["win"] = d.ret > 0
    print(f"\nentries: resting limit {(~d.mkt).sum()} (win {d[~d.mkt].win.mean()*100:.1f}%, avg {d[~d.mkt].ret.mean()*100:+.3f}%), "
          f"market at/under trigger {d.mkt.sum()} (win {d[d.mkt].win.mean()*100:.1f}%, avg {d[d.mkt].ret.mean()*100:+.3f}%)")
    print(f"exits: {d.why.value_counts().to_dict()} (0 take-profit, 1 time stop, 2 stop-loss, 3 break-even)")
    print(f"losers: {(~d.win).sum()}, their avg {d[~d.win].ret.mean()*100:+.2f}%, sum of losses / sum of wins "
          f"{-d[~d.win].ret.sum() / d[d.win].ret.sum():.2f}; 10 worst {np.round(np.sort(d.ret.values)[:10]*100, 1)}")
    print("\nfeature quantiles (winners | losers):")
    for f in ("dd7", "btcw", "breadth", "tr24", "hold_h"):
        qw = d[d.win][f].quantile([.1, .25, .5, .75, .9]).values; ql = d[~d.win][f].quantile([.1, .25, .5, .75, .9]).values
        print(f"  {f:8s} {np.round(qw, 3)} | {np.round(ql, 3)}")
    print("\nby feature bucket (trades, win rate, avg net, sum of net):")
    for f, bins in (("dd7", [-1, -0.35, -0.25, -0.15, -0.08, 0, 1]), ("btcw", [-1, -0.04, -0.02, -0.01, -0.005, 1]),
                    ("breadth", [-0.01, 0.1, 0.25, 0.5, 1.01]), ("tr24", [-1, -0.15, -0.08, -0.03, 0, 0.05, 10])):
        b = pd.cut(d[f], bins)
        t = d.groupby(b, observed=True).ret.agg(["count", lambda x: (x > 0).mean(), "mean", "sum"])
        t.columns = ["n", "win", "avg", "sum"]
        print(f"  {f}:\n" + "\n".join(f"    {str(ix):18s} n {r.n:6.0f} win {r.win*100:5.1f}% avg {r.avg*100:+.3f}% sum {r['sum']*100:+8.1f}%"
                                       for ix, r in t.iterrows()))
    by = d.groupby("k").ret.agg(["count", lambda x: (x > 0).mean(), "mean", "sum"])
    by.index = [S.SYMS[i] for i in by.index]
    by.columns = ["n", "win", "avg", "sum"]
    print("\nper coin:\n" + by.round(4).to_string())
    return d


if __name__ == "__main__":
    a = sys.argv[1:]
    anatomy(int(a[0]), float(a[1]), float(a[2]), int(a[3]), int(a[4]), float(a[5]))
