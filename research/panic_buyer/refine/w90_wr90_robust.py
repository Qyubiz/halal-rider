"""wr90 robustness of the candidate finalists (a measurement, nothing is re-picked here):
  - neighbours: depth +-0.25/+-0.5 pt, tp +-0.25 pt, hm +-24h, cd +-1..3h, guard 30/40% (one at a time),
    reported as min / median of min(%/month) and the share that keep win rate >= 90% in 2021-23 and 2024-26;
  - fresh starts: all cash at 2024-01-01 + d and 2025-01-01 + d (d = 0..29 days): %/month and drop band;
  - per-year returns, worst calendar month, longest hold, trades/month, worst single trades as % of the account;
  - stressed version, also with buy-side-only 0.2% penalty (the REFINE.md convention) for comparison.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

CANDS = {
    "F1s rule winner + 30% stop (hindsight grid F)": dict(N=1, fmax=1.0, depth=0.13, tp=0.0125, hm=96, cd=3, guard=0.35,
                                                         hs=0.30),
    "M mid drop (hindsight grid F)": dict(N=2, fmax=0.5, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.25),
    "F1 rule winner (hindsight grid D)": dict(N=1, fmax=1.0, depth=0.13, tp=0.0125, hm=96, cd=3, guard=0.35),
    "F2 +2.5% target (hindsight grid D)": dict(N=1, fmax=1.0, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.35),
    "F3 low drop (pre-declared grid A)": dict(N=4, fmax=0.5, depth=0.16, tp=0.02, hm=96, cd=24, guard=0.35),
    "LIVE": dict(N=8, fmax=0.2, depth=0.12, tp=0.015, hm=48, cd=24, guard=0.0),
}


def neighbours(kw):
    out = []
    for dd in (-0.005, -0.0025, 0.0025, 0.005):
        out.append(dict(kw, depth=round(kw["depth"] + dd, 4)))
    for dt in (-0.0025, 0.0025):
        out.append(dict(kw, tp=round(kw["tp"] + dt, 4)))
    for dh in (-24, 24):
        out.append(dict(kw, hm=kw["hm"] + dh))
    for c in sorted({max(0, kw["cd"] - 3), max(0, kw["cd"] - 1), kw["cd"] + 1, kw["cd"] + 3} - {kw["cd"]}):
        out.append(dict(kw, cd=c))
    if kw["guard"] > 0:
        for g in (kw["guard"] - 0.05, kw["guard"] + 0.05):
            out.append(dict(kw, guard=round(g, 2)))
    return out


def fresh(kw, start, days=30):
    """Fresh all-cash runs from start + d; trigger ref and guard mask from the full panel (keeps their warm-up)."""
    syms, idx, P = W.market(False)
    gm = W.guard_mask(P, kw["guard"]) if kw["guard"] > 0 else np.ones(P["o"].shape)
    k = {x: v for x, v in kw.items() if x != "guard"}
    res = []
    for d in range(days):
        i0 = int(np.searchsorted(idx, pd.Timestamp(start) + pd.Timedelta(days=d)))
        Ps = {f: v.iloc[i0:] for f, v in P.items()}
        e, lg = W.wrun(Ps, idx[i0:], smult=np.ascontiguousarray(gm[i0:]), **k)
        m, mo, dd = W.period(e, e.index[0], None)
        res.append((mo, dd))
    a = np.array(res)
    return a[:, 0].min(), np.median(a[:, 0]), a[:, 0].max(), a[:, 1].min()


def details(kw):
    syms, idx, P = W.market(False)
    e, lg = W.wrun(P, idx, **kw)
    y = E.years(e)
    m = e.resample("ME").last()
    mr = pd.concat([pd.Series([1.0], [e.index[0]]), m]).pct_change().dropna()
    acct = lg.ret * lg.amt / lg.eq0
    lg = lg.assign(acct=acct, coin=[syms[int(k)][:-4] for k in lg.k], t=idx[lg.eb.astype(int).values])
    worst = lg.nsmallest(4, "acct")
    yr = {}
    for yy in range(2021, 2027):
        x = lg[lg.t.dt.year == yy]
        yr[yy] = (float((x.ret > 0).mean()) if len(x) else np.nan, len(x))
    return y, mr, worst, lg, yr


def main(only=None):
    for name, kw in CANDS.items():
        if only and name.split()[0] not in only:
            continue
        r, e, lg = W.evaluate(kw)
        rs, _, _ = W.evaluate(kw, stress=True)
        syms, idx, P = W.market(True)
        eb, lb = W.wrun(P, idx, **dict(kw, mtp_now=True, cd=kw["cd"] + 1, pen=0.002, liquidate=True))
        rb = W.summarize(eb, lb, idx)
        print("=" * 120)
        print(W.fmt(name + "  NORMAL", r))
        print(W.fmt("   STRESSED (bot timing, buys+sells 0.2% worse, +LUNA/FTT/SRM)", rs))
        print(W.fmt("   stressed, buy-side-only penalty (REFINE.md convention)", rb))
        y, mr, worst, lg2, yr = details(kw)
        print("   per year: " + " ".join(f"{k}: {v*100:+.0f}% (WR {yr[k][0]*100:.0f}%, {yr[k][1]} tr)" for k, v in y.items()))
        print(f"   worst month {mr.min()*100:+.1f}% ({mr.idxmin():%Y-%m}), months up {(mr > 0).mean()*100:.0f}%, "
              f"longest hold {int((lg2.xb - lg2.eb).max())}h, trades/month 2024-26 {r['t_n'] / 33.7:.1f}")
        print("   worst trades (% of account): " + "; ".join(f"{w.coin} {w.t:%Y-%m-%d} {w.acct*100:+.1f}% (trade {w.ret*100:+.1f}%)"
                                                         for w in worst.itertuples()))
        N = []
        for nk in neighbours(kw):
            q = W.evaluate(nk)[0]
            N.append(dict(q, **{f"p_{a}": b for a, b in nk.items()}))
        N = pd.DataFrame(N)
        ok = (N.d_wr >= 0.9) & (N.t_wr >= 0.9)
        print(f"   neighbours ({len(N)}): min(%/mo) min {N.minmo.min()*100:+.2f} median {N.minmo.median()*100:+.2f} "
              f"max {N.minmo.max()*100:+.2f} | 2024-26 %/mo {N.t_mo.min()*100:+.2f}..{N.t_mo.max()*100:+.2f} | since-25 "
              f"{N.s_mo.min()*100:+.2f}..{N.s_mo.max()*100:+.2f} | WR>=90% both: {ok.mean()*100:.0f}% | worst drop "
              f"{N.mindd.min()*100:.0f}..{N.mindd.max()*100:.0f}%")
        lo = N.loc[N.minmo.idxmin()]
        print(f"      weakest neighbour: " + ", ".join(f"{a[2:]}={lo[a]}" for a in N.columns if a.startswith("p_")) +
              f" -> {lo.d_mo*100:+.1f}/{lo.t_mo*100:+.1f}/{lo.s_mo*100:+.1f}, WR {lo.d_wr*100:.1f}/{lo.t_wr*100:.1f}")
        for st in ("2024-01-01", "2025-01-01"):
            a, b, c, d = fresh(kw, st)
            print(f"   fresh starts {st}+0..29d: %/mo {a*100:+.2f} / median {b*100:+.2f} / {c*100:+.2f}, worst drop {d*100:.0f}%")
        N.to_csv(os.path.join(W.OUT, f"robust_neighbours_{name.split()[0]}.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
