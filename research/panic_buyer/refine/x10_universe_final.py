"""x10 lever "universe": final reports for the chosen designs (honest picks and hindsight bests).

For each design: the continuous 2021-26 run (periods 2024-26, since 2025, 2021-23), FRESH starts on 2024-01-01 and
2025-01-01 (a run that was wiped out and stuck in 2022 would otherwise show 0%/month later), the stress run (every
fill 0.2% worse + FTT added + LUNA/FTT/SRM always eligible; continuous and fresh 2024 start), the adversarial
in-hour order for watch mode (+0.3% on watch fills), monthly returns, worst trade as % of the account, and which
coins made the money."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_grid as G  # noqa: E402
import x10_universe_lib as X  # noqa: E402
import x10_universe_push2 as G4  # noqa: E402
import x10_universe_round2 as G2  # noqa: E402
from lab import month_returns  # noqa: E402

FORCE = ("LUNAUSDT", "FTTUSDT", "SRMUSDT")

DESIGNS = {
    # honest: pre-registered rule on 2021-23, round 1 (universe lever pick)
    "A  honest pick r1": dict(univ="T30Q", mode="watch", N=2, fmax=0.5, depth=0.12, tp=0.015, cd=24, hm=48),
    # semi-honest: same rule, round 2 (designed after seeing part of round 1)
    "A2 pick r2": dict(univ="L30T", mode="watch", N=1, fmax=1.0, depth=0.12, tp=0.015, cd=24, hm=48),
    # hindsight bests
    "H3 hindsight r3": dict(univ="L50T", mode="watch", N=1, fmax=1.0, depth=0.15, tp=0.05, cd=12, hm=48),
}


def one(U_, d, force=(), pen=0.0, wslip=0.0, worder=0, start=None):
    univ = d["univ"]
    g = 0.0
    if "+g" in univ:
        univ, gs = univ.split("+g")
        g = int(gs) / 100
    if univ in G.UNIVS and univ != "U19" or univ.endswith("Q"):
        sc = X.standalone_scores(U_, d["depth"], d["tp"], d["hm"], d["cd"]) if univ.endswith("Q") else None
        m = G.umask(U_, univ, sc, force)
        if g:
            m = m * G4.guard_mask(U_, g)
        P, sl, mm, i19, cols = X.sub(U_, np.ascontiguousarray(m))
        idx = U_["idx"]
        if start is not None:
            i0 = int(np.searchsorted(idx, pd.Timestamp(start)))
            P = {f: v.iloc[i0:] for f, v in P.items()}
            sl, mm, idx = sl[i0:], mm[i0:], idx[i0:]
        e, lg = X.urun(P, idx, sl, smult=mm, N=d["N"], fmax=d["fmax"], depth=d["depth"], tp=d["tp"], hm=d["hm"],
                       cd=d["cd"], pen=pen, watch=d["mode"] == "watch", wslip=wslip, worder=worder)
    else:
        e, lg, i19, cols = G4.run_one4(U_, univ, d["mode"], d["N"], d["fmax"], d["depth"], d["tp"], d["cd"],
                                       force=force, pen=pen, wslip=wslip, worder=worder, hm=d["hm"], guard=g,
                                       start=start)
    syms = [U_["syms"][c] for c in cols]
    return e, lg, i19, syms


def fresh(U_, d, start, **kw):
    e, lg, i19, _ = one(U_, d, start=start, **kw)
    mo, dd, mult = X.per_month(e, start)
    ts = X.trade_stats(lg, e.index, i19, start)
    return mo, dd, mult, ts


def run_all(U_, US):
    out = {}
    for name, d in DESIGNS.items():
        print(f"\n=== {name}: {d}", flush=True)
        e, lg, i19, syms = one(U_, d)
        rep = X.report(e, lg, U_["idx"], i19)
        for p, desc in (("24-26", "(1) 2024-01..2026-10"), ("25-26", "(2) since 2025-01-01"), ("21-23", "(3) 2021-23")):
            print(f"  {desc:22s} continuous: {rep[p + '_mo'] * 100:+6.2f}%/month  worst drop {rep[p + '_dd'] * 100:5.1f}%  "
                  f"x{rep[p + '_x']:.3g}  trades {rep.get(p + '_n', 0)}  worst trade {rep.get(p + '_worst_acct', np.nan) * 100:+.1f}% "
                  f"of account  P&L outside the 19: {rep.get(p + '_pnl_out19', 0) * 100:+.0f} of "
                  f"{rep.get(p + '_pnl_sum', 0) * 100:+.0f} points")
        for st in ("2024-01-01", "2025-01-01"):
            mo, dd, mult, ts = fresh(U_, d, st)
            print(f"  fresh start {st}: {mo * 100:+6.2f}%/month  worst drop {dd * 100:5.1f}%  x{mult:.3g}  worst trade "
                  f"{ts.get('worst_acct', np.nan) * 100:+.1f}% of account  P&L outside the 19: "
                  f"{ts.get('pnl_out19', 0) * 100:+.0f} of {ts.get('pnl_sum', 0) * 100:+.0f} points")
        print("  years: " + " ".join(f"{y} {rep[f'y{y}'] * 100:+.0f}%" for y in range(2021, 2027)))
        mr = month_returns(e[e.index >= "2024-01-01"])
        print(f"  2024-26 months: median {mr.median() * 100:+.1f}%, months >= +10%: {(mr >= 0.10).sum()} of {len(mr)}, "
              f"losing months: {(mr < 0).sum()}, worst month {mr.min() * 100:+.1f}%")
        t = lg[(lg.xb >= 0) | (lg.xb < 0)].copy()
        t["sym"] = [syms[int(k)] for k in t.k]
        t["acct"] = t.ret * t.amt / t.eq0
        t["t"] = U_["idx"][t.eb.astype(int).values]
        tt = t[t.t >= "2024-01-01"]
        top = tt.groupby("sym").acct.agg(["count", "sum"]).sort_values("sum", ascending=False)
        print("  2024-26 account P&L by coin (points): best " +
              ", ".join(f"{s} {v * 100:+.0f} ({int(c)})" for s, (c, v) in top.head(6).iterrows()) +
              " | worst " + ", ".join(f"{s} {v * 100:+.0f} ({int(c)})" for s, (c, v) in top.tail(4).iterrows()))
        print("  worst 2024-26 trades: " + ", ".join(f"{r.t:%Y-%m-%d} {r.sym} {r.acct * 100:+.1f}%"
                                                    for r in tt.sort_values("acct").head(4).itertuples()))
        if d["mode"] == "watch":
            ea, la, ia, _ = one(U_, d, wslip=0.003, worder=1)
            ra = X.report(ea, la, U_["idx"], ia)
            mo, dd, mult, _ = fresh(U_, d, "2024-01-01", wslip=0.003, worder=1)
            print(f"  adversarial in-hour order +0.3% on watch fills: {X.fmt(ra)}; fresh 2024 start {mo * 100:+.2f}%/month "
                  f"DD {dd * 100:.0f}%")
        es, ls, is_, _ = one(US, d, force=FORCE, pen=0.002)
        rs = X.report(es, ls, US["idx"], is_)
        mo, dd, mult, ts = fresh(US, d, "2024-01-01", force=FORCE, pen=0.002)
        print(f"  (4) fills 0.2% worse + FTT + LUNA/FTT/SRM always eligible: {X.fmt(rs)}; worst trade 21-23 "
              f"{rs.get('21-23_worst_acct', np.nan) * 100:+.1f}%, 24-26 {rs.get('24-26_worst_acct', np.nan) * 100:+.1f}% of "
              f"account; fresh 2024 start {mo * 100:+.2f}%/month DD {dd * 100:.0f}%")
        out[name] = dict(rep=rep, stress=rs)
    return out


if __name__ == "__main__":
    extra = sys.argv[1:]
    for a in extra:                       # extra designs as univ,mode,N,fmax,depth,tp,cd,hm[,label]
        p = a.split(",")
        DESIGNS[p[8] if len(p) > 8 else a] = dict(univ=p[0], mode=p[1], N=int(p[2]), fmax=float(p[3]),
                                                  depth=float(p[4]), tp=float(p[5]), cd=int(p[6]), hm=int(p[7]))
    U_ = G2.build2()
    G2.rank365(U_)
    US = G2.build2(stress=True)
    G2.rank365(US)
    run_all(U_, US)
