"""Round 3, task "tphigh": POST-HOC diagnostics, run after r3_tphigh.py's grid had run once and its verdict was fixed
(pick "tp = 0.10 x (ref - entry), no floor" rejected: fails the fills-0.2%-worse check in 2024-26). Nothing here can
change that verdict or make another variant a candidate; it only explains it.

1. Jitter band of the live settings (ROUND3_PREREG noise rule: start date, depth +-0.1 point, fee +-0.005%).
2. Why the pick loses under fills 0.2% worse: exit anatomy, live vs pick, normal vs worse fills.
3. The floored near-misses (family B) under the two stress checks. NOT eligible: they were not the 2021-23 pick.
"""
import numpy as np

import engine as E
import study as S
from r3_tphigh import GRID, trun

E.run = trun
PICK = "tp = 0.10 x (ref - entry), no floor"


def slice_from(P, idx, start):
    m = idx >= start
    return {f: v[m] for f, v in P.items()}, idx[m]


print("1. Jitter band, live settings (each a tiny perturbation of the live bot):")
syms, idx, P = S.market(False)
rows = []
for lab, kw, st in ([("live", {}, None)] + [(f"depth {d:.3f}", {"depth": d}, None) for d in (0.119, 0.121)] +
                    [(f"fee {x:.5f}", {"fee": x}, None) for x in (0.00095, 0.00105)] +
                    [(f"start +{dd}d", {}, dd) for dd in (1, 2, 3, 5, 7, 10, 14)]):
    PP, ii = (P, idx) if st is None else slice_from(P, idx, idx[0] + np.timedelta64(st, "D"))
    e, lg = trun(PP, ii, **kw)
    r = E.score(e, lg, ii)
    rows.append(r)
    print(E.line(f"  {lab}", r))
dm = [r["d_mult"] for r in rows]; tm = [r["t_mult"] for r in rows]
print(f"  band: 2021-23 x{min(dm):.2f}..x{max(dm):.2f}, 2024-26 x{min(tm):.2f}..x{max(tm):.2f}")

print("\n2. Exit anatomy, normal fills vs fills 0.2% worse. Sums are return x order size divided by the equity at the")
print("   start of the period (so +1.00 = one start-of-period equity earned, before compounding):")
R_, O_ = P["ref"].values, P["o"].values
for pen in (0.0, 0.002):
    for lab, spec in (("LIVE", {}), ("pick", GRID[PICK]), ("B f=0.10", GRID["tp = max(1.5%, 0.10 x (ref - entry))"])):
        e, lg = trun(P, idx, pen=pen, **spec)
        xi = idx[lg.xb.astype(int).values]
        ebi, ki = lg.eb.astype(int).values, lg.k.astype(int).values
        mk = O_[ebi, ki] <= R_[ebi, ki] * (1 - E.LIVE["depth"])
        for p, m, e0 in (("21-23", xi < E.DESIGN_END, e.iloc[0]), ("24-26", xi >= E.DESIGN_END, e[e.index < E.DESIGN_END].iloc[-1])):
            g = lg[m]; gm = mk[m]
            tp_, ts = g[g.why == 0], g[g.why == 1]
            w = g.ret * g.amt / e0
            print(f"  pen {pen:.3f} {lab:8s} {p}: trades {len(g)}, TP {len(tp_)} ({w[g.why == 0].sum():+.2f}, avg "
                  f"{tp_.ret.mean() * 100:+.2f}%), time stops {len(ts)} ({w[g.why == 1].sum():+.2f}, avg "
                  f"{ts.ret.mean() * 100:+.1f}%), net {w.sum():+.2f} [limit fills {w[~gm].sum():+.2f}, market buys "
                  f"{int(gm.sum())} {w[gm].sum():+.2f}], avg held {(g.xb - g.eb).mean():.1f}h")

print("\n3. Floored near-misses under the stress checks (post-hoc, NOT eligible):")
for label, dead, pen in (("+LUNA/FTT/SRM", True, 0.0), ("fills 0.2% worse", False, 0.002)):
    b, _, _ = S.evaluate({}, dead=dead, pen=pen)
    print(E.line(f"  live, {label}", b))
    for name in [k for k in GRID if "max(1.5%" in k] + [PICK]:
        c, _, _ = S.evaluate(GRID[name], dead=dead, pen=pen)
        print(E.line(f"  {name[:34]}, {label[:6]}", c) + ("  beats" if E.beats(c, b) else "  fails"))
