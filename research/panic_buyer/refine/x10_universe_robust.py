"""x10 lever "universe": robustness of the hindsight best designs from round 4 (x10_universe_push2.py).

Checks for each design (all hindsight):
  1. plateau: depth +-1 point in half steps, tp +-0.5 points, cooldown 18/24/30h (or the design's +-6h), guard 30/35/40%
     -> distribution of 2024-26 %/month;
  2. fresh starts on 2024-01-01 + 0..30 days;
  3. costs: every fill 0.1% / 0.2% worse; BNB fees (0.075%/side); limit buys need 0.3% through instead of 0.1%;
  4. the real bot's timing: take-profit placed at once after a market buy (mtp_now) and cooldown one hour longer.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_lib as X  # noqa: E402
import x10_universe_push2 as G4  # noqa: E402
import x10_universe_round2 as G2  # noqa: E402

DESIGNS = {
    "H4 19 core + top-50 tier, guard 35%": dict(univ="U19L50T", g=0.35, depth=0.14, tp=0.05, cd=24, hm=48),
    "H4b top-40 tier only (no hand-picked list), guard 35%": dict(univ="L40T", g=0.35, depth=0.15, tp=0.05, cd=6,
                                                                 hm=96),
}


def run(U_, d, start=None, **kw):
    """watch mode, N1 x 100%; kw may override depth/tp/cd/hm/g and pass pen/fee/eps/mtp_now."""
    p = {**d, **{k: v for k, v in kw.items() if k in d}}
    extra = {k: v for k, v in kw.items() if k not in d}
    univ, g = p["univ"], p["g"]
    m, dm = G4.setup4(U_, univ, p["depth"])
    if g > 0:
        m = np.ascontiguousarray(m * G4.guard_mask(U_, g))
    P, sl, mm, i19, cols = X.sub(U_, m)
    idx = U_["idx"]
    dm = np.ascontiguousarray(dm[:, cols])
    if start is not None:
        i0 = int(np.searchsorted(idx, pd.Timestamp(start)))
        P = {f: v.iloc[i0:] for f, v in P.items()}
        sl, mm, dm, idx = sl[i0:], mm[i0:], dm[i0:], idx[i0:]
    e, lg = X.urun(P, idx, sl, smult=mm, N=1, fmax=1.0, depth=dm, tp=p["tp"], hm=p["hm"], cd=p["cd"],
                   watch=True, **extra)
    return e, lg, i19


if __name__ == "__main__":
    U_ = G2.build2()
    G2.rank365(U_)
    for name, d in DESIGNS.items():
        print(f"\n######## {name}: {d}")
        e, lg, i19 = run(U_, d)
        r = X.report(e, lg, U_["idx"], i19)
        print("  base: " + X.fmt(r))
        rows = []
        for dd_, tp_, cd_, g_ in itertools.product(np.round(d["depth"] + np.array([-0.01, -0.005, 0, 0.005, 0.01]), 4),
                                                   np.round(d["tp"] + np.array([-0.005, 0, 0.005]), 4),
                                                   [max(0, d["cd"] - 6), d["cd"], d["cd"] + 6], [0.30, 0.35, 0.40]):
            e2, lg2, _ = run(U_, d, depth=float(dd_), tp=float(tp_), cd=int(cd_), g=float(g_))
            mo, dd, _ = X.per_month(e2, "2024-01-01")
            mo5, _, _ = X.per_month(e2, "2025-01-01")
            mo3, dd3, _ = X.per_month(e2, "2021-01-01", "2024-01-01")
            rows.append(dict(depth=dd_, tp=tp_, cd=cd_, g=g_, mo2426=mo, dd2426=dd, mo2526=mo5, mo2123=mo3, dd2123=dd3))
        R = pd.DataFrame(rows)
        R.to_csv(os.path.join(X.OUT, f"plateau_{d['univ']}.csv"), index=False)
        q = R.mo2426 * 100
        print(f"  1. plateau ({len(R)} neighbours): 2024-26 %/month median {q.median():+.2f}, min {q.min():+.2f}, "
              f"max {q.max():+.2f}, share >= 10%: {(q >= 10).mean() * 100:.0f}%, share >= 8%: {(q >= 8).mean() * 100:.0f}%; "
              f"since 2025 median {R.mo2526.median() * 100:+.2f}; 2021-23 median {R.mo2123.median() * 100:+.2f} "
              f"(drop median {R.dd2123.median() * 100:.0f}%); 2024-26 drop median {R.dd2426.median() * 100:.0f}%")
        for k in ("depth", "tp", "cd", "g"):
            print(f"     by {k}: " + ", ".join(f"{v}: {s * 100:+.2f}" for v, s in R.groupby(k).mo2426.median().items()))
        out = []
        for off in (0, 3, 7, 10, 14, 21, 30):
            st = pd.Timestamp("2024-01-01") + pd.Timedelta(days=off)
            e2, _, _ = run(U_, d, start=st)
            out.append(f"+{off}d {X.per_month(e2, st)[0] * 100:+.2f}%")
        print("  2. fresh starts (2024-01-01 + n days, to 2026-10-22): " + ", ".join(out))
        for lab, kw in (("fills 0.1% worse", dict(pen=0.001)), ("fills 0.2% worse", dict(pen=0.002)),
                        ("BNB fees 0.075%", dict(fee=0.00075)), ("limit buys+take-profits 0.3% through", dict(eps=0.003)),
                        ("real-bot timing (mtp_now, cd+1)", dict(mtp_now=True, cd=d["cd"] + 1))):
            e2, lg2, _ = run(U_, d, **kw)
            print(f"  3/4. {lab:32s}: " + X.fmt(X.report(e2, lg2, U_["idx"], i19)))
