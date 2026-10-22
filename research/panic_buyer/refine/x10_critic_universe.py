"""x10 critic: the universe agent's hindsight design H4 (19 coins as core + monthly top-50-by-365-day-volume tier at
depth +6 points and half size, 35% crash guard, watch mode, one coin x 100%, depth 14%, +5%, 48h, 24h cooldown), run
with their own code (x10_universe_robust.run logic), then:
  - no new entries 2025-10-10 20:00..23:59 UTC (the flash-crash evening);
  - adversarial in-hour order alone (worder=1, no extra slippage) and +0.3% slippage alone, to split their stress;
  - the live bot's mechanism (resting bids, mode 'bid').
Needs PB_OUT = the universe agent's output folder (read-only use of its cached panels)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_lib as X  # noqa: E402
import x10_universe_push2 as G4  # noqa: E402
import x10_universe_round2 as G2  # noqa: E402

H4 = dict(univ="U19L50T", g=0.35, depth=0.14, tp=0.05, cd=24, hm=48)


def run(U_, d, block=None, watch=True, **extra):
    m, dm = G4.setup4(U_, d["univ"], d["depth"])
    m = np.ascontiguousarray(m * G4.guard_mask(U_, d["g"]))
    if block is not None:
        idx = U_["idx"]
        b = (idx >= pd.Timestamp(block[0])) & (idx < pd.Timestamp(block[1]))
        m = m.copy(); m[b] = 0.0
    P, sl, mm, i19, cols = X.sub(U_, m)
    dm = np.ascontiguousarray(dm[:, cols])
    e, lg = X.urun(P, U_["idx"], sl, smult=mm, N=1, fmax=1.0, depth=dm, tp=d["tp"], hm=d["hm"], cd=d["cd"],
                   watch=watch, **extra)
    syms = [U_["syms"][c] for c in cols]
    return e, lg, i19, syms


if __name__ == "__main__":
    U_ = G2.build2()
    G2.rank365(U_)
    idx = U_["idx"]
    for lab, kw in (("H4 base (their 11.04%)", {}),
                    ("H4, no entries 10 Oct 2025 20:00-24:00", dict(block=("2025-10-10 20:00", "2025-10-11 00:00"))),
                    ("H4, adversarial in-hour order only", dict(worder=1)),
                    ("H4, +0.3% slippage on watch fills only", dict(wslip=0.003)),
                    ("H4, adversarial order +0.3% (their stress)", dict(worder=1, wslip=0.003)),
                    ("H4, bid mode (live mechanism)", dict(watch=False))):
        e, lg, i19, syms = run(U_, H4, **kw)
        r = X.report(e, lg, idx, i19)
        print(f"{lab:46s} {X.fmt(r)} | 2024-26 x{r['24-26_x']:.2f} worst trade {r['24-26_worst_acct']*100:+.1f}%", flush=True)
        if not kw:
            t = lg[(lg.eb >= 0)].copy()
            t["tin"] = idx[t.eb.astype(int).values]
            t["acct"] = t.ret * t.amt / t.eq0
            t["coin"] = [syms[int(k)] for k in t.k]
            sel = (t.tin >= "2025-10-10") & (t.tin < "2025-10-12")
            print("   trades entered 10-11 Oct 2025:", "; ".join(f"{x.coin} {x.tin} @ {x.entry:.4g} {x.ret*100:+.1f}% "
                  f"({x.acct*100:+.1f}% of acct)" for x in t[sel].itertuples()) or "none")
            tt = t[t.tin >= "2024-01-01"].sort_values("acct", ascending=False)
            print("   best 5 trades 2024-26:", "; ".join(f"{x.coin} {x.tin:%Y-%m-%d %H:%M} {x.acct*100:+.1f}%" for x in tt.head(5).itertuples()))
