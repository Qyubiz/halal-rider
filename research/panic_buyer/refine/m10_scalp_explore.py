"""m10 scalp, step 1 (design data only, 2021-23): coarse map of minute dip-scalping. Every design: bids re-placed each
minute at (max close of the last W minutes) x (1 - depth), take-profit limit at entry x (1 + tp) active from the next
minute, time stop hmm minutes, cooldown 0, wicks beyond 0.5% of the body trimmed, 19 coins, fees 0.1%/side,
slippage 0.02% (top-50 volume) / 0.1%. Only 2021-01-01 .. 2023-12-31 is simulated.
Usage: python m10_scalp_explore.py [grid-name]"""
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_scalp_lib as S  # noqa: E402

GRIDS = {
    "pilot": dict(W=[60, 240], depth=[0.02, 0.04], tp=[0.005, 0.01], hmm=[360, 1440], size=[(1, 1.0), (4, 0.25)]),
    "coarse": dict(W=[15, 60, 240, 360], depth=[0.01, 0.02, 0.03, 0.04, 0.06], tp=[0.005, 0.007, 0.01, 0.015],
                   hmm=[240, 1440, 5760], size=[(1, 1.0), (4, 0.25)]),
    # per-signal edge: every coin can hold a 1/19 slice, so coins do not compete for cash
    "edge": dict(W=[5, 15, 60, 240, 720], depth=[0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12], tp=[0.005, 0.01, 0.02],
                 hmm=[1440, 5760], size=[(19, 1 / 19)]),
    # step 2: concentrated sizing on the region with a positive per-trade edge in "edge", crash guard 35% (as 90% mode)
    "conc": dict(W=[15, 60, 240], depth=[0.06, 0.08, 0.10, 0.12], tp=[0.01, 0.015, 0.02], hmm=[1440, 5760],
                 size=[(1, 1.0), (2, 0.5), (4, 0.25)], guard=0.35),
}


def main(name):
    g = GRIDS[name]
    A, hidx, Ph, SLIP = S.load(19)
    i1 = hidx.searchsorted(S.T24)
    guard = g.get("guard", 0.0)
    gref = S.guard_ref(Ph) if guard > 0 else None
    rows = []
    out = os.path.join(S.OUT, f"explore_{name}.csv")
    os.makedirs(S.OUT, exist_ok=True)
    t0 = time.time()
    for W, depth, tp, hmm, (N, fmax) in itertools.product(g["W"], g["depth"], g["tp"], g["hmm"], g["size"]):
        kw = S.scalp(W=W, depth=depth, tp=tp, hmm=hmm, N=N, fmax=fmax, guard=guard)
        e, lg, inv, lo, ntr = S.srun(A, hidx, Ph, SLIP, i0=0, i1=i1, gref=gref, **kw)
        r = S.period_stats(e, lg, lo, hidx[:i1], S.T21, S.T24)
        r.update(W=W, depth=depth, tp=tp, hmm=hmm, N=N, fmax=fmax, guard=guard, inv=float(inv.mean()), ntr=ntr)
        rows.append(r)
        print(f"W{W:4d} d{depth*100:4.1f} tp{tp*100:4.2f} hm{hmm:5d} N{N} {S.line(r, '21-23')}", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{len(rows)} designs in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
