"""Lever "cycle": re-run the STRESSED runs of the phase-1 and phase-2 grids with collapse-coin liquidation on.

Why (found after the phase-2 run): the panel cuts LUNA/FTT/SRM at their collapse. A position bought in a cut coin's
last data hour was frozen forever at that price; with 1-2 coins x 50-100% this froze the whole account (phase-2 rule-B2
pick: FTT bought at its Sep 2023 reopen, then +0% every later year). x10_cycle_guard `liquidate` sells such a position
at the last real close instead. This changes only the stress model; the selection rules are unchanged.

Phase-1 settings run through ckernel3 with every phase-2 switch off (proven equal to ckernel via ckernel2), plus
liquidation. A sanity check first confirms that liquidation off reproduces grid_stressed.csv rows exactly.
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_cycle_grid as G1  # noqa: E402
import x10_cycle_grid2 as G2  # noqa: E402
import x10_cycle_lib as L  # noqa: E402

P1 = dict(lcd=0, stop=0.0, skip=0.0, exit="plain", dual=0.0)


def settings(phase, C, W):
    if phase == 1:
        for d in G1.DEPTHS:
            for tp in G1.TPS:
                for hm in G1.HMS:
                    for cd in G1.CDS:
                        for N, f in G1.SIZING:
                            yield dict(C=C, W=W, depth=d, tp=tp, hm=hm, cd=cd, N=N, fmax=f, **P1)
    else:
        for d in G2.DEPTHS:
            for tp in G2.TPS:
                for hm in G2.HMS:
                    for N, f in G2.SIZING:
                        for lcd in G2.LCDS:
                            for stop in G2.STOPS:
                                for skip in G2.SKIPS:
                                    for x in G2.EXITS:
                                        for dual in G2.DUALS:
                                            yield dict(C=C, W=W, depth=d, tp=tp, hm=hm, cd=0, N=N, fmax=f, lcd=lcd,
                                                       stop=stop, skip=skip, exit=x, dual=dual)


def job(args):
    phase, C, W = args
    B = L.load_bars(C, dead=True)
    bph = 60 // C
    cf = L.ffill_close(B); clk = L.Clock(B["t"])
    refs = {W: L.make_ref(B, W * bph), 24: L.make_ref(B, 24 * bph), 168: L.make_ref(B, 168 * bph)}
    rows = []
    for s in settings(phase, C, W):
        eq, lg = G2.run_one(B, refs, cf, bph, s, pen=G2.STRESS_PEN, liquidate=True)
        r = L.summarize(clk, eq, lg)
        r["run_dd"] = float((eq / np.maximum.accumulate(eq) - 1).min())
        r["forced"] = int((lg[:, 3] == 5).sum()) if len(lg) else 0
        r.update(s); r["fmax"] = round(s["fmax"], 4)
        rows.append(r)
    return phase, rows


def sanity():
    """liquidation off through run_one == the stored phase-1 / phase-2 stressed rows (a few settings)."""
    S1 = pd.read_csv(os.path.join(L.OUT, "grid_stressed.csv"))
    S2 = pd.read_csv(os.path.join(L.OUT, "grid2_stressed.csv"))
    ok = True
    B = L.load_bars(15, dead=True); bph = 4; cf = L.ffill_close(B); clk = L.Clock(B["t"])
    refs = {1: L.make_ref(B, 4), 3: L.make_ref(B, 12), 24: L.make_ref(B, 96), 168: L.make_ref(B, 672)}
    for S, rowsel in ((S1, (S1.C == 15) & S1.W.isin([1, 3])), (S2, (S2.C == 15))):
        for _, row in S[rowsel].sample(6, random_state=1).iterrows():
            s = {k: row[k] for k in ("C", "W", "depth", "tp", "hm", "cd", "N", "fmax")}
            s.update({k: (row[k] if k in row else P1[k]) for k in P1})
            s["W"] = int(s["W"]); s["N"] = int(s["N"]); s["hm"] = int(s["hm"]); s["cd"] = int(s["cd"]); s["lcd"] = int(s["lcd"])
            if s["N"] == 3:
                s["fmax"] = 1.0 / 3.0
            eq, lg = G2.run_one(B, refs, cf, bph, s, pen=G2.STRESS_PEN, liquidate=False)
            r = L.summarize(clk, eq, lg)
            same = all(np.isclose(r[c], row[c], rtol=0, atol=1e-12) or (np.isnan(r[c]) and np.isnan(row[c]))
                       for c in ("d_mult", "t_mult", "s_mult", "d_dd", "t_dd"))
            ok &= bool(same)
            print(f"  sanity {s}: {'same' if same else 'DIFFERENT'}")
    return ok


def main(workers=2):
    assert sanity(), "liquidation-off does not reproduce the stored stressed grids"
    tasks = [(p, C, W) for p in (1, 2) for C in (5, 15) for W in (G1.WS if p == 1 else G2.WS)]
    t0 = time.time()
    res = {1: [], 2: []}
    with Pool(workers) as pool:
        for p, rows in pool.imap_unordered(job, tasks):
            res[p] += rows
            print(f"  done phase {p}: {len(res[1])} + {len(res[2])} runs, {time.time() - t0:.0f}s", flush=True)
    k1 = ["C", "W", "depth", "tp", "hm", "cd", "N", "fmax"]
    pd.DataFrame(res[1]).sort_values(k1).reset_index(drop=True).to_csv(os.path.join(L.OUT, "grid_stressed_liq.csv"), index=False)
    pd.DataFrame(res[2]).sort_values(G2.KEYS).reset_index(drop=True).to_csv(os.path.join(L.OUT, "grid2_stressed_liq.csv"), index=False)
    print(f"saved {len(res[1])} + {len(res[2])} runs")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2)
