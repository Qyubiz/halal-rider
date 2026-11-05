"""m11 allocator, step 3: the ONE look. The pick of m11_allocator_search.py ($PB_OUT/pick.txt, chosen on 2021-23
only) and 90% mode through the same one-account engine, on 2021-23, 2024-26, since 2025 and the 2018-20 holdout,
normal fills, every fill 0.2% worse, and stressed (real bot timing + fills 0.2% worse + LUNA/FTT/SRM; its
allocator reads the stressed 2021-26 history joined to the worse-fill holdout). Also shown (information only, not
used to choose): the pick with the rebalance on day 15 and started 2 weeks later, which sleeve held the money each
year, and the share of trades per sleeve.
Output: $PB_OUT/final.log
"""
import ast
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_allocator_lib as A  # noqa: E402

LATE = pd.Timestamp("2021-01-15")


def main():
    log = open(os.path.join(A.OUT, "final.log"), "w")

    def p(s=""):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    pick = ast.literal_eval(open(os.path.join(A.OUT, "pick.txt")).read().strip())
    designs = {"90% mode": dict(static=[1.0] + [0.0] * 7), f"PICK {A.dname(pick)}": pick}
    rows = (("normal", "m", "2021-23"), ("normal", "m", "2024-26"), ("normal", "m", "since 2025"),
            ("normal", "h", "2018-20"),
            ("worse", "mw", "2021-23"), ("worse", "mw", "2024-26"), ("worse", "mw", "since 2025"),
            ("worse", "hw", "2018-20"),
            ("stressed", "ms", "2021-23"), ("stressed", "ms", "2024-26"), ("stressed", "ms", "since 2025"))
    for name, d in designs.items():
        p(f"\n{name}")
        for lab, scen, per in rows:
            e, tk, *_ = A.run(scen, d)
            p(f"  {lab:8s} {per:10s} " + A.fmt(A.stats(e, tk, *A.PERIODS[per])))
    p("\nPICK robustness (normal fills, information only):")
    for lab, kw in (("rebalance day 15", dict(day=15)), ("start 2021-01-15", dict(start=LATE))):
        e, tk, *_ = A.run("m", pick, **kw)
        for per in ("2021-23", "2024-26", "since 2025"):
            a, b = A.PERIODS[per]
            p(f"  {lab:17s} {per:10s} " + A.fmt(A.stats(e, tk, max(a, kw.get('start', a)), b)))
    for scen, lab in (("h", "2018-20 holdout"), ("m", "2021-26")):
        e, tk, LAM, Wt, taus = A.run(scen, pick)
        W = pd.DataFrame(Wt, taus, A.NAMES)
        held = W.idxmax(axis=1).where(W.max(axis=1) > 0, "cash")
        p(f"\nmonths per sleeve by year ({lab}, normal fills):")
        p(pd.crosstab(held.index.year, held).to_string())
        sh = tk.groupby("s").agg(n=("ret", "size"), wr=("ret", lambda x: 100 * (x > 0).mean()))
        sh.index = [A.NAMES[i] for i in sh.index]
        p("trades taken per sleeve (whole run): " + "; ".join(f"{i} {int(r.n)} ({r.wr:.1f}% won)" for i, r in sh.iterrows()))
        p("last 6 monthly choices: " + ", ".join(f"{t:%Y-%m} {h}" for t, h in held.tail(6).items()))


if __name__ == "__main__":
    main()
