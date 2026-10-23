"""m11 allocator, step 2: the design search, on 2021-23 ONLY (2024-26, since 2025 and the 2018-20 holdout account
runs are not computed here).

SELECTION RULE (written before any allocator result was computed):
  Each design runs as one account on the 2021-26 scenario and is scored on 2021-01-01 .. 2023-12-31 only.
  Qualifies if ALL of:
    (1) normal fills, monthly rebalance on day 1 (weekly designs: Monday): trade win rate >= 93.8% and worst drop
        no deeper than -64.5% (90% mode: 93.8% / -64.4%, published as -64);
    (2) every fill 0.2% worse: the same two thresholds (win rate >= 93.8%, drop >= -64.5%);
    (3) robustness, normal fills: the same two thresholds with the rebalance on day 15 (weekly: Thursday), and with
        the account starting 2 weeks later (2021-01-15; periods measured from that day).
  Pick = the qualifier with the highest score = the LOWEST 2021-23 %/month of its three normal-fill runs (day 1,
  day 15, late start). Ties: fewer sleeves in its universe, then the earlier design in the grid.
  Fallback written in advance: if nothing qualifies, (2) becomes "at least 90% mode with worse fills"
  (win rate >= 93.4%, drop >= -65.9%).
  Then ONE look (m11_allocator_final.py) at 2024-26, since 2025, the 2018-20 holdout, worse fills and the stressed
  run. No re-picking after that look.

Grid (288 designs): universes A (all 8 sleeves), B (90% mode, velocity, regime, meanrev, scalp), C (90% mode,
velocity, regime); trailing window L = 1, 3, 6, 12 months; rules: top-k trailing return (k 1-3, with/without the
"positive return only" filter), top-k trailing win rate (k 1-2, positive return, >= 10 trades), inverse volatility
and inverse worst drop (with/without the positive filter), fractional Kelly (0.25, 0.5 on daily returns), win-rate
gate (best trailing return among sleeves with trailing win rate >= 93% / 95%), anchor (90% mode 50% / 70% + the
best trailing sleeve), loss-streak switch (sleeves with >= 2 losers in their last 5 trades excluded) on three rules;
plus weekly rebalancing for three rules. Weights < 10% are dropped (the rest keeps its share; unused share = cash).
Output: $PB_OUT/search.csv, search.log
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m11_allocator_lib as A  # noqa: E402

WR_MIN, DD_MIN = 93.8, -64.5
FALLBACK_W = (93.4, -65.9)
LATE = pd.Timestamp("2021-01-15")


def grid():
    D = []
    for u in ("A", "B", "C"):
        for L in (1, 3, 6, 12):
            for k in (1, 2, 3):
                for absf in (0, 1):
                    D.append(dict(univ=u, L=L, rule="topk", k=k, absf=absf))
            for k in (1, 2):
                D.append(dict(univ=u, L=L, rule="topwr", k=k))
            for r in ("ivol", "idd"):
                for absf in (0, 1):
                    D.append(dict(univ=u, L=L, rule=r, absf=absf))
            for f in (0.25, 0.5):
                D.append(dict(univ=u, L=L, rule="kelly", f=f))
            for g in (0.93, 0.95):
                D.append(dict(univ=u, L=L, rule="wrgate", g=g))
            for b in (0.5, 0.7):
                D.append(dict(univ=u, L=L, rule="anchor", b=b))
            D.append(dict(univ=u, L=L, rule="topk", k=1, absf=1, streak=1))
            D.append(dict(univ=u, L=L, rule="topk", k=2, absf=1, streak=1))
            D.append(dict(univ=u, L=L, rule="ivol", absf=1, streak=1))
    for u in ("A", "B", "C"):
        for L in (1, 3, 6, 12):
            D.append(dict(univ=u, L=L, rule="topk", k=1, absf=1, freq="W"))
            D.append(dict(univ=u, L=L, rule="topk", k=2, absf=1, freq="W"))
            D.append(dict(univ=u, L=L, rule="ivol", absf=1, freq="W"))
    return D


def evaluate(d):
    a, b = A.PERIODS["2021-23"]
    alt_day = 15 if d.get("freq", "M") == "M" else 3
    r = {}
    for tag, scen, kw, start in (("base", "m", {}, a), ("worse", "mw", {}, a), ("day", "m", dict(day=alt_day), a),
                                 ("late", "m", dict(start=LATE), LATE)):
        e, tk, LAM, Wt, taus = A.run(scen, d, **kw)
        s = A.stats(e, tk, start, b)
        for k, v in s.items():
            r[f"{tag}_{k}"] = v
        if tag == "base":
            r["avg_sleeves"] = float((Wt[(taus >= a) & (taus < b)] > 0).sum(axis=1).mean())
    return r


def main():
    out = A.OUT
    log = open(os.path.join(out, "search.log"), "w")

    def p(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    # reproduction: 100% 90% mode through the allocator kernel == the standalone 90% mode
    ref = dict(static=[1.0] + [0.0] * 7)
    p("100% 90% mode through the one-account allocator kernel (must equal the published 90% mode):")
    for scen, lab, pers in (("h", "normal  ", ("2018-20",)), ("m", "normal  ", ("2021-23", "2024-26", "since 2025")),
                            ("mw", "worse   ", ("2021-23", "2024-26", "since 2025")), ("ms", "stressed", ("2024-26",))):
        e, tk, *_ = A.run(scen, ref)
        for per in pers:
            p(f"  {lab} {per:10s} " + A.fmt(A.stats(e, tk, *A.PERIODS[per])))
    D = grid()
    p(f"\n{len(D)} designs, 2021-23 only")
    rows = []
    t0 = time.time()
    for i, d in enumerate(D):
        r = evaluate(d)
        r.update(i=i, design=A.dname(d), nuniv=len(A.UNIV[d["univ"]]), spec=repr(d))
        rows.append(r)
        if i % 24 == 0:
            p(f"  {i:3d} {time.time() - t0:5.0f}s {r['design']:34s} base WR {r['base_wr']:5.1f} {r['base_mo']:+6.2f}%/mo "
              f"DD {r['base_dd']:6.1f}")
    df = pd.DataFrame(rows)
    df["score"] = df[["base_mo", "day_mo", "late_mo"]].min(axis=1)
    norm_ok = np.ones(len(df), bool)
    for tag in ("base", "day", "late"):
        norm_ok &= (df[f"{tag}_wr"] >= WR_MIN) & (df[f"{tag}_dd"] >= DD_MIN)
    q = norm_ok & (df.worse_wr >= WR_MIN) & (df.worse_dd >= DD_MIN)
    used = "main rule"
    if not q.any():
        q = norm_ok & (df.worse_wr >= FALLBACK_W[0]) & (df.worse_dd >= FALLBACK_W[1])
        used = "FALLBACK rule (worse fills >= 90% mode with worse fills)"
    df["qualifies"] = q
    df.to_csv(os.path.join(out, "search.csv"), index=False)
    p(f"\nqualifying normal-fill thresholds (all 3 variants): {int(norm_ok.sum())}; qualifying overall ({used}): "
      f"{int(q.sum())}")
    cols = ["design", "score", "base_wr", "base_mo", "base_dd", "worse_wr", "worse_mo", "worse_dd", "day_mo",
            "late_mo", "base_tpm", "avg_sleeves"]
    pd.set_option("display.width", 250)
    if q.any():
        top = df[q].sort_values(["score", "nuniv", "i"], ascending=[False, True, True])
        p("\ntop 15 qualifiers (2021-23):\n" + top[cols].head(15).round(2).to_string(index=False))
        pick = top.iloc[0]
        p(f"\nPICK: {pick.design}\n  spec {pick.spec}")
        with open(os.path.join(out, "pick.txt"), "w") as f:
            f.write(pick.spec + "\n")
    p("\nbest 10 by 2021-23 %/month regardless of the rule (for context):\n" +
      df.sort_values("base_mo", ascending=False)[cols].head(10).round(2).to_string(index=False))
    p("\nhighest 2021-23 win rate (base) designs:\n" +
      df.sort_values("base_wr", ascending=False)[cols].head(8).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
