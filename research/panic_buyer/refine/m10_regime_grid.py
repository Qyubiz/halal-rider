"""m10 regime grid: entry-blocking regime filters on top of 90% mode, searched and CHOSEN ON 2021-23 ONLY.

Stage 1 (S1): ~90 single rules from the families below (block = no bid on that coin/hour).
Stage 2 (S2): OR-pairs of the 24 best singles (ranking R below) = 276.
Stage 3 (S3): the 12 best designs of S1+S2 (ranking R) + each of the 24 best singles = up to 288 triples.
Stage 4 (S4): the 15 best designs so far with "deeper bid instead of skip" (blocked hours bid at 18/22/26% under the
              24h high instead of 14%) = 45.
Ranking R (2021-23 only): designs with 2021-23 %/month >= 70% of 90% mode's; among them the highest 2021-23 win
rate, ties -> higher 2021-23 %/month.

Selection (fixed before any run, 2021-23 columns only):
  PICK  = highest 2021-23 win rate among designs with 2021-23 %/month >= 90% of 90% mode's AND 2021-23 worst drop no
          more than 1 point deeper than 90% mode's; ties -> higher 2021-23 %/month.
  BANDk = for k in 95, 96, 97, 98, 99: highest 2021-23 %/month among designs with 2021-23 win rate >= k% and >= 100
          trades in 2021-23.
2024-26 / since-2025 numbers are written to the CSV but never printed or used here (m10_regime_final.py reports them).

Families (block if ...):  breadth (share of the 19 coins >= x under their 24h high, or down >= 10% in 24h, or under
their 7/30-day average), BTC drop speed (1/4/24/72h return, 7/30-day drawdown), realized volatility regime (24h/72h
vol vs 30 days), volume spike (24h quote volume vs its 30-day median: BTC, all coins, the coin), consecutive red UTC
days (BTC, median coin), time since the last crash hour, falling knife (coin's 3/7/30-day trend, drawdown, distance
to its 30-day average, last-hour return), bear trend (BTC vs its 50/200-day average, median coin 7-day return).

Usage: python m10_regime_grid.py   -> $PB_OUT/grid.csv, grid.log
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_regime_lib as L  # noqa: E402

SINGLES = {}


def add(name, cond):
    SINGLES[name] = [cond]


for x in (0.25, 0.4, 0.6):
    add(f"brd_dip10>={x}", [("brd_dip10", ">=", x)])
for x in (0.15, 0.3):
    add(f"brd_dip14>={x}", [("brd_dip14", ">=", x)])
for x in (0.5, 0.75):
    add(f"brd_dip8>={x}", [("brd_dip8", ">=", x)])
for x in (0.25, 0.5):
    add(f"brd_r24_10>={x}", [("brd_r24_10", ">=", x)])
for x in (0.8, 0.9, 0.95):
    add(f"brd_sma7>={x}", [("brd_sma7", ">=", x)])
for x in (0.8, 0.9):
    add(f"brd_sma30>={x}", [("brd_sma30", ">=", x)])
for f, xs in (("btc_r1", (-0.01, -0.02)), ("btc_r4", (-0.02, -0.03, -0.05)), ("btc_r24", (-0.04, -0.06, -0.08)),
              ("btc_r72", (-0.08, -0.12)), ("btc_dd7", (-0.10, -0.15)), ("btc_dd30", (-0.15, -0.25)),
              ("mkt_r24", (-0.06, -0.09)), ("mkt_r7d", (-0.10, -0.15)), ("btc_sma200", (0.0, -0.10, -0.20)),
              ("btc_sma50", (-0.05, -0.10)), ("c_pre3d", (-0.05, -0.10)), ("c_pre7d", (-0.10, -0.20)),
              ("c_r7d", (-0.20, -0.30)), ("c_r30d", (-0.30, -0.45)), ("c_dd30", (-0.25, -0.35)),
              ("c_sma30", (-0.15, -0.25))):
    for x in xs:
        add(f"{f}<={x}", [(f, "<=", x)])
for f, xs in (("mkt_rv", (1.2, 1.4, 1.7)), ("btc_rv", (1.3, 1.6, 2.0)), ("btc_rv72", (1.3, 1.6)),
              ("mkt_qv", (1.5, 2.0, 2.5)), ("btc_qv", (1.5, 2.0, 2.5)), ("c_qv", (2.0, 3.0)), ("c_rv", (1.5, 2.0)),
              ("btc_red", (2, 3, 4)), ("mkt_red", (2, 3, 4)), ("c_r1", (0.0, -0.005))):
    for x in xs:
        add(f"{f}>={x}", [(f, ">=", x)])
for f, xs in (("since_b7", (24, 72, 168)), ("since_b10", (24, 72, 168)), ("since_brd30", (6, 24, 72)),
              ("since_brd50", (6, 24, 72))):
    for x in xs:
        add(f"{f}<={x}", [(f, "<=", x)])
# AND singles: coin knife inside a weak market (round 2 "knife"), cascade inside a weak market
for pre in (-0.03, -0.06):
    for br in (0.8, 0.9):
        add(f"knife pre3d<={pre}&brd_sma7>={br}", [("c_pre3d", "<=", pre), ("brd_sma7", ">=", br)])
add("c_r1>=0&brd_sma7>=0.9", [("c_r1", ">=", 0.0), ("brd_sma7", ">=", 0.9)])
add("c_sma30<=-0.15&btc_sma200<=0", [("c_sma30", "<=", -0.15), ("btc_sma200", "<=", 0.0)])
add("c_r7d<=-0.2&btc_sma50<=0", [("c_r7d", "<=", -0.20), ("btc_sma50", "<=", 0.0)])

COLS = [f"{p}_{m}" for p in ("d", "t", "s") for m in ("mult", "mo", "dd", "n", "tpm", "wr", "nl", "wt", "avgw", "avgl")]


def evaluate(name, rules, stage, deep=None):
    r, _, _, _ = L.run("d", rules=rules, deep=deep)
    row = {"name": name, "stage": stage, "rules": L.rules_str(rules), "deep": deep if deep else 0.0,
           "nrules": len(rules)}
    row.update({c: r[c] for c in COLS})
    return row


def rank(R, base):
    ok = R[R.d_mo >= 0.7 * base["d_mo"]]
    return ok.sort_values(["d_wr", "d_mo"], ascending=False)


def dshort(r):
    return (f"2021-23 WR {r['d_wr']*100:5.1f}% {r['d_mo']*100:+6.2f}%/mo DD {r['d_dd']*100:4.0f}% wt {r['d_wt']*100:+4.0f}% "
            f"{r['d_tpm']:4.1f} tr/mo {int(r['d_nl'])} losers")


def main():
    t0 = time.time()
    log = open(os.path.join(L.OUT, "grid.log"), "w")

    def p(s):
        print(s, flush=True)
        log.write(s + "\n")

    base = evaluate("90% mode", [], "base")
    p(f"90% mode: {dshort(base)}")
    rows = [base]
    rules_of = {"90% mode": []}
    for name, rules in SINGLES.items():
        rows.append(evaluate(name, rules, "S1"))
        rules_of[name] = rules
    R = pd.DataFrame(rows)
    p(f"S1: {len(SINGLES)} singles, {time.time() - t0:.0f}s")
    top_s = [n for n in rank(R[R.stage == "S1"], base).name if n in SINGLES][:24]
    p("top singles (R): " + ", ".join(top_s))
    for a, b in itertools.combinations(top_s, 2):
        nm = f"{a} | {b}"
        rules_of[nm] = rules_of[a] + rules_of[b]
        rows.append(evaluate(nm, rules_of[nm], "S2"))
    R = pd.DataFrame(rows)
    p(f"S2 done ({len(R)} designs), {time.time() - t0:.0f}s")
    top12 = [n for n in rank(R[R.stage.isin(["S1", "S2"])], base).name][:12]
    seen = set(R.name)
    for d in top12:
        for s in top_s:
            if s in d.split(" | "):
                continue
            parts = sorted(d.split(" | ") + [s])
            nm = " | ".join(parts)
            if nm in seen:
                continue
            seen.add(nm)
            rules_of[nm] = [c for q in parts for c in rules_of[q]]
            rows.append(evaluate(nm, rules_of[nm], "S3"))
    R = pd.DataFrame(rows)
    p(f"S3 done ({len(R)} designs), {time.time() - t0:.0f}s")
    top15 = [n for n in rank(R[R.stage != "base"], base).name][:15]
    for d in top15:
        for deep in (0.18, 0.22, 0.26):
            rows.append(evaluate(f"{d} => depth {deep:.2f}", rules_of[d], "S4", deep=deep))
            rules_of[f"{d} => depth {deep:.2f}"] = rules_of[d]
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(L.OUT, "grid.csv"), index=False)
    p(f"all stages: {len(R)} designs, {time.time() - t0:.0f}s")

    # ---------------- selection on 2021-23 only
    ok = R[(R.d_mo >= 0.9 * base["d_mo"]) & (R.d_dd >= base["d_dd"] - 0.01)]
    pick = ok.sort_values(["d_wr", "d_mo"], ascending=False).iloc[0]
    p(f"\nPICK (highest 2021-23 WR with %/mo >= 90% of 90% mode and drop <= 1pt deeper): {pick['name']}")
    p(f"   rules: {pick['rules']} deep {pick['deep']}\n   {dshort(pick)}")
    for k in (0.95, 0.96, 0.97, 0.98, 0.99):
        B = R[(R.d_wr >= k) & (R.d_n >= 100)]
        if not len(B):
            p(f"BAND{int(k*100)}: none")
            continue
        b = B.sort_values(["d_mo", "d_wr"], ascending=False).iloc[0]
        p(f"BAND{int(k*100)}: {b['name']}\n   rules: {b['rules']} deep {b['deep']}\n   {dshort(b)}")
    p("\nTop 25 by ranking R (2021-23 only):")
    for _, r in rank(R, base).head(25).iterrows():
        p(f"  {r['name'][:90]:90s} {dshort(r)}")


if __name__ == "__main__":
    main()
