"""Shared test protocol for every refinement idea.

1. Each idea is a small grid of variants declared before any result is seen.
2. The variant is chosen on 2021-23 only (engine.pick_on_design): it must beat the live settings there.
3. 2024-26 is then looked at once for that pick. The idea passes only if the pick also beats the live
   settings in 2024-26 (engine.beats), and still does with LUNA/FTT/SRM added and with fills 0.2% worse.
"""
import os

import numpy as np
import pandas as pd

import engine as E

OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out"))
os.makedirs(OUT, exist_ok=True)
_cache = {}


def market(dead=False):
    """(syms, idx, P) for the 19 live coins, optionally with the LUNA/FTT/SRM collapses."""
    if dead not in _cache:
        _cache[dead] = E.panel(E.load_h19(dead=dead))
    return _cache[dead]


def evaluate(spec, dead=False, **over):
    syms, idx, P = market(dead)
    kw = dict(spec(P, syms)) if callable(spec) else dict(spec)
    kw.update(over)
    e, lg = E.run(P, idx, **kw)
    return E.score(e, lg, idx), e, lg


def study(title, variants, tag):
    """variants: {name: dict of engine.run settings, or f(P, syms) -> dict}. Returns the result table."""
    print(f"\n######## {title}")
    base, base_e, _ = evaluate({})
    print(E.line("LIVE SETTINGS", base))
    rows = []
    for name, spec in variants.items():
        r, _, _ = evaluate(spec)
        r["name"] = name
        r["beats_design"], r["beats_test"], r["beats_both"] = E.beats(r, base, "design"), E.beats(r, base, "test"), E.beats(r, base)
        rows.append(r)
        print(E.line(name, r) + ("  <- beats live in both" if r["beats_both"] else ""))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, f"{tag}.csv"), index=False)
    print(f"{len(R)} variants | beat live in 2021-23: {int(R.beats_design.sum())} | in 2024-26: {int(R.beats_test.sum())} "
          f"| in both: {int(R.beats_both.sum())}")
    pick = E.pick_on_design(rows, base)
    verdict = {"idea": title, "variants": len(R), "beat_design": int(R.beats_design.sum()),
               "beat_test": int(R.beats_test.sum()), "beat_both": int(R.beats_both.sum()), "pick": None, "passes": False}
    if pick is None:
        print("PICK (2021-23 only): none of the variants beats the live settings in 2021-23 -> idea rejected")
        return R, verdict
    print(f"PICK (2021-23 only): {pick['name']}")
    print(E.line("  pick", pick))
    ok = pick["beats_test"]
    print(f"  2024-26 check: {'BEATS' if ok else 'does NOT beat'} the live settings")
    checks = {}
    for label, dead, pen in (("+LUNA/FTT/SRM", True, 0.0), ("fills 0.2% worse", False, 0.002)):
        b, _, _ = evaluate({}, dead=dead, pen=pen)
        c, _, _ = evaluate(variants[pick["name"]], dead=dead, pen=pen)
        checks[label] = E.beats(c, b)
        print(E.line(f"  live, {label}", b))
        print(E.line(f"  pick, {label}", c) + ("  ok" if checks[label] else "  FAILS"))
    verdict.update(pick=pick["name"], pick_d_mult=pick["d_mult"], pick_t_mult=pick["t_mult"], pick_d_dd=pick["d_dd"],
                   pick_t_dd=pick["t_dd"], base_d_mult=base["d_mult"], base_t_mult=base["t_mult"],
                   base_d_dd=base["d_dd"], base_t_dd=base["t_dd"], **{f"check {k}": v for k, v in checks.items()},
                   passes=bool(ok and all(checks.values())))
    print(f"  VERDICT: {'PASSES - candidate for the live bot' if verdict['passes'] else 'rejected, live bot unchanged'}")
    return R, verdict


def save_verdict(v):
    p = os.path.join(OUT, "verdicts.csv")
    old = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
    if len(old):
        old = old[old.idea != v["idea"]]
    pd.concat([old, pd.DataFrame([v])]).to_csv(p, index=False)


# ------------------------------------------------------------------ features known at each hour's start
def roll_vol(P, w):
    """Daily volatility from the last w closed hourly candles (std of hourly log returns x sqrt 24)."""
    r = np.log(P["cf"]).diff()
    return (r.rolling(w, min_periods=w // 2).std() * np.sqrt(24)).shift(1)


def dd24(P):
    """Last close vs the highest close of the last 24 hours (<= 0), known at the hour's start."""
    return (P["cf"] / P["cf"].rolling(24, min_periods=1).max() - 1).shift(1)


def btc_dd24(P):
    return dd24(P)["BTCUSDT"]
