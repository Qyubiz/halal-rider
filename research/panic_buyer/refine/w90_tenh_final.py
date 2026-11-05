"""Task "ten-hourly" finalists, ready for the 2018-20 holdout. All three were chosen by looking at 2024-26 and since
2025 (the round's fixed rule uses those periods), so every number here is HINDSIGHT / in-sample.

F1  wr90 rule pick: 1 coin x 100%, 15% dip, +4%, 96h time stop, 24h cooldown after a loss, re-arm the coin at the
    next hourly check after a win (cdw=1), crash guard 35% / 168h.
F2  robust 95% win rate: 1 coin x 100%, 13% dip, +1.5%, 96h, 24h cooldown after a loss, re-arm after a win,
    crash guard 35% / 168h (40% lets LUNA through in the stressed run: keep 35%).
F3  closest to 10% under stress, config only (no code change): 1 coin x 100%, 15% dip, +6%, 48h, 24h cooldown,
    crash guard 40% / 168h.

holdout_candidate_F1/F2/F3() -> (name, spec, runner): spec(P, syms) -> kwargs for runner(P, idx, **kw), runner has
engine.run's signature (returns equity Series, trade log) and works on ANY hourly panel from engine.panel (needs
P['o','h','l','c','cf','ref']). The crash-guard mask is computed from P['cf'] with no caching.
holdout_candidate_Fx_stressed(): same with the stressed version (real bot timing: take-profit at once after a market
buy, cooldowns counted from the next hourly check -> cd+1 and cdw+1; every buy and sell 0.2% worse; a coin whose
data ends early is sold at its last hour). Add LUNA/FTT/SRM to the panel yourself if they exist in the period.
holdout_candidate() = F1.

Run as a script: reproduces each finalist through its own holdout function on the 19 coins (must equal
w90_tenh_lib.evaluate) and prints the full report (normal, stressed, config-only fallback for F1/F2).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402

FINALISTS = {
    "F1 wr90 pick: N1 x100% d15 tp4 hm96 cd24 re-arm-after-win guard35/168":
        dict(N=1, fmax=1.0, depth=0.15, tp=0.04, hm=96, cd=24, cdw=1, g=0.35, gL=168),
    "F2 robust 95% win: N1 x100% d13 tp1.5 hm96 cd24 re-arm-after-win guard35/168":
        dict(N=1, fmax=1.0, depth=0.13, tp=0.015, hm=96, cd=24, cdw=1, g=0.35, gL=168),
    "F3 ten-direction, config only: N1 x100% d15 tp6 hm48 cd24 guard40/168":
        dict(N=1, fmax=1.0, depth=0.15, tp=0.06, hm=48, cd=24, g=0.40, gL=168),
}
BOT_CFG = {
    "F1": {"n_orders": 1, "max_order_pct": 100.0, "depth_pct": 15.0, "take_profit_pct": 4.0, "max_hours": 96,
           "cooldown_hours": 24, "crash_guard_pct": 35.0, "cooldown_after_win_hours": 0},
    "F2": {"n_orders": 1, "max_order_pct": 100.0, "depth_pct": 13.0, "take_profit_pct": 1.5, "max_hours": 96,
           "cooldown_hours": 24, "crash_guard_pct": 35.0, "cooldown_after_win_hours": 0},
    "F3": {"n_orders": 1, "max_order_pct": 100.0, "depth_pct": 15.0, "take_profit_pct": 6.0, "max_hours": 48,
           "cooldown_hours": 24, "crash_guard_pct": 40.0},
}


def guard_mask(P, g, L=168):
    """1.0 = allowed: the coin's last close is no more than g under its highest close of the previous L hours
    (known at the hour's open; panic_buyer.py crash_guard_pct with L=168). Uncached, safe for any panel."""
    cf = P["cf"]
    hi = cf.rolling(L, min_periods=1).max()
    return ((cf / hi - 1) >= -g).shift(1, fill_value=True).values.astype(float)


def _spec(design, stressed=False):
    d = dict(design)
    g, L = d.pop("g"), d.pop("gL")

    def spec(P, syms=None):
        kw = dict(d)
        kw["smult"] = guard_mask(P, g, L)
        if stressed:
            kw.update(mtp_now=True, cd=d["cd"] + 1, pen=0.002, pens=0.002, liq=1)
            if d.get("cdw", -1) >= 0:
                kw["cdw"] = d["cdw"] + 1
        return kw
    return spec


def runner(P, idx, **kw):
    """w90_tenh_lib.trun: engine.run's signature + cdw (cooldown after a win), liq, pens, ladder/tpf switches.
    slip defaults to 'auto' (0.02% for the 19 live coins, 0.1% for any other coin)."""
    return T.trun(P, idx, **kw)


def _cand(i, stressed=False):
    name, design = list(FINALISTS.items())[i]
    return (name + (" [stressed]" if stressed else ""), _spec(design, stressed), runner)


def holdout_candidate_F1():
    return _cand(0)


def holdout_candidate_F2():
    return _cand(1)


def holdout_candidate_F3():
    return _cand(2)


def holdout_candidate_F1_stressed():
    return _cand(0, True)


def holdout_candidate_F2_stressed():
    return _cand(1, True)


def holdout_candidate_F3_stressed():
    return _cand(2, True)


def holdout_candidate():
    return holdout_candidate_F1()


if __name__ == "__main__":
    out = open(os.path.join(T.OUT, "final.log"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n"); out.flush()

    syms, idx, P = T.market(False)
    sd, idd, Pd = T.market(True)
    for i, (name, design) in enumerate(FINALISTS.items()):
        n, spec, run = _cand(i)
        e, lg = run(P, idx, **spec(P, syms))
        r = T.summarize(e, lg, idx)
        r0, e0, _ = T.evaluate(design)
        same = np.array_equal(e.values, e0.values)
        ns, specs, _ = _cand(i, True)
        es, lgs = run(Pd, idd, **specs(Pd, sd))
        rs = T.summarize(es, lgs, idd)
        rs0, es0, _ = T.evaluate(design, stress=True)
        same_s = np.array_equal(es.values, es0.values)
        say("=" * 150)
        say(T.fmt(name, r, rs))
        say(f"   holdout function reproduces evaluate(): normal {same}, stressed {same_s}")
        say(f"   live-bot config: {BOT_CFG[name[:2]]}")
        if design.get("cdw", -1) >= 0:
            d2 = {k: v for k, v in design.items() if k != "cdw"}
            r2, _, _ = T.evaluate(d2)
            rs2, _, _ = T.evaluate(d2, stress=True)
            say(T.fmt("   config-only fallback (no re-arm code change, plain 24h cooldown)", r2, rs2))
        y = " ".join(f"{yy} {T.year_ret(e, yy)*100:+.0f}%" for yy in range(2021, 2027))
        say(f"   years: {y}")
    out.close()
