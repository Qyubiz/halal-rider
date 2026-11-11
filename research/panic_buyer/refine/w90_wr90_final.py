"""Task "wr90": finalists, their holdout functions, the frontier by win-rate band, and the profit-only time stop anatomy.

Finalists (all chosen by looking at 2021-23 AND 2024-26 AND since-2025, i.e. tuned in-sample on all of 2021-26; F1 and
F2 come from grid D and the stop / guard-25% variants from grid F, both placed after seeing 2024-26 results):
  F1  rule pick: 1 coin x 100%, 13% dip, +1.25%, 96h time stop, 3h cooldown, crash guard 35%, catastrophe stop 30%.
      Ties F1-without-stop exactly on min %/month (the stop never fires after 2023) with a shallower drop, so it is the
      rule's literal pick. holdout_candidate_f1_nostop() is the config-only version.
  F2  1 coin x 100%, 14% dip, +2.5%, 96h, 3h cooldown, guard 35%: within 0.05 points of F1 on the rule, flatter
      neighbourhood, better under stress.
  M   2 coins x 50%, 14% dip, +2.5%, 96h, 3h cooldown, guard 25%: best design with worst drop >= -40% normal and >= -45%
      stressed.
Reference (not a finalist): LOWDD 4 coins x <=50%, 16% dip, +2%, 96h, 24h cooldown, guard 35% (pre-declared grid A).

holdout_candidate*() -> (name, spec, runner): spec is an engine.run-style dict usable on ANY hourly panel from
engine.panel (the crash guard mask is computed from the panel passed in); runner = w90_wr90_lib.wrun, which has
engine.run's signature (P, idx, **kw) and returns (equity Series, log with eb/ret columns). Positions still open at the
end of a run are logged with why=9 and their marked return, so they count in the win rate.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

SPEC = {
    "F1": dict(N=1, fmax=1.0, depth=0.13, tp=0.0125, hm=96, cd=3, guard=0.35, hs=0.30),
    "F1_nostop": dict(N=1, fmax=1.0, depth=0.13, tp=0.0125, hm=96, cd=3, guard=0.35),
    "F2": dict(N=1, fmax=1.0, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.35),
    "M": dict(N=2, fmax=0.5, depth=0.14, tp=0.025, hm=96, cd=3, guard=0.25),
    "LOWDD": dict(N=4, fmax=0.5, depth=0.16, tp=0.02, hm=96, cd=24, guard=0.35),
}
NAMES = {
    "F1": "wr90 F1: 1x100%, 13%, +1.25%, 96h, cd3, guard35, stop30",
    "F1_nostop": "wr90 F1 config-only: 1x100%, 13%, +1.25%, 96h, cd3, guard35",
    "F2": "wr90 F2: 1x100%, 14%, +2.5%, 96h, cd3, guard35",
    "M": "wr90 M: 2x50%, 14%, +2.5%, 96h, cd3, guard25",
    "LOWDD": "wr90 LOWDD (reference): 4x<=50%, 16%, +2%, 96h, cd24, guard35",
}
BOT_CFG = {k: {"n_orders": s["N"], "max_order_pct": s["fmax"] * 100, "depth_pct": s["depth"] * 100,
               "take_profit_pct": s["tp"] * 100, "max_hours": s["hm"], "cooldown_hours": s["cd"],
               "crash_guard_pct": s["guard"] * 100} for k, s in SPEC.items()}


def _cand(k):
    return NAMES[k], dict(SPEC[k]), W.wrun


def holdout_candidate():
    """The rule's literal pick (F1, needs the 30% catastrophe stop in the bot)."""
    return _cand("F1")


def holdout_candidate_f1():
    return _cand("F1")


def holdout_candidate_f1_nostop():
    return _cand("F1_nostop")


def holdout_candidate_f2():
    return _cand("F2")


def holdout_candidate_m():
    return _cand("M")


def holdout_candidate_lowdd():
    return _cand("LOWDD")


def via_holdout_path(fn, dead=False, **over):
    """Run a holdout_candidate function exactly the way holdout_eval.evaluate does, on the 2021-26 panel."""
    name, spec, runner = fn()
    syms, idx, P = E.panel(E.load_h19(dead=dead))
    kw = dict(spec(P, syms)) if callable(spec) else dict(spec)
    kw.update(over)
    e, lg = runner(P, idx, **kw)
    return W.summarize(e, lg, idx), E.score(e, lg[lg.why != 9], idx)


def short(r):
    return (f"WR {r['d_wr']*100:4.1f}/{r['t_wr']*100:4.1f}/{r['s_wr']*100:4.1f}% | %/mo {r['d_mo']*100:+5.2f}/{r['t_mo']*100:+5.2f}/"
            f"{r['s_mo']*100:+5.2f} | drop {r['d_dd']*100:3.0f}/{r['t_dd']*100:3.0f}/{r['s_dd']*100:3.0f}% | worst trade "
            f"{r['d_wt']*100:+.0f}/{r['t_wt']*100:+.0f}/{r['s_wt']*100:+.0f}% of account")


def finalists():
    print("FINALISTS (2021-23 / 2024-26 / since 2025); stressed = real bot timing (market-buy take-profit at once, "
          "cooldown +1h), every fill 0.2% worse on buys and sells, LUNA/FTT/SRM added")
    fns = {"F1": holdout_candidate_f1, "F1_nostop": holdout_candidate_f1_nostop, "F2": holdout_candidate_f2,
           "M": holdout_candidate_m, "LOWDD": holdout_candidate_lowdd}
    for k, fn in fns.items():
        r, _, _ = W.evaluate(SPEC[k])
        rs, _, _ = W.evaluate(SPEC[k], stress=True)
        syms, idx, P = W.market(True)
        eb, lb = W.wrun(P, idx, **dict(SPEC[k], mtp_now=True, cd=SPEC[k]["cd"] + 1, pen=0.002, liquidate=True))
        rb = W.summarize(eb, lb, idx)
        hp, sc = via_holdout_path(fn)
        same = all(np.isclose(hp[c], r[c], rtol=0, atol=1e-12) for c in ("d_mo", "t_mo", "s_mo", "d_dd", "d_wr", "t_wr"))
        print(f"\n{NAMES[k]}\n  bot config: {BOT_CFG[k]}" + ("  + code change: sell at an hourly close <= entry x 0.70"
                                                               if SPEC[k].get("hs") else ""))
        print("  normal            " + short(r) + f" | trades {r['d_n']}/{r['t_n']}/{r['s_n']}, 2022 {r['y2022']*100:+.0f}%")
        print("  STRESSED          " + short(rs))
        print("  stressed buy-only " + short(rb) + "  (REFINE.md convention)")
        print(f"  holdout_candidate path reproduces the normal run: {'yes' if same else 'NO'} "
              f"(engine.score: 2021-23 x{sc['d_mult']:.2f} dd {sc['d_dd']*100:.0f}%, 2024-26 x{sc['t_mult']:.2f})")
        # the same path with the holdout_eval overrides (fills 0.2% worse on buys; +LUNA/FTT/SRM)
        hp2, _ = via_holdout_path(fn, pen=0.002)
        hp3, _ = via_holdout_path(fn, dead=True)
        print(f"  via holdout path, pen=0.002: %/mo {hp2['d_mo']*100:+.2f}/{hp2['t_mo']*100:+.2f}/{hp2['s_mo']*100:+.2f}; "
              f"+LUNA/FTT/SRM (no liquidation, as holdout_eval runs it): {hp3['d_mo']*100:+.2f}/{hp3['t_mo']*100:+.2f}/"
              f"{hp3['s_mo']*100:+.2f} drop {hp3['d_dd']*100:.0f}%")


def frontier_bands():
    import w90_wr90_frontier as F
    import w90_wr90_grid as G
    R = F.load()
    R0 = R[R.pmode == 0]
    print(f"\nFRONTIER BY WIN-RATE BAND ({len(R0)} designs with the normal time stop; win rate = lower of 2021-23 and 2024-26)")
    print("band: best min(%/mo) [rule] | best with worst drop >= -40% | best with worst drop >= -25%")
    for lo, hi in ((0.85, 0.88), (0.88, 0.90), (0.90, 0.92), (0.92, 0.94), (0.94, 0.95), (0.95, 0.96), (0.96, 0.97),
                   (0.97, 0.98), (0.98, 1.01)):
        B = R0[(R0.minwr >= lo) & (R0.minwr < hi)]
        if not len(B):
            print(f"  WR {lo*100:.0f}-{min(hi, 1)*100:.0f}%: none")
            continue
        cells = []
        for cap in (-1.0, -0.40, -0.25):
            x = G.rank(B[B.mindd >= cap])
            if len(x):
                r = x.iloc[0]
                cells.append(f"{F.name(r)}: {r.d_mo*100:+.1f}/{r.t_mo*100:+.1f}/{r.s_mo*100:+.1f} dd {r.mindd*100:.0f}% "
                             f"(stressed {r.x_d_mo*100:+.1f}/{r.x_t_mo*100:+.1f}/{r.x_s_mo*100:+.1f} dd {r.x_mindd*100:.0f}%)")
            else:
                cells.append("none")
        print(f"  WR {lo*100:.0f}-{min(hi, 1)*100:.0f}% ({len(B)}):\n     " + "\n     ".join(cells))
    Q = G.qualify(R0)
    print(f"\nAt WR>=90% (normal time stop, {len(Q)} designs): max 2024-26 {Q.t_mo.max()*100:+.2f}%/mo, max since-2025 "
          f"{Q.s_mo.max()*100:+.2f}%/mo, max 2021-23 {Q.d_mo.max()*100:+.2f}%/mo; designs >= 10%/mo in both 2021-23 and "
          f"2024-26: {int(((Q.d_mo >= 0.1) & (Q.t_mo >= 0.1)).sum())}; any design (any WR) >= 10% in 2024-26: "
          f"{int((R.t_mo >= 0.1).sum())} of {len(R)}")


def profit_only_anatomy():
    """What the profit-only time stop does: longest holds, capital frozen, the win rate it reports."""
    print("\nPROFIT-ONLY TIME STOP (after hm hours the target drops to break-even +0.25%; no forced loss unless a stop)")
    cases = [("F2 plain time stop", SPEC["F2"]),
             ("F2 profit-only, no stop", dict(SPEC["F2"], pmode=1)),
             ("F2 profit-only, stop 40%", dict(SPEC["F2"], pmode=1, hs=0.40)),
             ("F2 profit-only, max hold 30 days", dict(SPEC["F2"], pmode=1, hmax=720)),
             ("M plain time stop", SPEC["M"]),
             ("M profit-only, no stop", dict(SPEC["M"], pmode=1)),
             ("M profit-only, max hold 30 days", dict(SPEC["M"], pmode=1, hmax=720)),
             ("LIVE plain", dict(N=8, fmax=0.2, depth=0.12, tp=0.015, hm=48, cd=24, guard=0.0)),
             ("LIVE profit-only, no stop", dict(N=8, fmax=0.2, depth=0.12, tp=0.015, hm=48, cd=24, guard=0.0, pmode=1))]
    for name, kw in cases:
        r, e, lg = W.evaluate(kw)
        rs, _, _ = W.evaluate(kw, stress=True)
        syms, idx, P = W.market(False)
        hold = (lg.xb - lg.eb)
        top = lg.loc[hold.idxmax()]
        print(f"  {name:36s} " + short(r))
        print(f"  {'':36s} longest hold {int(hold.max())}h ({hold.max()/24:.0f} days): {syms[int(top.k)][:-4]} bought "
              f"{idx[int(top.eb)]:%Y-%m-%d}, {'still open, marked' if top.why == 9 else 'sold'} at {top.ret*100:+.0f}%; "
              f"holds > 30 days: {int((hold > 720).sum())}; open at end: {int((lg.why == 9).sum())} | stressed WR "
              f"{rs['d_wr']*100:.1f}/{rs['t_wr']*100:.1f}% %/mo {rs['d_mo']*100:+.1f}/{rs['t_mo']*100:+.1f}/{rs['s_mo']*100:+.1f} "
              f"drop {rs['mindd']*100:.0f}%")


if __name__ == "__main__":
    finalists()
    frontier_bands()
    profit_only_anatomy()
