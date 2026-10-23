"""wr90 calibration: reproduce the stated facts (live settings, 10% mode) with w90_wr90_lib metrics, and show what the
stress model (bot timing, fills 0.2% worse on buys only vs buys+sells, LUNA/FTT/SRM with liquidation) does."""
import w90_wr90_lib as W
import engine as E

LIVE = dict(E.LIVE)
TEN = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, guard=0.35)
for name, kw in (("LIVE", LIVE), ("10% mode", TEN)):
    r, e, lg = W.evaluate(kw)
    print(W.fmt(name + " (normal)", r))
    for lab, k, dead in (("bot timing + buys 0.2% worse, 19 coins", dict(kw, mtp_now=True, cd=kw["cd"] + 1, pen=0.002), False),
                         ("bot timing + buys 0.2% worse, 22 coins", dict(kw, mtp_now=True, cd=kw["cd"] + 1, pen=0.002, liquidate=True), True)):
        syms, idx, P = W.market(dead)
        e, lg = W.wrun(P, idx, **k)
        print(W.fmt(f"{name} | {lab}", W.summarize(e, lg, idx)))
    r, e, lg = W.evaluate(kw, stress=True)
    print(W.fmt(f"{name} | STRESSED (bot timing, buys+sells 0.2% worse, +LUNA/FTT/SRM)", r))
