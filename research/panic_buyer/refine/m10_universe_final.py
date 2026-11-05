"""m10 "universe": the one look at 2024-26 / since 2025 for the pick of m10_universe_grid.py, plus the 2018-20
holdout of its 19-coin core.

For the pick and for 90% mode (same panel and costs): per period win rate, %/month, worst drop, worst trade as % of
the account, trades/month, with normal fills, fills 0.2% worse (every buy and sell) and stressed (real bot timing +
fills 0.2% worse + LUNA/FTT/SRM forced in). P&L, trades and losses from coins outside the 19, the coins used.
Checks: same settings on the 19 coins alone; orders must trade 0.3% through; no buys in a coin's last 10 days of data
(a lookahead stand-in for reading Binance's delisting notices); fresh starts.
Holdout: the pick's core (every holdout coin is one of the 19; no universe data exists for 2018-20).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_universe_lib as M  # noqa: E402
from m10_universe_lib import E, L  # noqa: E402


def row(r, t):
    return (f"WR {r[t + '_win'] * 100:5.1f}% | {r[t + '_mo'] * 100:+6.2f}%/mo | drop {r[t + '_dd'] * 100:4.0f}% | "
            f"worst trade {r[t + '_wacct'] * 100:+4.0f}% | {r[t + '_tpm']:5.1f} tr/mo | out-19: {r[t + '_n_out']} tr, "
            f"{r[t + '_loss_out']} losses ({r[t + '_lossp_out'] * 100:+.0f} pts), P&L {r[t + '_pnl_out'] * 100:+.0f} "
            f"of {r[t + '_pnl'] * 100:+.0f} pts | all losses {r[t + '_loss']}")


def hold_sum(e, lg, idx):
    lg = lg[lg.why != 9]
    mult, mo, dd = L.period(e, E.H_START, E.H_END)
    eb = idx[lg.eb.astype(int).values]
    m = np.asarray((eb >= E.H_START) & (eb < E.H_END), bool)
    t = lg[m]
    acct = t.ret * t.amt / t.eq0
    months = 36.0
    return (f"WR {(t.ret > 0).mean() * 100:5.1f}% | {mo * 100:+6.2f}%/mo | drop {dd * 100:4.0f}% | worst trade "
            f"{acct.min() * 100:+4.0f}% | {len(t) / months:5.1f} tr/mo | x{mult:.2f}")


def main():
    W = M.build()
    pick = eval(open(os.path.join(M.OUT, "pick.txt")).read())
    log = open(os.path.join(M.OUT, "final.log"), "w")

    def say(s=""):
        print(s, flush=True); log.write(s + "\n"); log.flush()
    cands = [("PICK", pick), ("90% mode", dict(M.MODE90))]
    if pick.get("M", 0) > 0:
        cands.append(("PICK, 19 coins only", dict(pick, M=0)))
    if pick.get("N", 1) > 1:
        cands.append(("PICK with fmax = 1/N", dict(pick, fmax=1.0 / pick["N"])))
    for lab, d in cands:
        say("=" * 150)
        say(f"{lab}: {M.name(d)}")
        r, e, lg, syms = M.evaluate(d, W=W)
        rp, *_ = M.evaluate(d, W=W, pen=0.002, pens=0.002)
        rs, *_ = M.evaluate(d, W=W, stress=True)
        for t, per in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025")):
            say(f"  {per:10s} normal       {row(r, t)}")
            say(f"  {per:10s} fills -0.2%  {row(rp, t)}")
            say(f"  {per:10s} stressed     {row(rs, t)}")
        say("  years: " + " ".join(f"{y} {L.year_ret(e, y) * 100:+.0f}%" for y in range(2021, 2027)))
        if d.get("M", 0) > 0:
            idx = W["idx"]
            for per, a, b in (("2021-23", M.T21, M.T24), ("2024-26", M.T24, None), ("since 2025", M.T25, None)):
                bc = M.by_coin(lg, idx, syms, a, b)
                bc = bc[[s not in E.H19 for s in bc.index]].sort_values("pnl", ascending=False)
                say(f"  {per} coins outside the 19 ({len(bc)}; P&L pts, trades, win%): " +
                    ", ".join(f"{s[:-4]} {p * 100:+.1f} ({int(n)}, {w * 100:.0f}%)" for s, (p, n, w) in bc.iterrows()))
            x = lg[lg.why != 9].copy()
            x["acct"] = x.ret * x.amt / x.eq0
            x["when"] = idx[x.eb.astype(int).values]
            x["coin"] = [syms[int(q)][:-4] for q in x.k]
            x["out"] = [syms[int(q)] not in E.H19 for q in x.k]
            w = x[x.out].sort_values("acct").head(8)
            say("  worst trades outside the 19: " + "; ".join(
                f"{c} {t:%Y-%m-%d} {a * 100:+.1f}% acct (why {int(y)})" for c, t, a, y in zip(w.coin, w.when, w.acct, w.why)))
            w = x.sort_values("acct").head(6)
            say("  worst trades overall: " + "; ".join(f"{c} {t:%Y-%m-%d} {a * 100:+.1f}% acct" for c, t, a in
                                                     zip(w.coin, w.when, w.acct)))
            allc = sorted({syms[int(q)][:-4] for q in x.k if syms[int(q)] not in E.H19})
            say(f"  EXTRA COINS BOUGHT 2021-26 ({len(allc)}): {' '.join(allc)}")
            allc24 = sorted({c for c, o, t in zip(x.coin, x.out, x.when) if o and t >= M.T24})
            say(f"  extra coins bought since 2024 ({len(allc24)}): {' '.join(allc24)}")
        rx, *_ = M.evaluate(dict(d, xlast=10), W=W)
        say("  no buys in a coin's last 10 days of data (delisting-notice stand-in): " + M.fmt(rx))
        re_, *_ = M.evaluate(d, W=W, eps=0.003)
        say("  orders must trade 0.3% through:                                     " + M.fmt(re_))
        fr = []
        for st in pd.date_range("2024-01-01", periods=4, freq="7D"):
            _, ef, *_x = M.evaluate(d, W=W, start=st)
            fr.append(L.period(ef, st, None)[1])
        say("  fresh starts 2024-01-01 +0/7/14/21 days, %/mo to date: " + ", ".join(f"{v * 100:.2f}" for v in fr))
    # holdout
    say("=" * 150)
    say("2018-20 HOLDOUT (19-coin core only; 16 of the 19 coins existed)")
    for lab, d in (("PICK core", pick), ("90% mode", dict(M.MODE90))):
        for tag, dead, kw in (("normal", False, {}), ("fills -0.2%", False, dict(pen=0.002, pens=0.002)),
                              ("bot timing + fills -0.2%", False, dict(pen=0.002, pens=0.002, mtp_now=True)),
                              ("+LUNA/FTT/SRM", True, {})):
            syms, idx, P = E.panel(E.load_pre(dead=dead))
            dd = dict(d, cd=d.get("cd", 3) + (1 if kw.get("mtp_now") else 0))
            e, lg = M.run_panel(P, idx, dd, **kw)
            say(f"  {lab:10s} {tag:26s} {hold_sum(e, lg, idx)}")
    log.close()


if __name__ == "__main__":
    main()
