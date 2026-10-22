"""Task "ten-wider": finalists, their checks, and the holdout functions.

Finalists (all found with 2024-26 visible, i.e. TUNED ON 2024-26; grids 2-3 were placed by hindsight):
  F1  rule's literal pick (highest worst-of-three %/month among the 10% qualifiers of grids 1-3):
      19 core at 14% + monthly top 30 other screened coins by 365-day volume at 26%, full size; +5%, 48h, cooldown 12h.
  F2  same with the 10% mode's 24h cooldown (the core is exactly today's 10% mode; only the tier is added).
  F3  19 core at 14% + monthly top 20 other screened coins by 30-day volume at 20%, size x0.75; 24h cooldown.

Checks per finalist: normal and stressed (real bot timing mtp_now + cd+1, every buy and sell fill 0.2% worse,
LUNA/FTT/SRM forced in as core coins, positions sold at a coin's last data hour); the same settings on the 19 alone;
no new buys on the 10 Oct 2025 flash-crash evening; stricter fills (orders must trade 0.3% / 1% through);
fresh starts; one-step neighbours; P&L by coin outside the 19; worst trades; holdout functions run through the
holdout_eval.evaluate path on the 19-coin 2021-26 panel.

holdout_candidate*() -> (name, spec, runner): spec = f(P, syms) -> kwargs for runner (engine.run's signature).
On a panel holding only the user's 19 coins (e.g. the 2018-20 holdout) the tier is empty, so the holdout tests the
14% core (= 10% mode, with F1's 12h cooldown for F1). If the panel holds other coins, they are ranked within the
panel by trailing quote volume (needs P['qv'], else all are tier-eligible).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_wider_lib as L  # noqa: E402

T = L.TENMODE
FINALISTS = {
    "F1": dict(T, rk="v365", M=30, dt=0.26, st=1.0, tt=0.05, cd=12),
    "F2": dict(T, rk="v365", M=30, dt=0.26, st=1.0, tt=0.05, cd=24),
    "F3": dict(T, rk="v30", M=20, dt=0.20, st=0.75, tt=0.05, cd=24),
}
BOT_CFG = {
    "F1": ("depth_pct 14, n_orders 1, max_order_pct 100, take_profit_pct 5, max_hours 48, cooldown_hours 12, "
           "crash_guard_pct 35; CODE CHANGE: tier list = monthly top 30 non-19 screened coins by trailing 365-day "
           "USDT volume (ranked on the 1st of each month from daily klines, min 30 days listed), tier_depth_pct 26, "
           "tier_size_mult 1.0, same take-profit; per-coin depth in triggers()"),
    "F2": ("depth_pct 14, n_orders 1, max_order_pct 100, take_profit_pct 5, max_hours 48, cooldown_hours 24, "
           "crash_guard_pct 35 (= panic_config.10pct.example.json); CODE CHANGE: tier list = monthly top 30 non-19 "
           "screened coins by trailing 365-day USDT volume, tier_depth_pct 26, tier_size_mult 1.0"),
    "F3": ("depth_pct 14, n_orders 1, max_order_pct 100, take_profit_pct 5, max_hours 48, cooldown_hours 24, "
           "crash_guard_pct 35; CODE CHANGE: tier list = monthly top 20 non-19 screened coins by trailing 30-day "
           "USDT volume, tier_depth_pct 20, tier_size_mult 0.75 (order = 75% of cash)"),
}
FLASH = [("2025-10-10 20:00", "2025-10-11 00:00")]


def _cand(k):
    d = FINALISTS[k]
    return (f"ten-wider {k}: {L.name(d)}", L.portable_spec(d), L.runner)


def holdout_candidate():
    return _cand("F1")


def holdout_candidate_F1():
    return _cand("F1")


def holdout_candidate_F2():
    return _cand("F2")


def holdout_candidate_F3():
    return _cand("F3")


def neighbours(d):
    """One-step neighbours: M +-5 (v30: +-5), tier depth +-2 pts, tier size, core depth +-1 pt, cooldown +-12h."""
    out = []
    for M in (d["M"] - 5, d["M"] + 5):
        out.append(dict(d, M=M))
    for dt in (d["dt"] - 0.02, d["dt"] + 0.02):
        out.append(dict(d, dt=round(dt, 4)))
    for st in (0.5, 0.75, 1.0):
        if st != d["st"]:
            out.append(dict(d, st=st))
    for dc in (d["dc"] - 0.01, d["dc"] + 0.01):
        out.append(dict(d, dc=round(dc, 4)))
    for cd in (d["cd"] - 12, d["cd"] + 12):
        if cd >= 0:
            out.append(dict(d, cd=cd))
    for tp in (0.04, 0.06):
        out.append(dict(d, tp=tp, tt=tp))
    return out


def pct(x):
    return f"{x * 100:+.2f}%"


def main():
    W = L.build()
    out = open(os.path.join(L.OUT, "final.log"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n"); out.flush()
    rows = []
    say("All numbers: wider panel (screened coins, delisted included), costs 0.1% fee/side, slippage 0.02% (monthly "
        "top 50 by 30d volume) or 0.1%, limit buys 0.1% through.\nColumns: win rate (trades bought in the period), "
        "%/month compounded, worst drop, for 2021-23 | 2024-26 | since 2025.")
    for k, d in FINALISTS.items():
        say("\n" + "=" * 160)
        say(f"{k}: {L.name(d)}")
        say(f"   live bot: {BOT_CFG[k]}")
        r, e, lg, syms = L.evaluate(d, W=W)
        rs, es, lgs, symss = L.evaluate(d, stress=True, W=W)
        say("   normal    " + L.fmt(r))
        say("   STRESSED  " + L.fmt(rs))
        say("   years normal:   " + " ".join(f"{y} {L.year_ret(e, y) * 100:+.0f}%" for y in range(2021, 2027)))
        say("   years stressed: " + " ".join(f"{y} {L.year_ret(es, y) * 100:+.0f}%" for y in range(2021, 2027)))
        say(f"   trades 21-23/24-26/25-26: {r['d_n']}/{r['t_n']}/{r['s_n']}; outside the 19: {r['d_n_out']}/{r['t_n_out']}/{r['s_n_out']}")
        say(f"   avg win / avg loss 24-26: {pct(r['t_avgw'])} / {pct(r['t_avgl'])}; 21-23: {pct(r['d_avgw'])} / {pct(r['d_avgl'])}")
        say(f"   worst single trade, % of account: normal {pct(r['d_wacct'])} / {pct(r['t_wacct'])} / {pct(r['s_wacct'])}; "
            f"stressed {pct(rs['d_wacct'])} / {pct(rs['t_wacct'])} / {pct(rs['s_wacct'])}")
        for tag, lab in (("d", "2021-23"), ("t", "2024-26"), ("s", "since 2025")):
            say(f"   {lab}: account P&L points {r[f'{tag}_pnl'] * 100:+.0f}, from coins outside the 19 "
                f"{r[f'{tag}_pnl_out'] * 100:+.0f} ({r[f'{tag}_pnl_out'] / r[f'{tag}_pnl'] * 100:.0f}%)")
        r19, *_ = L.evaluate(dict(d, M=0), W=W)
        rs19, *_ = L.evaluate(dict(d, M=0), stress=True, W=W)
        say("   same settings, 19 coins only:          " + L.fmt(r19))
        say("   same settings, 19 coins only, STRESSED: " + L.fmt(rs19))
        idx = W["idx"]
        for lab, a, b in (("2024-26", L.T24, None), ("since 2025", L.T25, None), ("2021-23", L.T21, L.T24)):
            bc = L.by_coin(lg, idx, syms, a, b)
            bc = bc[[s not in E.H19 for s in bc.index]].sort_values("pnl", ascending=False)
            say(f"   {lab} coins outside the 19 ({len(bc)}; P&L points, trades): " +
                ", ".join(f"{s[:-4]} {p * 100:+.0f} ({int(n)})" for s, (p, n, _) in bc.iterrows()))
        lg2 = L.closed(lg).copy()
        lg2["acct"] = lg2.ret * lg2.amt / lg2.eq0
        lg2["when"] = idx[lg2.eb.astype(int).values]
        lg2["coin"] = [syms[int(q)][:-4] for q in lg2.k]
        w_ = lg2.sort_values("acct").head(6)
        say("   worst trades: " + "; ".join(f"{c} {t:%Y-%m-%d} {a * 100:+.0f}% of account" for c, t, a in
                                         zip(w_.coin, w_.when, w_.acct)))
        fl = lg2[(lg2.when >= "2025-10-10") & (lg2.when < "2025-10-12")]
        say(f"   trades bought 10-11 Oct 2025: " + "; ".join(f"{c} {t:%m-%d %H}h {a * 100:+.1f}% of account" for c, t, a in
                                                           zip(fl.coin, fl.when, fl.acct)))
        # robustness
        rb, *_ = L.evaluate(d, W=W, block=FLASH)
        say("   no new buys 10 Oct 2025 20-24h UTC:  " + L.fmt(rb))
        for eps in (0.003, 0.01):
            re_, *_ = L.evaluate(d, W=W, eps=eps)
            say(f"   orders must trade {eps * 100:.1f}% through:     " + L.fmt(re_))
        fr, fr25 = [], []
        for st in pd.date_range("2024-01-01", periods=5, freq="7D"):
            _, ef, *_x = L.evaluate(d, W=W, start=st)
            fr.append(L.period(ef, st, None)[1])
        for st in pd.date_range("2025-01-01", periods=5, freq="7D"):
            _, ef, *_x = L.evaluate(d, W=W, start=st)
            fr25.append(L.period(ef, st, None)[1])
        say(f"   fresh starts 2024-01-01 +0/7/14/21/28 days, %/month to date: " + ", ".join(f"{x * 100:.2f}" for x in fr))
        say(f"   fresh starts 2025-01-01 +0/7/14/21/28 days, %/month to date: " + ", ".join(f"{x * 100:.2f}" for x in fr25))
        nb = []
        for dn in neighbours(d):
            rn, *_ = L.evaluate(dn, W=W)
            nb.append(rn)
            say(f"      neighbour {L.name(dn)}: {L.fmt(rn)}")
        nbm = np.array([x["min_mo"] for x in nb]); nbt = np.array([x["t_mo"] for x in nb])
        say(f"   neighbours ({len(nb)}): worst-of-three %/month median {np.median(nbm) * 100:.2f}, min {nbm.min() * 100:.2f}, "
            f"max {nbm.max() * 100:.2f}; 2024-26 median {np.median(nbt) * 100:.2f}; qualify for 10%: "
            f"{sum(x['ten'] for x in nb)} of {len(nb)}")
        # holdout path on the 19-coin 2021-26 panel (what the lead's 2018-20 run will test) and on the wider panel
        name, spec, run = _cand(k)
        for dead in (False, True):
            s19, i19, P19 = E.panel(E.load_h19(dead=dead))
            eh, lh = run(P19, i19, **spec(P19, s19))
            sh = L.summarize(eh, lh, i19, np.array([s in E.H19 for s in s19]))
            say(f"   holdout_candidate_{k} via holdout path, 19 coins{' + LUNA/FTT/SRM' if dead else ''} (core only): " + L.fmt(sh))
        rows.append(dict(k=k, name=L.name(d), **r, **{f"S{q}": v for q, v in rs.items()}))
    pd.DataFrame(rows).to_csv(os.path.join(L.OUT, "finalists.csv"), index=False)
    out.close()


if __name__ == "__main__":
    main()
