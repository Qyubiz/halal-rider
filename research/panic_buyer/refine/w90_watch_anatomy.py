"""w90 ten-watch, step 4: anatomy of the finalists (watch bot, wicks 1%, 19 coins unless noted).
  - per-year returns, average win / loss, losing streaks, months, P&L by coin (2024-26)
  - the 10 Oct 2025 flash crash: trades entered that day and the lowest minute-close mark of the account
  - stressed runs (22 coins, fills 0.2% worse both ways, cd+1h): trades in LUNA/FTT/SRM and positions still open at the
    end of the data (a position in a coin that stopped trading cannot be sold by the bot and blocks its slot; it is
    marked at its last traded price, as in every earlier stress run)
  - low-drop >=90%-win alternatives under the stressed version.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_watch_final as F  # noqa: E402
import w90_watch_lib as W  # noqa: E402

ctx = W.Ctx()
hidx = ctx.hidx
DES = list(F.CANDS[:3]) + [
    ("alt N2 d14 tp2 g35 hm96", dict(N=2, fmax=0.5, depth=0.14, tp=0.02, guard=0.35, hm=96, cd=24)),
    ("alt N3 d12 tp2 g35 hm48", dict(N=3, fmax=1 / 3, depth=0.12, tp=0.02, guard=0.35, hm=48, cd=24)),
    ("alt N2 d12 tp2 g35 hm48", dict(N=2, fmax=0.5, depth=0.12, tp=0.02, guard=0.35, hm=48, cd=24)),
    ("alt N3 d14 tp3 g45 hm96", dict(N=3, fmax=1 / 3, depth=0.14, tp=0.03, guard=0.45, hm=96, cd=24)),
    ("alt N3 d14 tp3 g25 hm96", dict(N=3, fmax=1 / 3, depth=0.14, tp=0.03, guard=0.25, hm=96, cd=24))]


def tlog(lg):
    t = lg.copy()
    t["tin"] = hidx[t.eb.astype(int).values]
    t["acct"] = t.ret * t.amt / t.eqb
    t["coin"] = [W.SYMS[int(k)] for k in t.k]
    return t


for name, kw in DES:
    print(f"\n######## {name}", flush=True)
    e, lg, inv, lo = ctx.run(19, **kw)
    r = W.report(e, lg, hidx, lo)
    print(f"  base: {W.line(r)}")
    if W.LAST_OPEN:
        print(f"  open at the end of the data: {W.LAST_OPEN}")
    t = tlog(lg)
    y = E.years(e)
    print("  years: " + ", ".join(f"{k} {v*100:+.0f}%" for k, v in y.items()))
    for a, b, lab in ((W.V.T21, W.V.T24, "2021-23"), (W.V.T24, None, "2024-26"), (W.V.T25, None, "since 2025")):
        g = t[(t.tin >= a) & ((t.tin < b) if b is not None else True)]
        win, los = g[g.ret > 0], g[g.ret <= 0]
        streak = mx = 0
        for v in g.ret.values:
            streak = streak + 1 if v <= 0 else 0
            mx = max(mx, streak)
        print(f"  {lab:10s}: trades {len(g)}, wins {len(win)} ({len(win)/max(1,len(g))*100:.1f}%), avg win "
              f"{win.ret.mean()*100:+.2f}%, avg loss {los.ret.mean()*100:+.2f}% (worst {los.ret.min()*100:+.1f}%), "
              f"take-profits {int((g.why == 0).sum())}, time stops {int((g.why == 1).sum())} "
              f"(of which {int(((g.why == 1) & (g.ret > 0)).sum())} profitable), longest losing streak {mx}")
    g = t[t.tin >= W.V.T24]
    pc = g.groupby("coin").acct.apply(lambda s: np.log1p(s).sum()).sort_values()
    tot = pc.sum()
    print(f"  2024-26 P&L by coin (log points, total {tot:+.2f}): best " +
          ", ".join(f"{c} {v:+.2f}" for c, v in pc[::-1].head(5).items()) + " | worst " +
          ", ".join(f"{c} {v:+.2f}" for c, v in pc.head(3).items()))
    oc = t[(t.tin >= "2025-10-10") & (t.tin < "2025-10-12")]
    for x in oc.itertuples():
        print(f"  10 Oct 2025: {x.coin} bought {hidx[int(x.fm)//60] + pd.Timedelta(minutes=int(x.fm) % 60)} @ "
              f"{x.entry:.4g}, sold {hidx[int(x.xm)//60] + pd.Timedelta(minutes=int(x.xm) % 60)} @ {x.px:.4g} "
              f"({'take-profit' if x.why == 0 else 'time stop'}) {x.ret*100:+.1f}% = {x.acct*100:+.1f}% of the account")
    m = (hidx >= "2025-10-10") & (hidx < "2025-10-12")
    pk = e[hidx < "2025-10-10 20:00"].iloc[-1]
    print(f"  10 Oct 2025: account before 20:00 x{pk:.3f}; lowest minute-close mark x{lo[m].min():.3f} "
          f"({(lo[m].min()/pk-1)*100:+.0f}%); lowest hourly mark x{e[m].min():.3f} ({(e[m].min()/pk-1)*100:+.0f}%)")
    # stressed
    s = dict(kw); s.update(W.STRESS)
    ex, lx, _, lox = ctx.run(22, **s)
    rx = W.report(ex, lx, hidx, lox)
    print(f"  STRESSED: {W.line(rx)}")
    tx = tlog(lx)
    d = tx[tx.k >= 19]
    for x in d.itertuples():
        print(f"    dead-coin trade: {x.coin} {x.tin:%Y-%m-%d %H:%M} {x.ret*100:+.1f}% = {x.acct*100:+.1f}% of the account")
    if W.LAST_OPEN:
        print(f"    STILL OPEN at the end (stuck, slot blocked): {[(c, round(p, 4), str(h)) for c, p, h in W.LAST_OPEN]}")
    print(f"    stressed trades per year: " + ", ".join(f"{yy} {n}" for yy, n in tx.tin.dt.year.value_counts().sort_index().items()))
