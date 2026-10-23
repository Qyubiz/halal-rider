"""Anatomy of 10% mode (1 coin x 100%, 14% dip, +5%, 48h, 24h cooldown, guard 35%/168h): where 2024-26 and
since-2025 money comes from and goes. Measurement only, nothing is picked here.
Per period: trades/month, share of hours in a position, exits by reason (take-profit / time stop), average win and
loss, P&L by coin (sum of account-% per trade), by vol tier of the coin at entry, by market vs limit buys, and
what the 24h high gap looked like at entry. Also: how many hours had at least one eligible coin under its trigger
while the account was fully invested (missed dips)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_tenh_lib as T  # noqa: E402

out = open(os.path.join(T.OUT, "anatomy.log"), "w")


def say(s=""):
    print(s, flush=True)
    out.write(s + "\n"); out.flush()


syms, idx, P = T.market(False)
S10 = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, g=0.35, gL=168)
r, e, lg = T.evaluate(S10)
say(T.fmt("10% mode", r))
eb = idx[lg.eb.astype(int).values]
xb = idx[lg.xb.astype(int).values]
L = lg.assign(t_in=eb, t_out=xb, coin=[syms[int(k)] for k in lg.k], acct=lg.ret * lg.amt / lg.eq0,
              hold=(lg.xb - lg.eb).astype(int))
vr = T.vol_rank(P, 720).values
L["vr"] = [vr[int(a), int(k)] for a, k in zip(lg.eb, lg.k)]
L["vtier"] = pd.cut(L.vr, [0, 1 / 3, 2 / 3, 1.0001], labels=["calm", "mid", "wild"])
ref = P["ref"].values
L["gap_open"] = [P["o"].values[int(a), int(k)] / ref[int(a), int(k)] - 1 for a, k in zip(lg.eb, lg.k)]
for tag, a, b in T.PERIODS:
    m = (L.t_in >= a) & ((L.t_in < b) if b is not None else True)
    g = L[m]
    months = ((idx[-1] if b is None else b) - a).days / 30.4375
    say(f"\n== {tag} {a.date()}..{b.date() if b is not None else 'end'}: {len(g)} trades, {len(g)/months:.1f}/month, "
        f"win {(g.ret>0).mean()*100:.1f}%, avg win {g.ret[g.ret>0].mean()*100:+.2f}% avg loss {g.ret[g.ret<=0].mean()*100:+.2f}%,"
        f" hold median {g.hold.median():.0f}h")
    say("  exits: " + ", ".join(f"{['tp','time','trail','tp2','liq'][int(w)]} {n}" for w, n in g.why.value_counts().items())
        + f" | market buys {int(g.mkt.sum())} (avg {g.ret[g.mkt==1].mean()*100:+.2f}%), limit {int((g.mkt==0).sum())} "
        f"(avg {g.ret[g.mkt==0].mean()*100:+.2f}%)")
    say("  time-stop exits: avg " + f"{g.ret[g.why==1].mean()*100:+.2f}%, sum acct {g.acct[g.why==1].sum()*100:+.0f} pts; "
        f"tp exits sum acct {g.acct[g.why==0].sum()*100:+.0f} pts")
    by = g.groupby("coin").agg(n=("ret", "size"), win=("ret", lambda x: (x > 0).mean()), acct=("acct", "sum"))
    say("  by coin (n, win, sum acct pts): " + "; ".join(f"{c[:-4]} {int(x.n)} {x.win*100:.0f}% {x.acct*100:+.0f}"
                                                         for c, x in by.sort_values("acct").iterrows()))
    bt = g.groupby("vtier", observed=True).agg(n=("ret", "size"), win=("ret", lambda x: (x > 0).mean()),
                                               avg=("ret", "mean"), acct=("acct", "sum"))
    say("  by vol tier at entry: " + "; ".join(f"{c} n {int(x.n)} win {x.win*100:.0f}% avg {x.avg*100:+.2f}% acct {x.acct*100:+.0f}"
                                               for c, x in bt.iterrows()))
    # time in market
    ia = int(np.searchsorted(idx, a)); ib = len(idx) if b is None else int(np.searchsorted(idx, b))
    held = np.zeros(len(idx), bool)
    for x0, x1 in zip(lg.eb.astype(int), lg.xb.astype(int)):
        held[x0:x1 + 1] = True
    say(f"  share of hours holding a coin: {held[ia:ib].mean()*100:.0f}%")
# monthly
mr = e.resample("ME").last().pct_change()
say("\nmonthly 2025-26: " + " ".join(f"{d.strftime('%y%m')} {v*100:+.0f}" for d, v in mr[mr.index >= T.T25].items()))
out.close()
