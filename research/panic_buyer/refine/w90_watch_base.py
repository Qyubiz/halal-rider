"""w90 ten-watch, step 1: 10% mode (1 coin x 100%, depth 14%, +5%, 48h, 24h cooldown, guard 35%) from the hourly
backtest to a realistic minute-level watch bot, one assumption at a time. 19 coins unless noted; fees 0.1%/side.

Rows:
  H  hourly engine (engine.run, hourly candles from the 1h files)            = the published 10% mode numbers
  U  hourly watch (x10_universe_lib.urun on hourly candles built from minutes) = the universe agent's watch mode
  M* minute candles, x10_critic_lib.xkernel: live resting-bid mechanism; idealised watch (fills AT the trigger)
  W* this task's watch bot (w90_watch_lib mode 1): market buy at the next minute's open + 0.05%, take-profit placed at
     once and filled at the limit only when a minute's high trades 0.1% through, wicks trimmed to 1% beyond the body.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import w90_watch_lib as W  # noqa: E402
import x10_critic_lib as X  # noqa: E402
import x10_universe_lib as U  # noqa: E402
import x10_velocity_lib as V  # noqa: E402

T0 = time.time()
TEN = dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24)
rows = []


def show(name, r):
    r = dict(r); r["name"] = name
    rows.append(r)
    print(f"{name:58s} {W.line(r)}", flush=True)


# ---- H: hourly engine on the hourly files (published 10% mode)
syms, idx, P = E.panel(E.load_h19())
c = P["cf"]
ok = ((c / c.rolling(168, min_periods=1).max() - 1) >= -0.35).shift(1, fill_value=True).values.astype(float)
for lab, kw in (("H  hourly engine, 10% mode", {}), ("H  hourly engine, bot timing (mtp_now, cd 25)", dict(mtp_now=True, cd=25))):
    e, lg = E.run(P, idx, smult=ok, **{**TEN, **kw})
    lg = lg.copy(); lg["eqb"] = e.shift(1).fillna(1.0).values[lg.eb.astype(int).values]
    show(lab, W.report(e, lg, idx, syms=syms))

ctx = W.Ctx()
A, hidx, Ph, SLIP = ctx.A, ctx.hidx, ctx.Ph, ctx.SLIP

# ---- U: hourly watch mode on hourly candles built from the minutes (universe kernel)
K = 19
ref = X.guard_ref(Ph, 0.35, K)
sm = (~np.isnan(ref)).astype(float)
Pm = {f: v.iloc[:, :K] for f, v in Ph.items()}
ue, ul = U.urun(Pm, hidx, SLIP[:, :K], smult=sm, watch=True, **TEN)
show("U  hourly watch (universe kernel, best-case in-hour order)", W.report(ue, ul.rename(columns={"eq0": "eqb"}), hidx))

# ---- M: minute candles, critic kernel
for w in (None, 0.01):
    B = ctx.B(K, w)
    wl = "orig wicks" if w is None else "wicks 1%"
    for lab, kw in (("live mechanism: 1 hourly resting bid", dict(cyc=60)),
                    ("ideal watch, fill AT trigger, hourly re-check", dict(watch=1, cyc=60)),
                    ("ideal watch, fill AT trigger, every minute", dict(watch=1, cyc=1, rearm=1))):
        e, lg, _ = X.xrun(B, hidx, Ph, SLIP, K=K, ref=ref, **TEN, **kw)
        show(f"M  {lab}, {wl}", W.report(e, lg, hidx))

# ---- W: the realistic watch bot
VAR = [("W  low crosses -> next open +0.05%, ORIGINAL wicks", dict(wick=None)),
       ("W  low crosses -> next open +0.05%, wicks 1%  [MAIN]", dict()),
       ("W  same, wicks 0.5%", dict(wick=0.005)),
       ("W  close under trigger -> next open (1/min poll)", dict(det=1)),
       ("W  low 0.1% through trigger -> next open", dict(deps=0.001)),
       ("W  same-minute coins in coin order (not deepest)", dict(ordm=1)),
       ("W  take-profit active in the fill minute too", dict(tpd=0)),
       ("W  take-profit active 1 hour after the fill", dict(tpd=60)),
       ("W  take-profit gap fills at the open", dict(gap=1)),
       ("W  time stop to the minute (48h after the fill)", dict(tsm=1)),
       ("W  market slippage 0.1%", dict(mslip=0.001)),
       ("W  cooldown hour-aligned like the engine", dict(cdmin=-1)),
       ("W  fills 0.2% worse (buys and sells)", dict(pen=0.002, spen=0.002)),
       ("W  cooldown +1h", dict(cdx=60))]
for lab, kw in VAR:
    show(lab, ctx.rep(19, **{**TEN, **kw}))
s = dict(TEN); s.update(W.STRESS)
show("W  MAIN + LUNA/FTT/SRM (22 coins)", ctx.rep(22, **TEN))
show("W  MAIN STRESSED: 22 coins, fills 0.2% worse, cd +1h", ctx.rep(22, **s))
show("W  MAIN, no crash guard", ctx.rep(19, **{**TEN, "guard": 0.0}))
show("W  MAIN, no crash guard, 22 coins", ctx.rep(22, **{**TEN, "guard": 0.0}))

R = pd.DataFrame(rows)
R.to_csv(os.path.join(W.OUT, "base.csv"), index=False)
print(f"done in {time.time()-T0:.0f}s")
