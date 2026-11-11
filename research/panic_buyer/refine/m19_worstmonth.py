"""m19: can the bot's existing risk tools cap the worst month of 90% mode?  stop-loss (hourly close) and the
30-day drop kill-switch (sell all, pause 7 days), plus smaller bets. Per period: compounded %/month, worst month,
losing months, worst drop, win rate. Hindsight comparison table (all periods shown), not a selection."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m10_velocity_lib as V, m11_fusion_lib as F, m14_judge as J

def row(name, d):
    out = [name]
    for hold, a, b, tag in ((False, V.T21, V.T24, "d"), (False, J.T24, None, "t"), (True, J.H18, J.H21, "h")):
        syms, idx, P = V.market_holdout(False) if hold else V.market(False)
        e, lg, inv = F.frun(P, idx, d)
        r = V.summarize(e, lg, inv, idx, V.HOLD if hold else V.MAIN)
        m = J.monthly(e, a, b) * 100
        out.append(f"{r[f'{tag}_mo']*100:+.1f} / {m.min():+.0f} / {(m<0).sum()} of {len(m)} / {r[f'{tag}_dd']*100:.0f} / {r[f'{tag}_wr']*100:.0f}")
    print("| " + " | ".join(out) + " |", flush=True)

print("%/month / worst month / losing months / worst drop / win rate\n")
print("| setup | 2021-23 | 2024-26 | 2018-20 |\n|---|---|---|---|")
B = F.MODE90
row("90% mode", B)
for hs in (0.05, 0.08, 0.12):
    row(f"+ stop-loss {hs*100:.0f}%", dict(B, hs=hs))
for km in (0.08, 0.12):
    row(f"+ pause 7 days after a {km*100:.0f}% drop in 30 days", dict(B, killm=km))
row("+ stop-loss 8% and pause after 12% drop", dict(B, hs=0.08, killm=0.12))
for fs in (0.5, 0.33):
    t = dict(F.TIERA, fs=fs)
    row(f"bet {fs*100:.0f}% of the money per trade (rest in USDT)", dict(B, tiers=[t], nslots=1))
