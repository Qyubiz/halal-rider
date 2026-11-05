"""m10 exits, POST-HOC only (run after the one look in m10_exits_final.py; nothing here can become the pick):
which half of the pick failed out of sample -- the range-scaled take-profit alone, or the 72h underwater time stop alone.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 m10_exits_posthoc.py"""
import m10_exits_lib as X
import m10_exits_final as F

TPA = F.PICK["tpa"]
PARTS = {"90% mode": dict(X.BASE90), "pick": F.PICK,
         "range-scaled tp only (96h stop)": dict(X.BASE90, tpa=TPA),
         "72h underwater stop only (tp 2.5%)": dict(X.BASE90, hm=72, uw=0.12, uwmode=1, hm2=240, be2=0.01)}
mk = {"main": X.W.market(False), "hold": X.holdout_market()}
for name, kw in PARTS.items():
    r, _ = F.run(kw, mk["main"][2], mk["main"][1], X.PERIODS)
    h, _ = F.run(kw, mk["hold"][2], mk["hold"][1], X.HPERIOD)
    print(f"{name:36s} " + " | ".join(f"{lab} WR {d[f'{t}_wr']*100:4.1f}% {d[f'{t}_mo']*100:+5.2f}%/mo dd {d[f'{t}_dd']*100:4.0f}%"
                                     for lab, t, d in (("18-20", "h", h), ("21-23", "d", r), ("24-26", "t", r),
                                                       ("25-", "s", r))), flush=True)
