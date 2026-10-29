"""Round 3, task "timeofday": motivation part 2, 2021-23 ONLY (panel cut at 2024-01-01 before anything runs).

Part 1 (r3_timeofday_motiv.py) showed the weak entries are those placed in the Asian-morning block UTC 00-07 and on
weekends (mostly the weekend's 00-07 hours), while the thin late-US hours UTC 21-23 are among the best. This pass
prints the combined groups the hypotheses would use, by year, and the standalone (each coin alone, no cash/N limit)
effect of skipping (smult 0) or deepening the trigger inside each window: total sum of trade returns, hours held,
and return per hour held over ALL trades (a skipped dip can still be bought at market when the window ends if the
coin is still under its 12% trigger). No 2024-26 number is computed or printed.
"""
import numpy as np
import pandas as pd

import engine as E
import study as S

pd.set_option("display.width", 220)
syms, idx_all, P_all = S.market(False)
keep = idx_all < E.DESIGN_END
idx = idx_all[keep]
P = {f: v[keep] for f, v in P_all.items()}
assert idx[-1] < E.DESIGN_END
n, K = P["o"].shape
hrs = idx.hour.values; wk = idx.dayofweek.values >= 5
WIN = {"UTC 00-07": hrs < 8, "UTC 03-06": (hrs >= 3) & (hrs <= 6), "Sat-Sun": wk, "Sat-Sun 00-07": wk & (hrs < 8)}


def summ(x):
    return (f"n {len(x):4d} win {((x.ret > 0).mean() * 100):5.1f}% avg {x.ret.mean() * 100:+.2f}% "
            f"TS {(x.why == 1).mean() * 100:4.1f}% sum ret x size {(x.ret * x.amt).sum():+.2f} hold {(x.xb - x.eb).mean():4.1f}h")


e, lg = E.run(P, idx)
lg = lg[lg.xb.astype(int) < n - 1].copy()
eb = lg.eb.astype(int).values
lg["year"] = idx[eb].year
print(f"live 2021-23 x{e.iloc[-1]:.2f}, {len(lg)} trades closed before 2024")
for w, m in WIN.items():
    inm = m[eb]
    print(f"\n{w}: share of hours {m.mean() * 100:.1f}%")
    print(f"  inside  all years: {summ(lg[inm])}")
    print(f"  outside all years: {summ(lg[~inm])}")
    for y in (2021, 2022, 2023):
        yy = lg.year.values == y
        print(f"  {y} inside {summ(lg[inm & yy])} | outside {summ(lg[~inm & yy])}")


def standalone(**kw):
    out = []
    for j, s in enumerate(syms):
        Pj = {f: v[[s]] for f, v in P.items()}
        _, lj = E.run(Pj, idx, **kw)
        lj = lj[lj.xb.astype(int) < n - 1]
        out.append(lj)
    return pd.concat(out, ignore_index=True)


print("\nStandalone (each coin alone), ALL trades: sum of returns, hours held, return per hour held")
a0 = standalone()
h0 = (a0.xb - a0.eb).sum()
print(f"  live            : n {len(a0):4d} sum {a0.ret.sum():+.2f} hours {h0:6.0f} bp/h {a0.ret.sum() / h0 * 1e4:5.2f} "
      f"TS {int((a0.why == 1).sum())}")
for w, m in WIN.items():
    for lab, kw in (("skip", dict(smult=np.where(m, 0.0, 1.0))), ("depth 14%", dict(depth=np.where(m, 0.14, 0.12))),
                    ("depth 16%", dict(depth=np.where(m, 0.16, 0.12)))):
        a = standalone(**kw)
        hh = (a.xb - a.eb).sum()
        print(f"  {w:13s} {lab:9s}: n {len(a):4d} sum {a.ret.sum():+.2f} hours {hh:6.0f} bp/h {a.ret.sum() / hh * 1e4:5.2f} "
              f"TS {int((a.why == 1).sum())}")
