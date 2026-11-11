"""Checks for m10_portfolio_lib.pkernel.
1. All money in the DIP sleeve (90% mode) == ens_lib.erun([MODE90], guard 0.35) exactly (equity and trades), on the
   2021-26 panel and the 2018-20 holdout panel, with normal fills and with buys+sells 0.2% worse.
2. Two DIP sleeves (90% + 10% mode, 30/70, rb=1) == ens_lib.erun(..., rb=1) (the free-cash rebalance logic).
3. MOM-only and TREND-only runs: no negative cash, the sleeve equity adds up to the account, every MOM trade's
   return matches its event prices (fees, slippage) and at most `slots` MOM trades overlap.
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 m10_portfolio_check.py
"""
import numpy as np
import pandas as pd

import ens_lib as L
import m10_portfolio_lib as M

ok_all = True
for hold in (False, True):
    syms, idx, P = M.market(hold)
    for tag, kw in (("normal", {}), ("stress", M.STRESS)):
        e1, lg1, _, _ = L.erun(P, idx, [L.MODE90], guard=0.35, **kw)
        e2, sq, lg2 = M.prun(hold, [1, 0, 0, 0], **kw)
        d = float(np.max(np.abs(e1.values - e2.values)))
        nt1, nt2 = int((lg1.why != 9).sum()), int((lg2.why != 9).sum())
        ok = d < 1e-9 and nt1 == nt2
        ok_all &= ok
        print(f"{'holdout' if hold else '2021-26'} {tag:6s} 90% mode: max |equity diff| {d:.2e}, trades {nt1} vs {nt2} "
              f"-> {'OK' if ok else 'MISMATCH'}")
    e1, _, _, _ = L.erun(P, idx, [L.MODE90, L.MODE10], weights=[0.3, 0.7], rb=1, guard=0.35)
    e2, _, _ = M.prun(hold, [0.3, 0.7, 0, 0, 0], dips=(L.MODE90, L.MODE10), rb=1)
    d = float(np.max(np.abs(e1.values - e2.values)))
    ok_all &= d < 1e-9
    print(f"{'holdout' if hold else '2021-26'} 30/70 sleeves rb=1: max |equity diff| {d:.2e} -> {'OK' if d < 1e-9 else 'MISMATCH'}")

    for name, w in (("MOM only", [0, 1, 0, 0]), ("TREND only", [0, 0, 1, 0]), ("mix 40/30/20/10", [0.4, 0.3, 0.2, 0.1])):
        e, sq, lg = M.prun(hold, w)
        add = float(np.max(np.abs(sq.sum(axis=1).values - e.values)))
        neg = float(sq.loc[:, np.array(w) > 0].min().min())   # sleeves that hold money
        msg = f"{'holdout' if hold else '2021-26'} {name:16s}: sleeves add up (max diff {add:.1e}), min sleeve equity {neg:.4f}"
        ok = add < 1e-9 and neg > 0
        if w[1] > 0:
            ev = M.mom_events(hold)
            m = lg[lg.sl == 1]
            # overlap: at most 5 MOM trades open at any hour
            occ = np.zeros(len(idx) + 1)
            for _, r in m.iterrows():
                occ[int(r.eb)] += 1; occ[int(r.xb)] -= 1
            mx = int(np.cumsum(occ).max())
            # every logged MOM trade return = event prices with costs
            fin = m[m.why != 9]
            msg += f", MOM trades {len(m)}, max open at once {mx}"
            ok &= mx <= 5
            got = set(np.round(fin.ret.values, 10))
            exp = set(np.round((1 - 0.001) ** 2 * ev[:, 5] / ev[:, 3] - 1, 10)[ev[:, 2] >= 0])
            miss = len(got - exp)
            msg += f", returns not matching an event {miss}"
            ok &= miss == 0
        ok_all &= ok
        print(msg + (" OK" if ok else " FAIL"))
print("ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")

# 4. MOM events re-derived independently (pandas resample + timestamps) for one month of every coin and 60 random
#    events: same signals, same fills, same exits.
import m10_portfolio_lib as M2  # noqa: E402,F811


def slow_events(sym, k, hold, t_from, t_to, prm=M2.MOM_DEF):
    syms, idx, P = M2.market(hold)
    t0 = int(idx[0].value // 10**9); nmin = len(idx) * 60
    o1, h1, l1, c1 = M2.grid_1m(sym, hold, t0, nmin)
    tix = idx[0] + pd.to_timedelta(np.arange(nmin), unit="min")
    m = pd.DataFrame({"o": o1, "h": h1, "l": l1, "c": c1}, index=tix)
    b = pd.Series(M2.grid_1m("BTCUSDT", hold, t0, nmin)[3], index=tix)
    c5 = m.resample("5min").agg({"o": "first", "h": "max", "l": "min", "c": "last"})
    ema = M2.daily_close("BTCUSDT").ewm(span=50, adjust=False).mean()
    rng = (c5.h - c5.l) / c5.c
    avg = rng.shift(1).rolling(prm["vlook"]).mean()
    out = []
    for ts in c5.index[(c5.index >= t_from) & (c5.index < t_to)]:
        i = c5.index.get_loc(ts)
        if i < prm["vlook"]:
            continue
        row = c5.iloc[i]
        tc = ts + pd.Timedelta(minutes=5)                     # candle close = bot cycle
        e_ = ema.get(tc.floor("D") - pd.Timedelta(days=1), np.nan)
        if not (b.loc[tc - pd.Timedelta(minutes=1)] > e_):
            continue
        if not (row.c / c5.c.iloc[i - 3] - 1 >= prm["rise"] and row.c > row.o):
            continue
        if prm["vmode"] == 1 and not (rng.iloc[i] >= prm["vm"] * avg.iloc[i]):
            continue
        trig = row.c * 1.01; lim = trig * 1.003
        w = m.loc[tc: tc + pd.Timedelta(minutes=59)]
        fill = None; trg = False
        for tm, x in w.iterrows():
            if not trg:
                if x.o >= trig:
                    trg = True
                    if x.o <= lim:
                        fill = (tm, min(x.o * 1.0002, lim)); break
                    if x.l <= lim:
                        fill = (tm, lim); break
                elif x.h >= trig:
                    trg = True; fill = (tm, trig * 1.0002); break
            elif x.l <= lim:
                fill = (tm, lim); break
        c0 = int((tc - idx[0]) / pd.Timedelta(minutes=1))
        if fill is None:
            out.append((c0, k, -1, np.nan, np.nan, np.nan)); continue
        fm, fp = fill
        B = fm.floor("5min") + pd.Timedelta(minutes=5)
        stop = fp * 0.95; peak = fp; xm = None
        if m.o[B] <= stop:
            xm, xp = B, m.o[B] * 0.9998
        else:
            lastc = B - pd.Timedelta(minutes=5)              # last 5m candle already counted (none)
            tm = B
            while xm is None:
                if tm.minute % 5 == 0:
                    if tm - B >= pd.Timedelta(hours=24):
                        xm, xp = tm, m.o[tm] * 0.9998; break
                    cs = tm - pd.Timedelta(minutes=5)
                    if cs > lastc:
                        peak = max(peak, c5.h[cs]); lastc = cs
                    peak = max(peak, m.o[tm])
                    want = max(stop, peak * 0.92)
                    if want >= stop * 1.003:
                        if m.o[tm] <= want:
                            xm, xp = tm, m.o[tm] * 0.9998; break
                        stop = want
                if m.l[tm] <= stop:
                    xm, xp = tm, min(m.o[tm], stop) * 0.9998; break
                tm += pd.Timedelta(minutes=1)
        f = lambda x: int((x - idx[0]) / pd.Timedelta(minutes=1))  # noqa: E731
        out.append((c0, k, f(fm), fp, f(xm), xp))
    return out


bad = 0; tot = 0
for hold, (a_, b_) in ((False, ("2023-11-01", "2023-12-01")), (True, ("2019-06-01", "2019-07-01"))):
    syms, idx, P = M2.market(hold)
    ev = M2.mom_events(hold)
    for k, s in enumerate(syms):
        sl = slow_events(s, k, hold, pd.Timestamp(a_), pd.Timestamp(b_))
        lo = (pd.Timestamp(a_) - idx[0]) / pd.Timedelta(minutes=1) + 5
        hi = (pd.Timestamp(b_) - idx[0]) / pd.Timedelta(minutes=1) + 5
        fast = ev[(ev[:, 1] == k) & (ev[:, 0] >= lo) & (ev[:, 0] < hi)]
        if len(fast) != len(sl):
            bad += 1; print(f"  {s} {a_}: {len(fast)} fast vs {len(sl)} slow signals"); continue
        for x, y in zip(fast, sl):
            tot += 1
            same = x[0] == y[0] and x[2] == y[2] and (y[2] < 0 or (abs(x[3] - y[3]) < 1e-9 * y[3] and x[4] == y[4]
                                                                    and abs(x[5] - y[5]) < 1e-9 * y[5]))
            if not same:
                bad += 1; print("  mismatch", s, x[:6], y)
print(f"MOM events re-derived independently: {tot} events compared, {bad} mismatches -> {'OK' if bad == 0 else 'FAIL'}")
