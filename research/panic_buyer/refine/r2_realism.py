"""Round 2, lens "realism" (1-minute candles). NOT a strategy change: no grid, no pick. It measures how much the
backtest of the live settings depends on optimistic fills.

(a) Anatomy of every live-settings limit-buy fill (1-minute engine, take-profit at the next hourly cycle): the fill
    minute, how far through the limit the price traded, and how long it stayed at/below the limit.
(b) Stricter buy fills: must trade 0.3% / 0.5% through (live 0.1%), or stay at/below the limit for >= 2 / >= 5
    consecutive minutes ("at/below" read two ways: the minute's low, or the stricter minute close).
(c) Stricter take-profit fills: must trade 0.2% through (live 0.1%).
(d) Market buys at the hour open: the open vs the next few minutes, and re-runs filling them later / worse.
Extra realism rows, also fixed here before any run:
(e) 1-minute latency on every hourly action (the bot acts ~10-30 s after the hour; one minute is a pessimistic bound).
(f) live-bot timing for market buys: panic_buyer.py places the take-profit right after a market buy (the engine
    waits for the next hour) and counts 48h from the cycle, so its time stop comes one hour earlier.
(g) realistic-strict: 0.3% through + take-profit 0.2% through + 1-minute latency;  (g+f) the same with (f) timing.
(h) stress bound: 0.5% through AND 5 consecutive minutes + take-profit 0.2% + latency + market buys at the
    first minute's high.
Plausible live range, defined before any run: low end = the worse of (g) and (g+f); high end = the better of
the live backtest and (f) (worse/better = lower/higher 2024-26 multiple; both rows are printed in full anyway).
(h) and "every fill 0.2% worse" are stress references, not the range.

Runs once, one process, the 1-minute panel loaded once. PB_DATA / PB_OUT as usual.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import engine as E
import study as S
import r2_realism_lib as R

V = {
    "ref: every fill 0.2% worse": dict(pen=0.002),
    "(b) buy must trade 0.3% through": dict(beps=0.003),
    "(b) buy must trade 0.5% through": dict(beps=0.005),
    "(b) buy: low<=limit 2 consecutive min": dict(bmin=2, bmode=0),
    "(b) buy: low<=limit 5 consecutive min": dict(bmin=5, bmode=0),
    "(b) buy: close<=limit 2 consecutive min": dict(bmin=2, bmode=1),
    "(b) buy: close<=limit 5 consecutive min": dict(bmin=5, bmode=1),
    "(c) take-profit must trade 0.2% through": dict(seps=0.002),
    "(d) market buys at 1st-minute close": dict(mpx=1),
    "(d) market buys at 1st-minute high": dict(mpx=2),
    "(e) 1-minute latency on all hourly actions": dict(lat=1, mpx=1),
    "(f) live-bot timing for market buys": dict(mdelay=1, mtsh=1),
    "(g) realistic-strict 0.3%+tp0.2%+latency": dict(beps=0.003, seps=0.002, lat=1, mpx=1),
    "(g+f) realistic-strict + live mkt timing": dict(beps=0.003, seps=0.002, lat=1, mpx=1, mdelay=1, mtsh=1),
    "(h) stress 0.5%&5min+tp0.2%+lat+1st-min hi": dict(beps=0.005, bmin=5, bmode=0, seps=0.002, lat=1, mpx=2),
}
STATIC = {"through >= 0.3%": dict(beps=0.003), "through >= 0.5%": dict(beps=0.005),
          "low<=limit >= 2 consecutive min": dict(bmin=2, bmode=0), "low<=limit >= 5 consecutive min": dict(bmin=5, bmode=0),
          "close<=limit >= 2 consecutive min": dict(bmin=2, bmode=1), "close<=limit >= 5 consecutive min": dict(bmin=5, bmode=1)}

OUT = S.OUT
LOGF = open(os.path.join(OUT, "r2_realism.log"), "w")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s); LOGF.write(s + "\n"); LOGF.flush()


def first_fill(l, c, lim, beps=E.EPS, bmin=0, bmode=0, start=0):
    """Python twin of rkernel's resting-buy rule on one hour of minutes: offset of the fill minute, or -1."""
    run = 0; thr = False
    for i in range(start, 60):
        if l[i] <= lim * (1.0 - beps):
            thr = True
        if bmin > 1:
            x = l[i] if bmode == 0 else c[i]
            run = run + 1 if x <= lim else 0
            if thr and run >= bmin:
                return i
        elif thr:
            return i
    return -1


def maxrun(b):
    best = cur = 0
    for v in b:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def period(lg, hidx):
    return np.where(hidx[lg.eb.astype(int).values] < E.DESIGN_END, "2021-23", "2024-26")


def pct(x, q):
    return float(np.nanpercentile(x, q)) * 100


def anatomy(lg, A, hidx):
    """(a) one row per resting limit-buy fill of the live run."""
    Lo, Cl = A["l"], A["c"]
    rows = []
    for r in lg[lg.mkt == 0].itertuples():
        m0 = 60 * int(r.eb); k = int(r.k); lim = float(r.lim); fm = int(r.fm); off = fm - m0
        l = Lo[m0:m0 + 60, k]; c = Cl[m0:m0 + 60, k]
        a = off
        while a > 0 and l[a - 1] <= lim:
            a -= 1
        b = off
        while b < 59 and l[b + 1] <= lim:
            b += 1
        lf = Lo[fm:fm + 240, k]
        nz = np.nonzero(~(lf <= lim))[0]
        row = dict(eb=int(r.eb), k=k, ret=r.ret, amt=r.amt, why=r.why, off=off,
                   pen_fill=1 - l[off] / lim, pen_hour=1 - np.nanmin(l[off:]) / lim,
                   touch_run=b - a + 1, touch_min=int((l <= lim).sum()),
                   close_run=maxrun(c <= lim), close_min=int((c <= lim).sum()),
                   fwd_min=int(nz[0]) if len(nz) else 240)
        for name, kw in STATIC.items():
            row["ok " + name] = first_fill(l, c, lim, **kw) >= 0
        assert first_fill(l, c, lim) == off, "live fill minute not reproduced by the Python twin"
        rows.append(row)
    F = pd.DataFrame(rows)
    F["period"] = np.where(hidx[F.eb.values] < E.DESIGN_END, "2021-23", "2024-26")
    return F


def show_anatomy(F):
    say("\n######## (a) anatomy of the live limit-buy fills (1-minute engine, live settings)")
    for p, G in F.groupby("period"):
        n = len(G)
        say(f"\n{p}: {n} resting limit-buy fills (sum of return x order size {float((G.ret * G.amt).sum()):+.2f}, "
            f"avg return {G.ret.mean()*100:+.2f}%)")
        say(f"  fill minute within the hour: median {G.off.median():.0f}, in minute 0: {int((G.off == 0).sum())} "
            f"({(G.off == 0).mean()*100:.1f}%), minutes 0-4: {(G.off <= 4).mean()*100:.1f}%")
        say(f"  how far through the limit (lowest low from the fill to the hour end): "
            f"p10 {pct(G.pen_hour,10):.2f}% p25 {pct(G.pen_hour,25):.2f}% median {pct(G.pen_hour,50):.2f}% "
            f"p75 {pct(G.pen_hour,75):.2f}% p90 {pct(G.pen_hour,90):.2f}%")
        say("    share of fills that traded less than X through: " + "  ".join(
            f"<{x:.1f}%: {(G.pen_hour < x / 100).mean()*100:.1f}%" for x in (0.2, 0.3, 0.5, 1.0, 2.0)))
        say(f"    in the fill minute itself: median {pct(G.pen_fill,50):.2f}%, <0.3%: {(G.pen_fill < 0.003).mean()*100:.1f}%")
        for col, lab in (("touch_run", "consecutive minutes with low <= limit (run holding the fill minute)"),
                         ("close_run", "longest run of minute closes <= limit in the hour"),
                         ("touch_min", "minutes in the hour with low <= limit"),
                         ("close_min", "minutes in the hour with close <= limit"),
                         ("fwd_min", "minutes from the fill until a minute's low is back above the limit (cap 240)")):
            x = G[col]
            say(f"  {lab}: p10 {x.quantile(.1):.0f} p25 {x.quantile(.25):.0f} median {x.median():.0f} "
                f"p75 {x.quantile(.75):.0f} | <2: {(x < 2).mean()*100:.1f}%  <5: {(x < 5).mean()*100:.1f}%")
        say("  fills a stricter rule would NOT have taken (same hour, live path; static count):")
        for name in STATIC:
            bad = G[~G["ok " + name]]
            good = G[G["ok " + name]]
            say(f"    {name:36s} drops {len(bad):4d} ({len(bad)/n*100:4.1f}%): avg return {bad.ret.mean()*100 if len(bad) else 0:+.2f}%, "
                f"sum ret x size {float((bad.ret * bad.amt).sum()):+.2f}, time stops {int((bad.why == 1).sum())} "
                f"| kept avg {good.ret.mean()*100:+.2f}%")
        m0f = G[G.off == 0]
        say(f"    fills in minute 0 (order may not be up yet): {len(m0f)}, avg return {m0f.ret.mean()*100 if len(m0f) else 0:+.2f}%, "
            f"sum ret x size {float((m0f.ret * m0f.amt).sum()):+.2f}")
        say("  avg return by how far through the limit it traded:")
        bins = [0, 0.002, 0.003, 0.005, 0.01, 0.02, 0.05, 1]
        cut = pd.cut(G.pen_hour, bins, right=False)
        for iv, H in G.groupby(cut, observed=True):
            say(f"    {iv.left*100:4.1f}-{iv.right*100:5.1f}%: {len(H):4d} fills, avg {H.ret.mean()*100:+.2f}%, "
                f"time stops {int((H.why == 1).sum()):3d}, sum ret x size {float((H.ret * H.amt).sum()):+.2f}")


def show_market(lg, A, hidx):
    """(d) market buys at the hour open, and take-profits that sold at an open, vs the next minutes."""
    O, H, Lo, Cl = A["o"], A["h"], A["l"], A["c"]
    say("\n######## (d) market buys at the hour open vs the next minutes (live run)")
    M = lg[lg.mkt == 1].copy()
    m0 = 60 * M.eb.astype(int).values; k = M.k.astype(int).values
    o = O[m0, k]
    cols = {f"close of minute {j}": Cl[m0 + j, k] / o - 1 for j in range(5)}
    cols["high of minute 0"] = H[m0, k] / o - 1
    cols["highest high, minutes 0-4"] = np.max(np.stack([H[m0 + j, k] for j in range(5)]), axis=0) / o - 1
    cols["lowest low, minutes 0-4"] = np.min(np.stack([Lo[m0 + j, k] for j in range(5)]), axis=0) / o - 1
    M["under"] = o / M.lim.values - 1
    for c_, v in cols.items():
        M[c_] = v
    M["period"] = period(M, hidx)
    for p, G in M.groupby("period"):
        say(f"\n{p}: {len(G)} market buys (open already under the trigger; median {pct(G.under,50):.2f}% under it, "
            f"p10 {pct(G.under,10):.2f}%), avg trade return {G.ret.mean()*100:+.2f}%, sum ret x size {float((G.ret*G.amt).sum()):+.2f}")
        say("  price vs the open (+ = later price higher = the open was a better buy):")
        for c_ in cols:
            x = G[c_]
            say(f"    {c_:28s} mean {x.mean()*100:+.3f}% median {x.median()*100:+.3f}% p10 {pct(x,10):+.3f}% "
                f"p90 {pct(x,90):+.3f}% | higher than the open {(x > 0).mean()*100:5.1f}%")
    say("\n  take-profits that sold at an open already above target (sell side: - = later price lower = the open was better)")
    T = lg[(lg.why == 0) & (lg.xo == 1)].copy()
    xm = T.xm.astype(int).values; k = T.k.astype(int).values; ox = O[xm, k]
    allt = lg[lg.why == 0]
    T["first"] = xm == 60 * (T.eb.astype(int).values + 1)
    T["period"] = period(T, hidx)
    allt = allt.assign(period=period(allt, hidx))
    for j in range(3):
        T[f"c{j}"] = Cl[xm + j, k] / ox - 1
    T["l0"] = Lo[xm, k] / ox - 1
    for p, G in T.groupby("period"):
        A_ = allt[allt.period == p]
        say(f"  {p}: {len(G)} of {len(A_)} take-profits sold at an open ({len(G)/len(A_)*100:.0f}%; {int(G['first'].sum())} at the "
            f"first cycle), avg {G.ret.mean()*100:+.2f}% vs {A_[A_.xo == 0].ret.mean()*100:+.2f}% for the resting fills")
        for c_, lab in (("c0", "close of minute 0"), ("c1", "close of minute 1"), ("c2", "close of minute 2"),
                        ("l0", "low of minute 0")):
            x = G[c_]
            say(f"    {lab:20s} mean {x.mean()*100:+.3f}% median {x.median()*100:+.3f}% | lower than the open {(x < 0).mean()*100:5.1f}%")


def counts(lg, hidx):
    p = period(lg, hidx)
    out = {}
    for tag in ("2021-23", "2024-26"):
        g = lg[p == tag]
        out[tag] = dict(n=len(g), mkt=int((g.mkt == 1).sum()), lim=int((g.mkt != 1).sum()), tp=int((g.why == 0).sum()),
                        ts=int((g.why == 1).sum()), avg=float(g.ret.mean()))
    return out


def verify_fills(name, kw, lg, A):
    """Kernel logic check: every resting-limit fill of a variant is the first minute its rule allows, and every
    market buy is at the price its rule names."""
    Lo, Cl, O, H = A["l"], A["c"], A["o"], A["h"]
    lat = kw.get("lat", 0); bad = 0
    rule = {x: kw[x] for x in ("beps", "bmin", "bmode") if x in kw}
    for r in lg[lg.mkt == 0].itertuples():
        m0 = 60 * int(r.eb); k = int(r.k)
        if first_fill(Lo[m0:m0 + 60, k], Cl[m0:m0 + 60, k], float(r.lim), start=lat, **rule) != int(r.fm) - m0:
            bad += 1
    M = lg[lg.mkt == 1]
    m0 = 60 * M.eb.astype(int).values; k = M.k.astype(int).values
    px = {0: O, 1: Cl, 2: H}[kw.get("mpx", 0)][m0, k] * (1 + E.SLIP_MAJOR) * (1 + kw.get("pen", 0.0))
    bad_m = int((~np.isclose(M.entry.values, px, rtol=1e-12, atol=0)).sum())
    if bad or bad_m:
        say(f"  !! {name}: {bad} limit fills / {bad_m} market fills do not follow the rule")
    return bad + bad_m


def main():
    t0 = time.time()
    say("######## lens realism: 19 live coins, 1-minute candles")
    hourly_archive, _, _ = S.evaluate({})
    say(E.line("live, hourly archive (engine.run)", hourly_archive))
    idx, A, hidx, Ph = E.minute_panel(E.H19)
    say(f"1-minute panel {idx[0]} -> {idx[-1]}, {A['c'].shape}, loaded in {time.time()-t0:.0f}s")
    eh, lh = E.run(Ph, hidx)
    say(E.line("live, hourly built from 1m (engine.run)", E.score(eh, lh, hidx)))
    em, lm = E.mrun(A, hidx, Ph, 0)
    base = E.score(em, lm, hidx)
    say(E.line("live, 1-minute engine.mrun(delay=0)", base))
    eb_, lb = R.rrun(A, hidx, Ph)
    rb = E.score(eb_, lb, hidx)
    say(E.line("live, r2_realism_lib.rrun (switches off)", rb))
    d_eq = float(np.max(np.abs(eb_.values - em.values)))
    same_log = len(lb) == len(lm) and np.array_equal(lb[list(lm.columns)].values, lm.values)
    say(f"EQUIVALENCE rrun(switches off) vs engine.mrun(delay=0): max |equity diff| {d_eq:.3g}, trade logs identical: {same_log}")
    if d_eq != 0.0 or not same_log:
        say("EQUIVALENCE FAILED"); sys.exit(1)
    say(f"  (and engine.mrun vs engine.run on hourly candles built from the same minutes: max |equity diff| "
        f"{float(np.max(np.abs(em.values - eh.values))):.3g})")

    F = anatomy(lb, A, hidx)
    F.to_csv(os.path.join(OUT, "r2_realism_fills.csv"), index=False)
    show_anatomy(F)
    show_market(lb, A, hidx)

    say("\n######## (b)-(h) re-runs of the live strategy with stricter / later fills (1-minute engine)")
    cb = counts(lb, hidx)
    hdr = lambda c: (f" | trades {c['2021-23']['n']}/{c['2024-26']['n']} mkt {c['2021-23']['mkt']}/{c['2024-26']['mkt']} "
                     f"ts {c['2021-23']['ts']}/{c['2024-26']['ts']} avg {c['2021-23']['avg']*100:+.2f}/{c['2024-26']['avg']*100:+.2f}%")
    say(E.line("LIVE (1-minute engine)", rb) + hdr(cb))
    rows = [dict(name="LIVE (1-minute engine)", **rb, **{f"{p}_{k}": v for p, d in cb.items() for k, v in d.items()})]
    nbad = 0
    for name, kw in V.items():
        e, lg = R.rrun(A, hidx, Ph, **kw)
        r = E.score(e, lg, hidx); c = counts(lg, hidx)
        nbad += verify_fills(name, kw, lg, A)
        r.update(name=name, beats_live=E.beats(r, rb), **{f"{p}_{k}": v for p, d in c.items() for k, v in d.items()})
        rows.append(r)
        say(E.line(name, r) + hdr(c))
    T = pd.DataFrame(rows)
    T.to_csv(os.path.join(OUT, "r2_realism_runs.csv"), index=False)
    say(f"kernel rule check: {'all fills follow their rule' if nbad == 0 else f'{nbad} fills break their rule'}")

    say("\n######## change vs live (1-minute engine)")
    for r in rows[1:]:
        say(f"  {r['name']:46s} 21-23 x{r['d_mult']:5.2f} ({(r['d_mult']/rb['d_mult']-1)*100:+5.1f}%) DD {r['d_dd']*100:4.0f}% "
            f"| 24-26 x{r['t_mult']:5.2f} ({(r['t_mult']/rb['t_mult']-1)*100:+5.1f}%) DD {r['t_dd']*100:4.0f}% "
            f"| CAGR {r['d_cagr']*100:+4.0f}% / {r['t_cagr']*100:+4.0f}% (live {rb['d_cagr']*100:+.0f}% / {rb['t_cagr']*100:+.0f}%)")
    by = {r["name"]: r for r in rows}
    lo = min((by["(g) realistic-strict 0.3%+tp0.2%+latency"], by["(g+f) realistic-strict + live mkt timing"]),
             key=lambda r: r["t_mult"])
    hi = max((by["LIVE (1-minute engine)"], by["(f) live-bot timing for market buys"]), key=lambda r: r["t_mult"])
    say("\n######## plausible live range (rule fixed before the run)")
    for tag, r in (("low end", lo), ("high end", hi)):
        say(f"  {tag:8s} {r['name']:44s} 2021-23 x{r['d_mult']:.2f} DD {r['d_dd']*100:.0f}% 2022 {r['y2022']*100:+.0f}% | "
            f"2024-26 x{r['t_mult']:.2f} DD {r['t_dd']*100:.0f}%")
    say(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
