"""Task "stress": stress-test the aggressive Panic Buyer settings that reached 5%/month in 2024-26 in the 960-setting
grid (picked there WITH hindsight), plus the live settings for reference. A measurement only: nothing is picked or tuned.

Settings (fixed list from the task; depth 12%, cooldown 24h unless noted):
  A N=1 fmax 1.00 tp 3%   48h       B N=1 fmax 1.00 tp 1.5% 96h      C N=2 fmax 0.50 tp 3% 48h
  D N=2 fmax 0.50 tp 3%   96h       E N=2 fmax 0.50 tp 5%   48h      F depth 15% N=1 fmax 1.00 tp 5% 48h
  G N=3 fmax 0.33 tp 3%   48h       LIVE N=8 fmax 0.20 tp 1.5% 48h
Rows per setting:
  normal              engine.run, hourly candles, 19 coins (must reproduce grid960_19coins.csv)
  bot timing          engine.run mtp_now=True, cd=25 (what panic_buyer.py really does, REFINE.md)
  1-min ref           r2_realism_lib.rrun with every switch off (= engine.mkernel delay 0): the base for the next rows
  strict 1-min fills  rrun beps=0.3%, seps=0.2%, lat=1, mpx=1 = round-2 row (g): limit buys must trade 0.3% through,
                      the take-profit 0.2% through, every hourly action 1 minute late, market buys at minute 1's close
  strict + bot timing the same + mdelay=1, mtsh=1, cd=25
  fills 0.2% worse    engine.run pen=0.002
  +LUNA/FTT/SRM       engine.run on the 22-coin panel (collapse coins cut as in engine.DEAD)
  worst combined      strict fills + bot timing + LUNA/FTT/SRM (1-minute engine, 22 coins). The collapse coins' minutes
                      with no real trade (the FTT halt, LUNA after 2022-05-13 00:39, SRM after 2022-11-28 02:59) are
                      blanked so the 1-minute engine cannot trade them while Binance did not (minute_panel fills them
                      flat up to the cut); this matches the corrected hourly engine.
  frozen check        a run whose equity never changes for 30+ days at the end is reported with the coin whose data
                      ended then: a coin held when trading stops is never sold, so the account is stuck in it.
  jitter band         depth -0.1 / +0.1 point and start +1..7 days, each alone (r3_noise method: a 2021-23 run starts at
                      2021-01-01+d and is measured to 2024-01-01; a 2024-26 run starts fresh, all cash, at 2024-01-01+d).
                      Band = min..max over the 9 runs.
  extra band          2024-26 fresh starts on each of the first 30 days of 2024 (d = 0..29), a wider path check.
%/month = multiple ** (1/months) - 1 with months = 36 (2021-23) and 33.7 (2024-26), less d/30.4375 for a start shift.
Concentration (normal, +LUNA/FTT/SRM, worst combined): biggest single-trade losses as % of the equity the hour before
the buy (trade return x order size / equity), trades losing > 5% / > 10% of equity, worst UTC day and Mon-Sun week
(close to close of the hourly equity), worst calendar month, worst 6- and 12-month stretch (daily closes, windows
inside the period) and the share of 6-month windows that lost money, longest time under water.
Only information known at each hour's start is used (no new signals; the engines are unchanged).
"""
import os
import resource
import time

import numpy as np
import pandas as pd

import engine as E
import study as S
import r2_realism_lib as R
from lab import metrics

OUT = S.OUT
HERE = os.path.dirname(os.path.abspath(__file__))
T0, T1 = pd.Timestamp("2021-01-01"), E.DESIGN_END
MD = 365.25 / 12
MONTHS = {"d": 36.0, "t": 33.7}

SETS = {
    "A d12 N1 f1.00 tp3% 48h": dict(depth=0.12, N=1, fmax=1.0, tp=0.03, hm=48, cd=24),
    "B d12 N1 f1.00 tp1.5% 96h": dict(depth=0.12, N=1, fmax=1.0, tp=0.015, hm=96, cd=24),
    "C d12 N2 f0.50 tp3% 48h": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=24),
    "D d12 N2 f0.50 tp3% 96h": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=96, cd=24),
    "E d12 N2 f0.50 tp5% 48h": dict(depth=0.12, N=2, fmax=0.5, tp=0.05, hm=48, cd=24),
    "F d15 N1 f1.00 tp5% 48h": dict(depth=0.15, N=1, fmax=1.0, tp=0.05, hm=48, cd=24),
    "G d12 N3 f0.33 tp3% 48h": dict(depth=0.12, N=3, fmax=0.33, tp=0.03, hm=48, cd=24),
    "LIVE d12 N8 f0.20 tp1.5% 48h": dict(E.LIVE),
}
STRICT = dict(beps=0.003, seps=0.002, lat=1, mpx=1)
BOT_H = dict(mtp_now=True, cd=25)
BOT_M = dict(mdelay=1, mtsh=1, cd=25)
ROWS = ["normal", "bot timing", "1-min ref", "strict 1-min fills", "strict + bot timing", "fills 0.2% worse",
        "+LUNA/FTT/SRM", "worst combined"]

_log = None


def say(s=""):
    print(s, flush=True)
    _log.write(s + "\n"); _log.flush()


def pm(mult, months):
    return mult ** (1.0 / months) - 1.0 if mult > 0 else -1.0


def periods(e, shift_d=0, shift_t=0):
    """Multiple, worst drop and %/month for 2021-23 and 2024-26 (start shifts in days shorten the month count)."""
    out = dict(y2022=E.years(e).get(2022, np.nan))
    if e.index[0] < T1:
        d = metrics(e, end=T1)
        out.update(d_mult=d["mult"], d_dd=d["maxdd"], d_pm=pm(d["mult"], MONTHS["d"] - shift_d / MD), d_worst_m=d["worst_m"])
    t = metrics(e, start=T1)
    out.update(t_mult=t["mult"], t_dd=t["maxdd"], t_pm=pm(t["mult"], MONTHS["t"] - shift_t / MD), t_worst_m=t["worst_m"])
    return out


def ntrades(r, lg, idx):
    tin = idx[lg.eb.astype(int).values]
    r["d_trades"], r["t_trades"] = int((tin < T1).sum()), int((tin >= T1).sum())
    return r


def frozen(e, P, syms):
    """If the equity never changes in the last 30+ days of the run: when it froze, and which coins' data ended then
    (a coin held when its data ends is never sold and keeps its last price: the account is stuck in it)."""
    ch = np.flatnonzero(np.diff(e.values) != 0)
    tf = e.index[ch[-1] + 1] if len(ch) else e.index[0]
    if e.index[-1] - tf < pd.Timedelta(days=30):
        return ""
    near = [f"{s} data ends {P['c'][s].last_valid_index()}" for s in syms
            if abs(P["c"][s].last_valid_index() - tf) <= pd.Timedelta(hours=72)]
    return (f"      FROZEN from {tf} to the end at x{e.iloc[-1]:.4g} of the start: "
            + ("; ".join(near) if near else "no coin's data ends then"))


def run_from(P, idx, kw, i0):
    """Fresh run (all cash) from hour i0; features (ref) were computed on the full panel, so they keep their warm-up."""
    if i0:
        P = {f: v.iloc[i0:] for f, v in P.items()}
        idx = idx[i0:]
    return E.run(P, idx, **kw)


def concentration(e, lg, idx, syms):
    """Per period: top single-trade losses as % of equity, big-loss counts, worst day / week / month / 6m / 12m."""
    eqb = np.r_[1.0, e.values[:-1]]
    eb = lg.eb.astype(int).values
    L = lg.assign(size=lg.amt.values / eqb[eb], loss=lg.ret.values * lg.amt.values / eqb[eb],
                  coin=[syms[int(k)] for k in lg.k.values], t_in=idx[eb], t_out=idx[lg.xb.astype(int).values])
    first = pd.Series([1.0], [e.index[0] - pd.Timedelta(hours=1)])
    ed = pd.concat([first, e.resample("D").last()])
    ew = pd.concat([first, e.resample("W-SUN").last()])
    dr, wr = ed.pct_change().dropna(), ew.pct_change().dropna()
    out = {}
    for p, (a, b) in {"d": (T0, T1), "t": (T1, e.index[-1] + pd.Timedelta(hours=1))}.items():
        g = L[(L.t_in >= a) & (L.t_in < b)].sort_values("loss")
        dd_, ww = dr[(dr.index >= a) & (dr.index < b)], wr[(wr.index >= a) & (wr.index < b)]
        ep = e[(e.index >= a) & (e.index < b)]
        edp = ep.resample("D").last()
        r6, r12 = (edp / edp.shift(182) - 1).dropna(), (edp / edp.shift(365) - 1).dropna()
        uw = (ep < ep.cummax()).values
        runs, cur = 0, 0
        for x in uw:
            cur = cur + 1 if x else 0
            runs = max(runs, cur)
        out[p] = dict(trades=len(g), top=g.head(5)[["coin", "t_in", "t_out", "ret", "size", "loss"]].to_dict("records"),
                      n_loss5=int((g.loss < -0.05).sum()), n_loss10=int((g.loss < -0.10).sum()),
                      n_loss20=int((g.loss < -0.20).sum()),
                      worst_day=float(dd_.min()), worst_day_at=dd_.idxmin(), worst_week=float(ww.min()),
                      worst_week_at=ww.idxmin(), worst_m=metrics(e, start=a, end=b)["worst_m"],
                      worst_6m=float(r6.min()), lose_6m=float((r6 < 0).mean()), worst_12m=float(r12.min()) if len(r12) else np.nan,
                      lose_12m=float((r12 < 0).mean()) if len(r12) else np.nan, underwater_days=runs / 24.0)
    return out


def say_conc(name, label, C):
    for p, tag in (("d", "2021-23"), ("t", "2024-26")):
        c = C[p]
        say(f"  [{label}] {tag}: {c['trades']} trades | trades losing >5% of equity {c['n_loss5']}, >10% {c['n_loss10']}, "
            f">20% {c['n_loss20']} | worst day {c['worst_day']*100:+.1f}% ({c['worst_day_at']:%Y-%m-%d}) | worst week "
            f"{c['worst_week']*100:+.1f}% (to {c['worst_week_at']:%Y-%m-%d}) | worst month {c['worst_m']*100:+.1f}% | "
            f"worst 6m {c['worst_6m']*100:+.1f}% ({c['lose_6m']*100:.0f}% of 6m windows lose) | worst 12m "
            f"{c['worst_12m']*100:+.1f}% ({c['lose_12m']*100:.0f}% of 12m windows lose) | longest under water "
            f"{c['underwater_days']:.0f} days")
        for r in c["top"]:
            say(f"      {r['coin']:10s} bought {r['t_in']:%Y-%m-%d %H:%M} sold {r['t_out']:%Y-%m-%d %H:%M}  trade "
                f"{r['ret']*100:+6.1f}%  size {r['size']*100:5.1f}% of equity  -> {r['loss']*100:+6.1f}% of equity")


def fmt(r):
    return (f"[{r.get('d_trades', -1):4.0f}/{r.get('t_trades', -1):4.0f} trades] 21-23 {r['d_pm']*100:+5.1f}%/mo x{r['d_mult']:8.2f} DD {r['d_dd']*100:4.0f}% | "
            f"24-26 {r['t_pm']*100:+5.1f}%/mo x{r['t_mult']:6.2f} DD {r['t_dd']*100:4.0f}% worst-mo {r['t_worst_m']*100:+4.0f}%")


def self_test_grid(name, kw, r):
    g = pd.read_csv(os.path.join(HERE, "results", "grid960_19coins.csv"))
    m = g[np.isclose(g.depth, kw["depth"]) & (g.N == kw["N"]) & np.isclose(g.fmax, kw["fmax"]) &
          np.isclose(g.tp, kw["tp"]) & (g.hm == kw["hm"])]
    if len(m) != 1:
        return f"(not in grid: {len(m)} rows)"
    m = m.iloc[0]
    ok = all(abs(r[k] - m[k]) < 1e-6 * max(1, abs(m[k])) for k in ("d_mult", "d_dd", "t_mult", "t_dd"))
    if not ok:
        raise SystemExit(f"{name}: normal run does not reproduce grid960 ({r['d_mult']} {r['t_mult']} vs {m.d_mult} {m.t_mult})")
    return "reproduces grid960: yes"


def main():
    global _log
    _log = open(os.path.join(OUT, "hr_stress.log"), "w")
    t00 = time.time()
    say(f"hr_stress  PB_DATA={E.DATA}  PB_OUT={OUT}")
    syms19, idx19, P19 = S.market(False)
    syms22, idx22, P22 = S.market(True)
    say(f"hourly panels: 19 coins {idx19[0]} .. {idx19[-1]}; 22 coins {idx22[0]} .. {idx22[-1]}")
    rows, conc, band_rows = [], {}, []
    i21 = {d: idx19.get_loc(T0 + pd.Timedelta(days=d)) for d in range(8)}
    i24 = {d: idx19.get_loc(T1 + pd.Timedelta(days=d)) for d in range(30)}

    # ---------------------------------------------------------------- hourly rows + jitter (cheap)
    for name, kw in SETS.items():
        e, lg = E.run(P19, idx19, **kw)
        r = ntrades(periods(e), lg, idx19); r.update(setting=name, row="normal")
        sc = E.score(e, lg, idx19); r.update(t_win=sc["t_win"], t_avg=sc["t_avg"], t_worst=sc["t_worst"])
        say(f"\n{name}: normal  {fmt(r)}  {self_test_grid(name, kw, r)}")
        rows.append(r)
        conc[(name, "normal")] = concentration(e, lg, idx19, syms19)
        for row, P, idx, extra in (("bot timing", P19, idx19, BOT_H), ("fills 0.2% worse", P19, idx19, dict(pen=0.002)),
                                   ("+LUNA/FTT/SRM", P22, idx22, {})):
            e2, lg2 = E.run(P, idx, **{**kw, **extra})
            r2 = ntrades(periods(e2), lg2, idx); r2.update(setting=name, row=row)
            rows.append(r2)
            say(f"  {row:22s} {fmt(r2)}")
            if row == "+LUNA/FTT/SRM":
                conc[(name, row)] = concentration(e2, lg2, idx, syms22)
                fz = frozen(e2, P22, syms22)
                if fz:
                    say(fz)
        # jitter: depth +-0.1 point (full runs), start +1..7 days (fresh runs per period)
        for dd in (-0.001, 0.001):
            e2, _ = E.run(P19, idx19, **{**kw, "depth": kw["depth"] + dd})
            r2 = periods(e2); r2.update(setting=name, kind="depth", label=f"depth {(kw['depth']+dd)*100:.1f}%")
            band_rows.append(r2)
        for d in range(1, 8):
            e21, _ = run_from(P19, idx19, kw, i21[d])
            e24, _ = run_from(P19, idx19, kw, i24[d])
            a, b = periods(e21, shift_d=d), periods(e24, shift_t=d)
            band_rows.append(dict(setting=name, kind="start", label=f"start +{d}d",
                                  **{k: a[k] for k in a if k.startswith("d_") or k == "y2022"},
                                  **{k: b[k] for k in b if k.startswith("t_")},
                                  t_mult_cont=a["t_mult"], t_dd_cont=a["t_dd"]))
        for d in range(30):
            e24, _ = run_from(P19, idx19, kw, i24[d])
            b = periods(e24, shift_t=d)
            band_rows.append(dict(setting=name, kind="start30", label=f"2024 fresh +{d}d", **{k: b[k] for k in b if k.startswith("t_")}))
    say(f"\nhourly part done in {time.time()-t00:.0f}s")

    # ---------------------------------------------------------------- 1-minute rows (panel loaded once, 22 coins)
    t1 = time.time()
    idxm, A, hidx, Ph = E.minute_panel(E.H19 + list(E.DEAD), cuts=E.DEAD)
    syms_m = E.H19 + list(E.DEAD)
    say(f"1-minute panel: {A['o'].shape} {hidx[0]} .. {hidx[-1]}, loaded in {time.time()-t1:.0f}s, "
        f"max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB")
    # Collapse coins: minute_panel fills missing minutes flat at the last close up to each coin's cut, so the 1-minute
    # engine could trade them while Binance did not: LUNA's last real minute is 2022-05-13 00:39 (cut 23:00), SRM's
    # 2022-11-28 02:59 (cut 23:00), and FTT was halted 2022-11-15 04:29 .. 2023-09-22 08:00 (a flat-filled FTT could be
    # sold at its frozen pre-halt price, or the reopen bought off a stale pre-halt reference price). Blank every gap of
    # more than 6 hours in their raw minutes and everything after the last real minute before the cut, as the corrected
    # hourly engine sees them (no trade, no reference price until 24h of new closes; a held coin sells when trading
    # resumes, or stays held at its last price if trading never resumes). The 19 live coins have no gap over 6 hours.
    for s_ in E.DEAD:
        k_ = syms_m.index(s_)
        tr = pd.to_datetime(np.load(os.path.join(E.DATA, "1m", f"{s_}.npz"))["t"], unit="s")
        tr = tr[tr <= pd.Timestamp(E.DEAD[s_]) + pd.Timedelta(minutes=59)]
        gaps = [(tr[i], tr[i + 1]) for i in np.flatnonzero(np.diff(tr.values) > np.timedelta64(6, "h"))]
        for g0, g1 in gaps + [(tr[-1], idxm[-1] + pd.Timedelta(minutes=1))]:
            i0_, i1_ = idxm.searchsorted(g0, side="right"), idxm.searchsorted(g1, side="left")
            for f in A:
                A[f][i0_:i1_, k_] = np.nan
            say(f"{s_}: no trading {g0} .. {g1} -> minutes {idxm[i0_]} .. {idxm[i1_-1]} blanked")
    kf = syms_m.index("FTTUSDT")
    b_ = idxm.get_loc(pd.Timestamp("2023-09-22 08:00"))
    nh = len(hidx)
    Ph = {"o": A["o"][::60], "c": A["c"][59::60], "h": np.fmax.reduce(A["h"].reshape(nh, 60, -1), axis=1),
          "l": np.fmin.reduce(A["l"].reshape(nh, 60, -1), axis=1)}
    Ph = {f: pd.DataFrame(v, hidx, syms_m) for f, v in Ph.items()}
    Ph["cf"] = Ph["c"].ffill()
    Ph["ref"] = Ph["c"].rolling(24, min_periods=1).max().shift(1)
    say(f"FTT ref at the reopen hour (should be nan): {Ph['ref'].iloc[b_ // 60, kf]}")
    A19 = {f: v[:, :19] for f, v in A.items()}                  # views: the 19 live coins only
    Ph19 = {f: v.iloc[:, :19] for f, v in Ph.items()}
    for name, kw in SETS.items():
        for row, AA, PP, extra in (("1-min ref", A19, Ph19, {}), ("strict 1-min fills", A19, Ph19, STRICT),
                                   ("strict + bot timing", A19, Ph19, {**STRICT, **BOT_M}),
                                   ("worst combined", A, Ph, {**STRICT, **BOT_M})):
            t2 = time.time()
            e2, lg2 = R.rrun(AA, hidx, PP, **{**kw, **extra})
            r2 = ntrades(periods(e2), lg2, hidx); r2.update(setting=name, row=row)
            rows.append(r2)
            say(f"  {name:28s} {row:22s} {fmt(r2)}  ({time.time()-t2:.0f}s)")
            if row == "worst combined":
                fz = frozen(e2, Ph, syms_m)
                if fz:
                    say(fz)
                f = lg2[lg2.k.astype(int) == kf]
                for _, x in f[(hidx[f.eb.astype(int).values] >= pd.Timestamp("2022-11-01")) &
                              (hidx[f.eb.astype(int).values] < pd.Timestamp("2023-10-01"))].iterrows():
                    say(f"      FTT trade around the halt: bought {hidx[int(x.eb)]} sold {hidx[int(x.xb)]} {x.ret*100:+.1f}%")
            if row == "worst combined":
                conc[(name, row)] = concentration(e2, lg2, hidx, syms_m)
    del A, A19
    say(f"1-minute part done in {time.time()-t1:.0f}s")

    # ---------------------------------------------------------------- tables
    T = pd.DataFrame(rows)
    B = pd.DataFrame(band_rows)
    T.to_csv(os.path.join(OUT, "hr_stress_rows.csv"), index=False)
    B.to_csv(os.path.join(OUT, "hr_stress_jitter.csv"), index=False)
    say("\n================ SUMMARY: %/month and worst drop (2021-23 | 2024-26)")
    summ = []
    for name in SETS:
        say(f"\n{name}")
        for row in ROWS:
            r = T[(T.setting == name) & (T.row == row)].iloc[0]
            say(f"  {row:22s} {fmt(r)}  2022 {r['y2022']*100:+5.0f}%")
            if r.t_trades == 0:
                say(f"  {'':22s} ^ no trades in 2024-26: the account is wiped or stuck in a delisted coin")
            summ.append(dict(setting=name, row=row, d_pm=r.d_pm, d_dd=r.d_dd, t_pm=r.t_pm, t_dd=r.t_dd, d_mult=r.d_mult,
                             t_mult=r.t_mult, t_worst_m=r.t_worst_m, d_worst_m=r.d_worst_m, y2022=r.y2022))
        g = B[(B.setting == name) & B.kind.isin(["depth", "start"])]
        g30 = B[(B.setting == name) & (B.kind == "start30")]
        s = (f"  {'jitter band (9 runs)':22s} 21-23 {g.d_pm.min()*100:+.1f}..{g.d_pm.max()*100:+.1f}%/mo DD "
             f"{g.d_dd.min()*100:.0f}..{g.d_dd.max()*100:.0f}% | 24-26 {g.t_pm.min()*100:+.1f}..{g.t_pm.max()*100:+.1f}%/mo "
             f"DD {g.t_dd.min()*100:.0f}..{g.t_dd.max()*100:.0f}%")
        say(s)
        say(f"  {'2024 fresh +0..29d':22s} 24-26 {g30.t_pm.min()*100:+.1f}..{g30.t_pm.max()*100:+.1f}%/mo (median "
            f"{g30.t_pm.median()*100:+.1f}%) DD {g30.t_dd.min()*100:.0f}..{g30.t_dd.max()*100:.0f}% (median "
            f"{g30.t_dd.median()*100:.0f}%); runs under 5%/mo: {int((g30.t_pm < 0.05).sum())}/30")
        for lab_ in g.label:
            x = g[g.label == lab_].iloc[0]
            say(f"      {lab_:14s} 21-23 {x.d_pm*100:+5.1f}%/mo DD {x.d_dd*100:4.0f}% | 24-26 {x.t_pm*100:+5.1f}%/mo DD {x.t_dd*100:4.0f}%")
        summ.append(dict(setting=name, row="jitter band", d_pm_min=g.d_pm.min(), d_pm_max=g.d_pm.max(), d_dd_min=g.d_dd.min(),
                         d_dd_max=g.d_dd.max(), t_pm_min=g.t_pm.min(), t_pm_max=g.t_pm.max(), t_dd_min=g.t_dd.min(),
                         t_dd_max=g.t_dd.max()))
        summ.append(dict(setting=name, row="2024 fresh +0..29d", t_pm_min=g30.t_pm.min(), t_pm_max=g30.t_pm.max(),
                         t_pm_med=g30.t_pm.median(), t_dd_min=g30.t_dd.min(), t_dd_max=g30.t_dd.max(), t_dd_med=g30.t_dd.median(),
                         under5=int((g30.t_pm < 0.05).sum())))
    pd.DataFrame(summ).to_csv(os.path.join(OUT, "hr_stress_summary.csv"), index=False)

    say("\n================ CONCENTRATION RISK (single-trade losses as % of equity; worst day / week)")
    crow = []
    for name in SETS:
        say(f"\n{name}")
        for label in ("normal", "+LUNA/FTT/SRM", "worst combined"):
            C = conc[(name, label)]
            say_conc(name, label, C)
            for p in ("d", "t"):
                c = C[p]
                t0_ = c["top"][0] if c["top"] else dict(loss=np.nan, coin="", t_in=pd.NaT)
                crow.append(dict(setting=name, run=label, period=p, trades=c["trades"], worst_trade_loss=t0_["loss"],
                                 worst_trade_coin=t0_["coin"], worst_trade_at=t0_["t_in"],
                                 n_loss5=c["n_loss5"], n_loss10=c["n_loss10"], n_loss20=c["n_loss20"],
                                 worst_day=c["worst_day"], worst_week=c["worst_week"], worst_m=c["worst_m"],
                                 worst_6m=c["worst_6m"], lose_6m=c["lose_6m"], worst_12m=c["worst_12m"],
                                 lose_12m=c["lose_12m"], underwater_days=c["underwater_days"]))
    pd.DataFrame(crow).to_csv(os.path.join(OUT, "hr_stress_concentration.csv"), index=False)
    say(f"\ndone in {time.time()-t00:.0f}s, max RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.1f} GB")


if __name__ == "__main__":
    main()
