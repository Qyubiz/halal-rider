"""w90 ten-watch, step 3: candidates from the grids (w90_watch_grid.py A-D, all in-sample on both periods, i.e. found
by tuning on 2024-26 too), full reports, the stressed version, sensitivity, neighbours and the same settings run by the
live bot's hourly resting bid (no code change), then the selection rule.

Selection rule (fixed by the lead): a design qualifies for a goal only if it meets it in BOTH 2021-23 and 2024-26
(win rate of trades bought in the period; %/month compounded). Among qualifiers: highest worst-of (2021-23, 2024-26,
since 2025) %/month; ties -> shallower worst drop.

Stressed version = 22 coins (LUNA/FTT/SRM added), every buy AND sell 0.2% worse, cooldown noticed 1h later (cd+1).
(The watch bot places its take-profit at once after the market buy, which is what mtp_now models for the hourly bot.)
holdout_candidate() / holdout_candidates(): (name, spec, runner) with runner(A, hidx, Ph, **spec) -> (equity, log),
A = 1-minute o/h/l/c arrays (minutes, coins), hidx/Ph as engine.minute_panel returns them.
"""
import glob
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
from lab import month_returns  # noqa: E402

TEN = dict(N=1, fmax=1.0, depth=0.14, tp=0.05, guard=0.35, hm=48, cd=24)
CANDS = [
    ("F1 wr90 rule pick: 1x100%, d15 tp3.5 g35 hm96", dict(N=1, fmax=1.0, depth=0.15, tp=0.035, guard=0.35, hm=96, cd=24)),
    ("F2 wr90 low-drop: 3x33%, d14 tp3 g35 hm96", dict(N=3, fmax=1 / 3, depth=0.14, tp=0.03, guard=0.35, hm=96, cd=24)),
    ("F3 10% mode as a watch bot (d14 tp5 g35 hm48)", TEN),
    ("a  wr90 plateau: 1x100%, d14 tp3 g45 hm96", dict(N=1, fmax=1.0, depth=0.14, tp=0.03, guard=0.45, hm=96, cd=24)),
    ("b  wr90 plateau: 1x100%, d14 tp2.5 g45 hm96", dict(N=1, fmax=1.0, depth=0.14, tp=0.025, guard=0.45, hm=96, cd=24)),
    ("c  wr90: 2x50%, d14 tp3 g35 hm96", dict(N=2, fmax=0.5, depth=0.14, tp=0.03, guard=0.35, hm=96, cd=24)),
    ("d  best 2024-26: 1x100%, d12 tp3 g35 hm48", dict(N=1, fmax=1.0, depth=0.12, tp=0.03, guard=0.35, hm=48, cd=24)),
    ("e  best worst-of-3 overall: 1x100%, d13 tp8 g35 hm96", dict(N=1, fmax=1.0, depth=0.13, tp=0.08, guard=0.35, hm=96, cd=24)),
]
SENS = [("close under trigger (1/min poll)", dict(det=1)), ("original wicks", dict(wick=None)),
        ("same-minute coins in coin order", dict(ordm=1)), ("market slippage 0.1%", dict(mslip=0.001)),
        ("take-profit active 1h after the fill", dict(tpd=60)), ("time stop to the minute", dict(tsm=1)),
        ("buys 0.2% worse only (engine convention)", dict(pen=0.002))]


# ------------------------------------------------------------------ holdout interface
def _runner(A, hidx, Ph, **spec):
    """1-minute watch bot on any minute panel: A = {o,h,l,c} (minutes, coins), hidx = hourly index, Ph = hourly
    DataFrames with 'ref' (highest close of the 24 closed hours) and 'cf' (forward-filled close), as
    engine.minute_panel returns them. Wicks are trimmed to 1% beyond the body inside (spec 'wick')."""
    K = A["c"].shape[1]
    e, lg, _, _ = W.wrun(A, hidx, Ph, None, K=K, **spec)
    return e, lg


def holdout_candidates():
    return [(n, dict(kw), _runner) for n, kw in CANDS[:3]]


def holdout_candidate():
    """The rule's pick (F1)."""
    return holdout_candidates()[0]


def _hourly_spec(kw):
    """engine.run kwargs for the same settings run by today's hourly bot (resting bid, config change only), with the
    crash guard as an smult mask computed on the panel itself (last close vs highest close of the previous 168 hours)."""
    def spec(P, syms):
        c = P["cf"]
        ok = ((c / c.rolling(168, min_periods=1).max() - 1) >= -kw["guard"]).shift(1, fill_value=True)
        return dict(depth=kw["depth"], N=kw["N"], fmax=kw["fmax"], tp=kw["tp"], hm=kw["hm"], cd=kw["cd"],
                    smult=ok.values.astype(float) if kw["guard"] > 0 else 1.0)
    return spec


def holdout_candidates_hourly():
    """Hourly fallback for a holdout that has no 1-minute data: the same three settings in today's hourly bot,
    (name, spec(P, syms) -> engine.run kwargs, engine.run). Stressed: add mtp_now=True, cd+1, pen=0.002."""
    return [(n + " [hourly bid, config only]", _hourly_spec(kw), E.run) for n, kw in CANDS[:3]]


# ------------------------------------------------------------------ main
def months_stats(e, a, b=None):
    x = e[(e.index >= a) & ((e.index < b) if b is not None else True)]
    mr = month_returns(x / x.iloc[0])
    return int((mr >= 0.10).sum()), int((mr < 0).sum()), len(mr), float(mr.min())


def hourly_bid(kw, dead=False, stress=False):
    """Same settings, live bot mechanism (hourly resting bid) in engine.run on the hourly files. Stress: bot timing
    (mtp_now, cd+1), buys 0.2% worse (engine convention), LUNA/FTT/SRM."""
    syms, idx, P = E.panel(E.load_h19(dead=dead))
    c = P["cf"]
    g = kw["guard"]
    ok = ((c / c.rolling(168, min_periods=1).max() - 1) >= -g).shift(1, fill_value=True).values.astype(float) \
        if g > 0 else 1.0
    ex = dict(mtp_now=True, cd=kw["cd"] + 1, pen=0.002) if stress else {}
    run = dict(depth=kw["depth"], N=kw["N"], fmax=kw["fmax"], tp=kw["tp"], hm=kw["hm"], cd=kw["cd"])
    run.update(ex)
    e, lg = E.run(P, idx, smult=ok, **run)
    lg = lg.copy(); lg["eqb"] = e.shift(1).fillna(1.0).values[lg.eb.astype(int).values]
    return W.report(e, lg, idx, syms=syms)


if __name__ == "__main__":
    T0 = time.time()
    ctx = W.Ctx()
    hidx = ctx.hidx
    rows = []

    def show(lab, r, extra=""):
        print(f"  {lab:44s} {W.line(r)}{extra}", flush=True)

    for name, kw in CANDS:
        print(f"\n######## {name}", flush=True)
        e, lg, _, lo = ctx.run(19, **kw)
        r = W.report(e, lg, hidx, lo)
        m10, mneg, mn, mworst = months_stats(e, W.V.T24)
        show("base (19 coins)", r, f" | 2024-26 months >=10%: {m10}/{mn}, losing {mneg}, worst {mworst*100:+.1f}%")
        s = dict(kw); s.update(W.STRESS)
        x = ctx.rep(22, **s)
        show("STRESSED (22 coins, fills 0.2% worse, cd+1h)", x)
        r22 = ctx.rep(22, **kw)
        show("LUNA/FTT/SRM added only", r22)
        rr = dict(name=name, **{f"p_{k}": v for k, v in kw.items()}, **r, **{f"x_{k}": v for k, v in x.items()},
                  **{f"l_{k}": v for k, v in r22.items()}, m10=m10, mneg=mneg)
        for lab, ov in SENS:
            q = ctx.rep(19, **{**kw, **ov})
            show(lab, q)
            rr[f"sens_{lab}"] = (q["d_pm"], q["t_pm"], q["s_pm"], q["d_win"], q["t_win"])
        # fresh starts
        fs = []
        for start in ("2024-01-01", "2024-01-08", "2024-01-15", "2024-01-22", "2024-02-01"):
            h0 = int(hidx.searchsorted(pd.Timestamp(start)))
            B = {f: v[h0 * 60:] for f, v in ctx.B(19, 0.01).items()}
            ref = ctx.ref(kw["guard"], 19)[h0:]
            ee, ll, _, _ = W.wrun(None, hidx[h0:], None, None, K=19, B=B, ref=ref, **kw)
            fr = W.report(ee, ll, hidx[h0:])
            fs.append(fr["t_pm"])
        print(f"  fresh starts Jan-Feb 2024 (5): 2024-26 %/month {min(fs)*100:+.2f} .. {max(fs)*100:+.2f}", flush=True)
        rr["fresh_min"], rr["fresh_max"] = min(fs), max(fs)
        # same settings, live mechanism
        hb = hourly_bid(kw)
        show("same settings, HOURLY resting bid (engine)", hb)
        hbx = hourly_bid(kw, dead=True, stress=True)
        show("  hourly bid, stressed (engine convention)", hbx)
        e2, l2, _ = X.xrun(ctx.B(19, 0.01), hidx, ctx.Ph, ctx.SLIP, K=19, ref=ctx.ref(kw["guard"], 19), cyc=60,
                            **{k: kw[k] for k in ("N", "fmax", "depth", "tp", "hm", "cd")})
        hm1 = W.report(e2, l2, hidx)
        show("  hourly bid on minutes, wicks 1% (critic kernel)", hm1)
        rr.update({f"hb_{k}": v for k, v in hb.items()}); rr.update({f"hbx_{k}": v for k, v in hbx.items()})
        rr.update({f"hm1_{k}": v for k, v in hm1.items()})
        # worst trades
        lgt = lg.copy(); lgt["tin"] = hidx[lgt.eb.astype(int).values]; lgt["acct"] = lgt.ret * lgt.amt / lgt.eqb
        lgt["coin"] = [W.SYMS[int(k)] for k in lgt.k]
        w = lgt.nsmallest(3, "acct")
        print("  worst 3 trades (% of account): " + "; ".join(f"{t.coin} {t.tin:%Y-%m-%d} {t.acct*100:+.1f}%"
                                                              for t in w.itertuples()), flush=True)
        rows.append(rr)

    # ---- neighbours of F1 in the grids (one step in one setting)
    G = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(os.path.join(W.OUT, "grid[ABCD]_*.csv")))], ignore_index=True)
    G = G.drop_duplicates(subset=["p_N", "p_depth", "p_tp", "p_guard", "p_hm"])
    G["w3"] = G[["d_pm", "t_pm", "s_pm"]].min(axis=1)
    f1 = CANDS[0][1]
    nb = G[(G.p_N == 1) & (G.p_depth.between(f1["depth"] - 0.0101, f1["depth"] + 0.0101))
           & (G.p_tp.between(f1["tp"] - 0.0051, f1["tp"] + 0.0051)) & (G.p_hm.isin([72, 96, 120]))]
    print(f"\nF1 neighbourhood in the grids (depth +-1, tp +-0.5, hm 72-120, any guard): {len(nb)} designs, worst-of-3 "
          f"%/month median {nb.w3.median()*100:.2f}, min {nb.w3.min()*100:.2f}, max {nb.w3.max()*100:.2f}; "
          f"both-period win rate >= 90% in {int(((nb.d_win >= .9) & (nb.t_win >= .9)).sum())}", flush=True)

    # ---- selection rule over every grid design
    G["wdd"] = G[["d_dd", "t_dd", "s_dd"]].min(axis=1)
    for goal, m in (("wr90", (G.d_win >= 0.9) & (G.t_win >= 0.9)), ("ten", (G.d_pm >= 0.1) & (G.t_pm >= 0.1))):
        q = G[m].sort_values(["w3", "wdd"], ascending=[False, False])
        print(f"rule, goal {goal}: {len(q)} of {len(G)} grid designs qualify" +
              (f"; pick {q.iloc[0]['name']} worst-of-3 {q.iloc[0].w3*100:.2f}%/month" if len(q) else ""), flush=True)
    print(f"max 2024-26 %/month in the grids: {G.t_pm.max()*100:.2f}; max since 2025: {G.s_pm.max()*100:.2f}")

    # ---- frontier: best worst-of-3 at each win-rate level (min of 2021-23 and 2024-26 win rates)
    G["wmin"] = G[["d_win", "t_win"]].min(axis=1)
    print("\nFRONTIER (grids A-D, 611 designs): best worst-of-3 %/month at each win-rate floor (both periods)")
    fr = []
    for lvl in (0.65, 0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.96):
        q = G[G.wmin >= lvl].sort_values(["w3", "wdd"], ascending=[False, False])
        if not len(q):
            continue
        b = q.iloc[0]
        fr.append(dict(level=lvl, **b.to_dict()))
        print(f"  win >= {lvl*100:4.0f}%: {b['name']:28s} win {b.d_win*100:.1f}/{b.t_win*100:.1f}/{b.s_win*100:.1f}% | "
              f"%/mo {b.d_pm*100:+.2f}/{b.t_pm*100:+.2f}/{b.s_pm*100:+.2f} | drop {b.d_dd*100:.0f}/{b.t_dd*100:.0f}/"
              f"{b.s_dd*100:.0f}% (minute marks 24-26 {b.t_dd_min*100:.0f}%)", flush=True)
    lim = G[G.wdd >= -0.40]
    print("same, only designs whose worst drop in all three periods is shallower than -40%:")
    for lvl in (0.80, 0.85, 0.90, 0.92, 0.94, 0.95):
        q = lim[lim.wmin >= lvl].sort_values(["w3", "wdd"], ascending=[False, False])
        if not len(q):
            continue
        b = q.iloc[0]
        print(f"  win >= {lvl*100:4.0f}%: {b['name']:28s} win {b.d_win*100:.1f}/{b.t_win*100:.1f}/{b.s_win*100:.1f}% | "
              f"%/mo {b.d_pm*100:+.2f}/{b.t_pm*100:+.2f}/{b.s_pm*100:+.2f} | drop {b.d_dd*100:.0f}/{b.t_dd*100:.0f}/"
              f"{b.s_dd*100:.0f}%", flush=True)
    pd.DataFrame(fr).to_csv(os.path.join(W.OUT, "frontier.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(W.OUT, "final.csv"), index=False)

    # ---- the holdout runner on a float64 engine.minute_panel panel (the lead's path) == the cache run
    name, spec, runner = holdout_candidate()
    idxm, A64, h64, P64 = E.minute_panel(E.H19, start="2026-01-01")
    e64, l64 = runner(A64, h64, P64, **spec)
    i0 = int(hidx.searchsorted(h64[0]))
    nh = min(len(h64), len(hidx) - i0)
    A32 = {f: np.asarray(ctx.A[f][i0 * 60:(i0 + nh) * 60, :19]) for f in ("o", "h", "l", "c")}
    Ph32 = {"c": pd.DataFrame(A32["c"][59::60], h64[:nh], E.H19)}
    Ph32["cf"] = Ph32["c"].ffill(); Ph32["ref"] = Ph32["c"].rolling(24, min_periods=1).max().shift(1)
    e32, l32 = runner(A32, h64[:nh], Ph32, **spec)
    print(f"\nholdout runner check (2026 window): float64 minute_panel trades {len(l64)}, final x{e64.iloc[-1]:.6f}; "
          f"float32 cache trades {len(l32)}, final x{e32.iloc[-1]:.6f}", flush=True)
    print(f"done in {time.time()-T0:.0f}s")
