"""Round 3, task "noise": how much do the live settings' numbers move under changes that should not matter?

A measurement only: no grid, no pick, nothing here changes the bot.

1. Jitter band of the live settings (backtest timing, 19 coins), per period (2021-23 and 2024-26). The 15
   perturbations named in the task, each alone:
     depth 11.9% / 12.1%, fee 0.095% / 0.105% a side, eps (a limit must trade this far through) 0.09% / 0.11%,
     order cap 19% / 21% of equity, and the backtest started 1..7 days later.
   A start shift is measured from its own start. 2021-23: a run started at 2021-01-01 + d days, measured from
   there to 2024-01-01. 2024-26: a fresh run (all cash, nothing held) started at 2024-01-01 + d days, measured
   from there to the end of the data. Also reported separately: the 2021 + d runs measured from 2024-01-01 (a start
   shift that only changes the state carried into 2024), and a fresh run started on 2024-01-01 itself.
   Band = min..max over the 15 runs; sd = their standard deviation (ddof 1). For a steadier sd, 100 random draws
   with all five perturbations at once (uniform in the same ranges, start shift 0..7 whole days, drawn
   separately for the two periods). The same bands are measured for the two robustness settings (LUNA/FTT/SRM
   added; every fill 0.2% worse).
2. Near misses from rounds 1-2, judged against the band. Rule fixed here, before any result is seen: a gap is a
   TIE when the near miss's own value lies inside the live band [min, max] of the 15 perturbations for that
   period, setting and metric (ROUND3_PREREG decision 3). Each near miss is also re-run under the same 15
   perturbations and the same random draws (paired: the same perturbation for both bots), counting how often it
   beats live in each period (higher multiple and a drop no more than 1 point deeper, engine.beats).
   A perturbation is applied to the variant's own setting: depth +-0.1 point on whatever depth it uses (16% ->
   15.9/16.1% on flagged coins), the cap x0.95 / x1.05 of whatever cap it uses (the split's 40% -> 38/42%), fee and
   eps set as for live. Specs and runners are imported from the round 1-2 scripts (their own grid dicts), not retyped.
   Supplementary views, printed next to the TIE verdict: the same test against the 8 parameter perturbations only,
   and the gap in units of the random-draw sd.
3. Paired block bootstrap of monthly returns, 2024-26: live vs the round-2 split ladder (r2_crowding.py H2 pick).
   Circular blocks of 1, 3 and 6 months, 20,000 resamples each, the same resampled months for both bots; reported:
   how often the ladder's compounded multiple is higher, how often its mean monthly return is higher, and how often
   it beats live under the rule with the drop measured on month-end values only (misses drops inside a month).
   Also shown, as context, for the other near misses, for 2021-23, and for the ladder in the two robustness settings.

Starting later is done by slicing the panel and every time-indexed setting (features are computed on the full panel
first, so they keep their warm-up). self_test() proves that this equals the full run with buying blocked before the
start (exactly), and that every imported near miss reproduces its reported round 1-2 numbers.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

import engine as E
import study as S
from lab import metrics, month_returns

import idea2b_slow_exit as I2B
import idea4_stress as I4
import r2_crowding as RC
import r2_entry_h2_cascade as EN2
import r2_losers as RL
import r2_trigger_h1_window as TW
import r2_trigger_lib as RT

OUT = S.OUT
NRAND = int(os.environ.get("NOISE_NRAND", 100))
NBOOT = 20000
SEED = 20261022
T0, T1 = pd.Timestamp("2021-01-01"), E.DESIGN_END
LIVE_RUN = E.run                      # the original engine; never replaced in this script

LADDER = "crowding H2 pick: split 50% at 12% / rest at 18%, 4 coins"
NEAR = {   # name: (spec, runner, reported numbers from rounds 1-2 for the self-test: d_mult, d_dd, t_mult, t_dd)
    "entry H2 pick: r1h>0 & knife(-3%,0.90) depth16%":
        (EN2.V["r1h>0 & knife(-3%,0.90) depth16%"], LIVE_RUN, (6.2622, -0.2857, 2.5985, -0.1549)),
    "trigger H1: ref = highest close 16h":
        (TW.V["ref = highest close 16h"], RT.trun, (4.56, -0.27, 2.6139, -0.16)),
    "trigger H1 pick: ref = highest close 12h, depth 11%":
        (TW.V["ref = highest close 12h, depth 11%"], RT.trun, (6.2466, -0.1982, 2.5424, -0.1423)),
    LADDER:
        (RC.H2["split 50% at 12% / rest at 18%, 4 coins"], RC.crun, (9.0738, -0.2345, 2.6723, -0.1512)),
    "idea4 stress-N x=5% N=5":
        (I4.V["stress-N x=5% N=5"], LIVE_RUN, (5.96, -0.31, 2.72, -0.16)),
    "idea4 pick: calm-deeper x=2% +4%":
        (I4.V["calm-deeper x=2% +4%"], LIVE_RUN, (5.77, -0.29, 2.14, -0.15)),
    "losers H1: break-even +0.5% after 24h":
        (RL.H1["break-even +0.5% after 24h"], RL.lrun, (5.70, -0.29, 2.58, -0.16)),
    "idea2b: take-profit placed after 2 hourly cycles":
        (I2B.V["take-profit placed after 2 hourly cycles"], LIVE_RUN, (7.79, -0.32, 2.66, -0.20)),
}
STRATS = {"LIVE": ({}, LIVE_RUN, (5.7500, -0.2924, 2.6088, -0.1628)), **NEAR}
SETTINGS = {"normal": (False, 0.0), "+LUNA/FTT/SRM": (True, 0.0), "fills 0.2% worse": (False, 0.002)}
PERT = [("depth 11.9%", {"dd": -0.001}), ("depth 12.1%", {"dd": 0.001}),
        ("fee 0.095%", {"fee": 0.00095}), ("fee 0.105%", {"fee": 0.00105}),
        ("eps 0.09%", {"eps": 0.0009}), ("eps 0.11%", {"eps": 0.0011}),
        ("cap 19%", {"fm": 0.95}), ("cap 21%", {"fm": 1.05})]
STARTS = list(range(1, 8))


def draws():
    """The random combined perturbations, drawn once and shared by every bot and setting (paired)."""
    rng = np.random.default_rng(SEED)
    return [dict(dd=rng.uniform(-0.001, 0.001), fee=rng.uniform(0.00095, 0.00105), eps=rng.uniform(0.0009, 0.0011),
                 fm=rng.uniform(0.95, 1.05), d21=int(rng.integers(0, 8)), d24=int(rng.integers(0, 8)))
            for _ in range(NRAND)]


# ------------------------------------------------------------------ running
def perturb(kw0, dd=0.0, fm=1.0, fee=None, eps=None):
    kw = dict(kw0)
    if dd:
        kw["depth"] = np.asarray(kw.get("depth", E.LIVE["depth"]), np.float64) + dd
    if fm != 1.0:
        kw["fmax"] = np.asarray(kw.get("fmax", E.LIVE["fmax"]), np.float64) * fm
    if fee is not None:
        kw["fee"] = fee
    if eps is not None:
        kw["eps"] = eps
    return kw


def run_from(runner, P, idx, kw, i0):
    """Run from hour i0 (fresh: all cash, nothing held), slicing the panel and every time-indexed setting."""
    for k, v in kw.items():
        assert not isinstance(v, (pd.Series, pd.DataFrame)), k
    if i0:
        n = len(idx)
        P = {f: v.iloc[i0:] for f, v in P.items()}
        kw = {k: (np.ascontiguousarray(v[i0:]) if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == n else v)
              for k, v in kw.items()}
        idx = idx[i0:]
    return runner(P, idx, **kw)


def m(e, start=None, end=None):
    r = metrics(e, start=start, end=end)
    return r["mult"], r["maxdd"]


def base_kw(spec, P, syms, pen):
    kw = dict(spec(P, syms)) if callable(spec) else dict(spec)
    if pen:
        kw["pen"] = pen
    return kw


def measure(name, spec, runner, setting, D):
    dead, pen = SETTINGS[setting]
    syms, idx, P = S.market(dead)
    kw0 = base_kw(spec, P, syms, pen)
    i21 = {d: idx.get_loc(T0 + pd.Timedelta(days=d)) for d in range(8)}
    i24 = {d: idx.get_loc(T1 + pd.Timedelta(days=d)) for d in range(8)}
    rows, keep = [], None

    def full_row(kind, label, kw, i0=0):
        e, _ = run_from(runner, P, idx, kw, i0)
        dm, ddd = m(e, end=T1)
        tm, tdd = m(e, start=T1)
        return e, dict(setting=setting, strat=name, kind=kind, label=label, d_mult=dm, d_dd=ddd,
                       y2022=E.years(e).get(2022, np.nan), t_mult=tm, t_dd=tdd, t_mult_cont=tm, t_dd_cont=tdd)

    e, r = full_row("base", "base", kw0)
    rows.append(r); keep = e
    for label, p in PERT:
        rows.append(full_row("param", label, perturb(kw0, **p))[1])
    for d in STARTS:
        _, r = full_row("start", f"start +{d}d", kw0, i21[d])
        e24, _ = run_from(runner, P, idx, kw0, i24[d])
        r["t_mult"], r["t_dd"] = m(e24)            # 2024-26 measured from its own (shifted, fresh) start
        rows.append(r)
    e24, _ = run_from(runner, P, idx, kw0, i24[0])
    tm, tdd = m(e24)
    rows.append(dict(setting=setting, strat=name, kind="fresh0", label="fresh run from 2024-01-01", t_mult=tm, t_dd=tdd))
    for j, dr in enumerate(D):
        kw = perturb(kw0, dd=dr["dd"], fm=dr["fm"], fee=dr["fee"], eps=dr["eps"])
        e21, _ = run_from(runner, P, idx, kw, i21[dr["d21"]])
        e24, _ = run_from(runner, P, idx, kw, i24[dr["d24"]])
        dm, ddd = m(e21, end=T1)
        tm, tdd = m(e24)
        rows.append(dict(setting=setting, strat=name, kind="random", label=f"draw {j}", d_mult=dm, d_dd=ddd,
                         y2022=E.years(e21).get(2022, np.nan), t_mult=tm, t_dd=tdd))
    return rows, keep


# ------------------------------------------------------------------ code checks (print no jitter results)
def self_test():
    ok = True
    syms, idx, P = S.market(False)
    # slicing == the full run with buying blocked before the start (equity is exactly 1.0 until then)
    i0 = idx.get_loc(T1 + pd.Timedelta(days=3))
    for name in ("LIVE", "entry H2 pick: r1h>0 & knife(-3%,0.90) depth16%", LADDER, "trigger H1: ref = highest close 16h"):
        spec, runner, _ = STRATS[name]
        kw = base_kw(spec, P, syms, 0.0)
        e1, l1 = run_from(runner, P, idx, kw, i0)
        mask = np.ones(P["o"].shape); mask[:i0] = 0.0
        kw2 = dict(kw); kw2["smult"] = mask * np.asarray(kw.get("smult", 1.0))
        e2, l2 = runner(P, idx, **kw2)
        same = np.array_equal(e1.values, e2.values[i0:]) and bool((e2.values[:i0] == 1.0).all()) and len(l1) == len(l2)
        print(f"  slice == full run blocked before {idx[i0]} for {name}: {same}")
        ok &= same
    # every custom runner with its switches off == engine.run (as proven in rounds 1-2)
    e0, _ = LIVE_RUN(P, idx)
    for nm, rr in (("trun", RT.trun), ("crun", RC.crun), ("lrun", RL.lrun)):
        same = np.array_equal(e0.values, rr(P, idx)[0].values)
        print(f"  {nm} with switches off == engine.run: {same}")
        ok &= same
    # every bot reproduces its reported round 1-2 numbers (4 decimals where the report gave them, else 2)
    for name, (spec, runner, rep) in STRATS.items():
        e, lg = runner(P, idx, **base_kw(spec, P, syms, 0.0))
        r = E.score(e, lg, idx)
        got = (r["d_mult"], r["d_dd"], r["t_mult"], r["t_dd"])
        tol = [0.00005 if len(f"{x}".split(".")[-1]) == 4 else 0.005 for x in rep]
        match = all(abs(g - x) <= t for g, x, t in zip(got, rep, tol))
        print(f"  reproduces round 1-2 numbers: {match}  {name}: " + " ".join(f"{g:.4f}" for g in got)
              + "  reported " + " ".join(f"{x}" for x in rep))
        ok &= match
    if not ok:
        raise SystemExit("self-test failed - stop")


# ------------------------------------------------------------------ summaries
def spread(x):
    x = np.asarray(x, float)
    return dict(n=len(x), min=x.min(), max=x.max(), sd=x.std(ddof=1) if len(x) > 1 else np.nan)


def bands(R):
    """Live jitter bands per setting, perturbation set, period and metric."""
    out = []
    L = R[R.strat == "LIVE"]
    for st in SETTINGS:
        g = L[L.setting == st]
        b = g[g.kind == "base"].iloc[0]
        sets = {"15 named (param + start)": g[g.kind.isin(["param", "start"])], "8 param": g[g.kind == "param"],
                "7 start": g[g.kind == "start"], f"{NRAND} random": g[g.kind == "random"]}
        for sname, h in sets.items():
            for met in ("d_mult", "d_dd", "y2022", "t_mult", "t_dd"):
                out.append(dict(setting=st, set=sname, metric=met, base=b[met], **spread(h[met])))
        h = g[g.kind == "start"]
        for met in ("t_mult_cont", "t_dd_cont"):
            out.append(dict(setting=st, set="7 start, 2024-26 measured from 2024-01-01", metric=met[:-5],
                            base=b[met], **spread(h[met])))
    return pd.DataFrame(out)


def judge(R, B):
    """Each near miss vs the live band, plus the paired win counts."""
    out = []
    for st in SETTINGS:
        L = R[(R.strat == "LIVE") & (R.setting == st)].set_index(["kind", "label"])
        lb = L.loc[("base", "base")]
        for name in NEAR:
            V = R[(R.strat == name) & (R.setting == st)].set_index(["kind", "label"])
            vb = V.loc[("base", "base")]
            row = dict(setting=st, strat=name)
            for per in ("d", "t"):
                for met in ("mult", "dd"):
                    k = f"{per}_{met}"
                    band = B[(B.setting == st) & (B.set == "15 named (param + start)") & (B.metric == k)].iloc[0]
                    row[f"{k}_var"], row[f"{k}_live"] = vb[k], lb[k]
                    row[f"{k}_gap"] = vb[k] - lb[k]
                    row[f"{k}_gap_in_sd"] = (vb[k] - lb[k]) / band["sd"]
                    row[f"{k}_tie"] = bool(band["min"] <= vb[k] <= band["max"])
                    # supplementary views (the TIE rule above is the one fixed in the docstring)
                    b8 = B[(B.setting == st) & (B.set == "8 param") & (B.metric == k)].iloc[0]
                    br = B[(B.setting == st) & (B.set == f"{NRAND} random") & (B.metric == k)].iloc[0]
                    row[f"{k}_tie_param8"] = bool(b8["min"] <= vb[k] <= b8["max"])
                    row[f"{k}_gap_in_sd_rand"] = (vb[k] - lb[k]) / br["sd"]
                row[f"{per}_beats_base"] = bool(vb[f"{per}_mult"] > lb[f"{per}_mult"]
                                                and vb[f"{per}_dd"] >= lb[f"{per}_dd"] - E.DD_TOL)
                for kinds, lab in ((["param", "start"], "15"), (["random"], "rand")):
                    lv = L[L.index.get_level_values(0).isin(kinds)]
                    vv = V.loc[lv.index]
                    w = (vv[f"{per}_mult"] > lv[f"{per}_mult"]) & (vv[f"{per}_dd"] >= lv[f"{per}_dd"] - E.DD_TOL)
                    row[f"{per}_win_{lab}"] = f"{int(w.sum())}/{len(w)}"
                    row[f"{per}_mult_up_{lab}"] = f"{int((vv[f'{per}_mult'] > lv[f'{per}_mult']).sum())}/{len(w)}"
            lv = L[L.index.get_level_values(0) == "random"]
            vv = V.loc[lv.index]
            both = ((vv.d_mult > lv.d_mult) & (vv.d_dd >= lv.d_dd - E.DD_TOL)
                    & (vv.t_mult > lv.t_mult) & (vv.t_dd >= lv.t_dd - E.DD_TOL))
            row["both_win_rand"] = f"{int(both.sum())}/{len(both)}"
            out.append(row)
    return pd.DataFrame(out)


def paired_boot(ra, rb, L, seed):
    """Circular block bootstrap of paired monthly returns (same months for both)."""
    rng = np.random.default_rng(seed)
    n = len(ra)
    nb = -(-n // L)
    st = rng.integers(0, n, size=(NBOOT, nb))
    ix = ((st[:, :, None] + np.arange(L)[None, None, :]) % n).reshape(NBOOT, -1)[:, :n]
    ca, cb = np.cumprod(1 + ra[ix], axis=1), np.cumprod(1 + rb[ix], axis=1)
    ma, mb = ca[:, -1], cb[:, -1]
    dm = (rb - ra)[ix].mean(axis=1)

    def mdd(c):                     # drop on the month-end path only (misses drops inside a month)
        c = np.concatenate([np.ones((len(c), 1)), c], axis=1)
        return (c / np.maximum.accumulate(c, axis=1) - 1).min(axis=1)
    rule = (mb > ma) & (mdd(cb) >= mdd(ca) - E.DD_TOL)
    q = np.quantile(mb / ma, [0.05, 0.5, 0.95])
    return dict(block=L, p_mult_higher=float(np.mean(mb > ma)), p_mean_higher=float(np.mean(dm > 0)),
                p_rule_monthly_dd=float(np.mean(rule)), ratio_q05=q[0], ratio_q50=q[1], ratio_q95=q[2])


def months(e, period):
    e = e[e.index < T1] if period == "2021-23" else e[e.index >= T1]
    return month_returns(e / e.iloc[0])


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    t0 = time.time()
    print(f"r3 noise: NRAND={NRAND}, NBOOT={NBOOT}, seed {SEED}, PB_OUT={OUT}")
    self_test()
    D = draws()
    rows, keep = [], {}
    for st in SETTINGS:
        for name, (spec, runner, _) in STRATS.items():
            rr, e = measure(name, spec, runner, st, D)
            rows += rr
            keep[(st, name)] = e
        print(f"  measured setting {st} ({time.time() - t0:.0f}s)")
        sys.stdout.flush()
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "r3_noise_runs.csv"), index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 500)
    B = bands(R)
    B.to_csv(os.path.join(OUT, "r3_noise_bands.csv"), index=False)
    print("\n######## 1. live settings: jitter bands (multiple; drop and 2022 as fractions)")
    for st in SETTINGS:
        print(f"\n--- {st}")
        b = B[B.setting == st].copy()
        print(b.drop(columns="setting").to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n--- every named perturbation, live, normal setting")
    L = R[(R.strat == "LIVE") & (R.setting == "normal") & R.kind.isin(["base", "param", "start", "fresh0"])]
    print(L[["kind", "label", "d_mult", "d_dd", "y2022", "t_mult", "t_dd", "t_mult_cont", "t_dd_cont"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for st in ("+LUNA/FTT/SRM", "fills 0.2% worse"):
        print(f"\n--- every named perturbation, live, {st}")
        L = R[(R.strat == "LIVE") & (R.setting == st) & R.kind.isin(["base", "param", "start", "fresh0"])]
        print(L[["kind", "label", "d_mult", "d_dd", "y2022", "t_mult", "t_dd", "t_mult_cont", "t_dd_cont"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    J = judge(R, B)
    J.to_csv(os.path.join(OUT, "r3_noise_nearmiss.csv"), index=False)
    print("\n######## 2. near misses vs the live band (tie = the near miss's value is inside live's 15-run range)")
    for st in SETTINGS:
        print(f"\n--- {st}")
        for _, j in J[J.setting == st].iterrows():
            print(f"{j.strat}")
            for per, lab in (("d", "2021-23"), ("t", "2024-26")):
                print(f"   {lab}: x{j[per + '_mult_var']:.4f} vs live x{j[per + '_mult_live']:.4f} "
                      f"(gap {j[per + '_mult_gap']:+.4f} = {j[per + '_mult_gap_in_sd']:+.1f} sd, "
                      f"{'TIE' if j[per + '_mult_tie'] else 'outside band'}) | drop {j[per + '_dd_var'] * 100:.2f}% vs "
                      f"{j[per + '_dd_live'] * 100:.2f}% (gap {j[per + '_dd_gap'] * 100:+.2f} pt = "
                      f"{j[per + '_dd_gap_in_sd']:+.1f} sd, {'TIE' if j[per + '_dd_tie'] else 'outside band'}) | "
                      f"param-only band: mult {'TIE' if j[per + '_mult_tie_param8'] else 'outside'}, "
                      f"drop {'TIE' if j[per + '_dd_tie_param8'] else 'outside'}; gap in random-draw sd: mult "
                      f"{j[per + '_mult_gap_in_sd_rand']:+.1f}, drop {j[per + '_dd_gap_in_sd_rand']:+.1f} | "
                      f"beats live at base {j[per + '_beats_base']} | paired wins: 15 named {j[per + '_win_15']}, "
                      f"random {j[per + '_win_rand']} (multiple higher {j[per + '_mult_up_15']}, {j[per + '_mult_up_rand']})")
            print(f"   beats live in both periods in random draws: {j.both_win_rand}")

    print("\n######## 3. paired block bootstrap of monthly returns (circular blocks, "
          f"{NBOOT} resamples, same months for both bots)")
    BT = []
    pairs = [(per, "normal", name) for per in ("2024-26", "2021-23") for name in NEAR]
    pairs += [("2024-26", st, LADDER) for st in ("+LUNA/FTT/SRM", "fills 0.2% worse")]
    for per, st, name in pairs:
        ra = months(keep[(st, "LIVE")], per)
        rb = months(keep[(st, name)], per)
        assert ra.index.equals(rb.index)
        a, b = ra.values, rb.values
        dd = b - a
        obs = dict(period=per, setting=st, strat=name, months=len(a), months_better=int((dd > 0).sum()),
                   months_worse=int((dd < 0).sum()), mean_diff=dd.mean(), sd_diff=dd.std(ddof=1),
                   t_stat=dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))),
                   mult_live=float(np.prod(1 + a)), mult_var=float(np.prod(1 + b)))
        for L_ in (1, 3, 6):
            BT.append({**obs, **paired_boot(a, b, L_, SEED + L_)})
    BT = pd.DataFrame(BT)
    BT.to_csv(os.path.join(OUT, "r3_noise_bootstrap.csv"), index=False)
    print(BT.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\ndone in {time.time() - t0:.0f}s")
