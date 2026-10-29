"""Task "pick": choose a high-return Panic Buyer setting HONESTLY (2021-23 only), then look at 2024-26 once.

PRE-REGISTRATION (written 22 Oct 2026 before this script was run; nothing below is changed after the run)
--------------------------------------------------------------------------------------------------------
Caveat known before running: the grid is centred on concentrated settings BECAUSE the 960-setting grid showed
they did best in 2024-26. The grid's location is therefore hindsight-informed; only the choice inside it is honest.

PICK GRID (192 settings, 19 live coins, normal fills, backtest timing):
    depth {0.10, 0.12, 0.15} x N {1, 2, 3} x fmax {0.5, 1.0} x tp {0.015, 0.02, 0.03, 0.05} x hm {48, 96}
    x cd {12, 24}.
    Duplicates skipped: with N >= 2 orders a 0.5 cap is >= 1/N, so it never binds while N coins are eligible and
    fmax 1.0 repeats fmax 0.5; only fmax 0.5 is run for N = 2 and 3 (sizing configs: N1/0.5, N1/1.0, N2, N3).
    A side check re-runs the skipped fmax 1.0 versions and reports how far they differ (not used to pick).

SELECTION RULE (uses 2021-23 only; the selection code sees only 2021-23 columns):
    keep settings with calendar-2022 return >= 0% AND 2021-23 worst drop >= -50%;
    pick the highest 2021-23 CAGR (tie: shallower 2021-23 drop). If none qualifies: no pick.

ONE LOOK AT 2024-26 for the pick: multiple, %/month, worst drop, worst month, months up, per-year returns.
    Monthly rate = multiple ** (1 / (years * 12)) - 1, years from the first to the last hour of the period.
    "Reaches the target" = 2024-26 >= 5.0%/month.

PLATEAU: neighbours = settings that differ from the pick in exactly one parameter by one grid step
    (depth, N, fmax, tp, hm, cd; an N >= 2 setting counts as both fmax 0.5 and 1.0). Report each neighbour,
    the share reaching 5%/month in 2024-26 and their 2024-26 worst drops.

STRESS (pick and live settings): (a) + LUNA/FTT/SRM collapses; (b) every fill 0.2% worse (pen=0.002);
    (c) the real bot's timing: mtp_now=True and cooldown one hour longer (cd + 1, i.e. 25 for cd 24);
    (d) all three together.

RISK FRONTIER: universe = the pick grid plus the 960-setting grid re-run here (depth {.08,.10,.12,.15} x
    N {1,2,3,5,8} x fmax {.2,.33,.5,1} x tp {.015,.02,.03,.05} x hm {48,96,168}, cd 24).
    For each 2024-26 monthly target X in {3, 4, 5, 6, 7}%:
    - hindsight: among settings with 2024-26 >= X/month, the shallowest 2024-26 worst drop.
    - ranked on 2021-23: among settings with 2021-23 >= X/month and 2022 >= 0%, the shallowest 2021-23 worst
      drop (tie: higher 2021-23 CAGR); report what that setting did in 2024-26 and whether it reached X.
      Also: how many settings qualified on 2021-23 and what share of them reached X in 2024-26.
"""
import hashlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import study  # noqa: E402
from lab import month_returns  # noqa: E402

OUT = study.OUT
KEYS = ("depth", "N", "fmax", "tp", "hm", "cd")
DEPTHS = [0.10, 0.12, 0.15]
NS = [1, 2, 3]
FMAXS = [0.5, 1.0]
TPS = [0.015, 0.02, 0.03, 0.05]
HMS = [48, 96]
CDS = [12, 24]
AX = dict(depth=DEPTHS, N=NS, fmax=FMAXS, tp=TPS, hm=HMS, cd=CDS)
MIN_2022, MAX_DD, TARGET = 0.0, -0.50, 0.05
TARGETS = [0.03, 0.04, 0.05, 0.06, 0.07]
LIVE = (0.12, 8, 0.20, 0.015, 48, 24)
DESIGN_COLS = list(KEYS) + ["d_mult", "d_mo", "d_cagr", "d_dd", "d_worst_m", "d_pos_m", "y2021", "y2022", "y2023"]


def canon(N, f):
    return (N, 0.5) if N >= 2 else (N, f)


PICK_GRID = sorted({(d,) + canon(N, f) + (tp, hm, cd)
                    for d in DEPTHS for N in NS for f in FMAXS for tp in TPS for hm in HMS for cd in CDS})
G960 = [(d, N, f, tp, hm, 24) for d in (0.08, 0.10, 0.12, 0.15) for N in (1, 2, 3, 5, 8)
        for f in (0.2, 0.33, 0.5, 1.0) for tp in (0.015, 0.02, 0.03, 0.05) for hm in (48, 96, 168)]


def kw(s):
    return dict(depth=float(s[0]), N=int(s[1]), fmax=float(s[2]), tp=float(s[3]), hm=int(s[4]), cd=int(s[5]))


def underwater(e):
    """Worst drop with its peak/trough dates, recovery date, longest time under a previous high, and the lowest
    point relative to the period's starting money."""
    e = e / e.iloc[0]
    dd = e / e.cummax() - 1
    tr = dd.idxmin(); pk = e[:tr].idxmax()
    rec = e[tr:][e[tr:] >= e[pk]]
    under = dd < 0
    grp = (~under).cumsum()
    lens = [(g.index[-1] - g.index[0]).days for _, g in e[under].groupby(grp[under])] if under.any() else [0]
    return dict(dd=float(dd.min()), peak=str(pk.date()), trough=str(tr.date()),
                recovered=str(rec.index[0].date()) if len(rec) else "not yet", longest_under_days=int(max(lens)),
                min_vs_start=float(e.min() - 1))


def run(s, dead=False, **over):
    spec = kw(s)
    spec.update(over)
    r, e, lg = study.evaluate(spec, dead=dead)
    r.update(kw(s))
    r["d_mo"] = (1 + r["d_cagr"]) ** (1 / 12) - 1
    r["t_mo"] = (1 + r["t_cagr"]) ** (1 / 12) - 1
    ys = E.years(e)
    for y in range(2021, 2027):
        r[f"y{y}"] = ys.get(y, np.nan)
    return r, e, lg


def name(s):
    return f"d{s[0]:.2f} N{s[1]} f{s[2]:.2f} tp{s[3]*100:.1f}% hm{s[4]} cd{s[5]}"


def dline(r):
    return (f"21-23 x{r['d_mult']:7.2f} {r['d_mo']*100:+5.2f}%/mo DD {r['d_dd']*100:4.0f}% 2022 {r['y2022']*100:+5.0f}% "
            f"worst-mo {r['d_worst_m']*100:+4.0f}%")


def tline(r):
    return (f"24-26 x{r['t_mult']:6.2f} {r['t_mo']*100:+5.2f}%/mo DD {r['t_dd']*100:4.0f}% worst-mo "
            f"{r['t_worst_m']*100:+4.0f}% up-mo {r['t_pos_m']*100:3.0f}%")


def neighbours(s):
    reps = [s] if s[1] == 1 else [(s[0], s[1], 0.5) + s[3:], (s[0], s[1], 1.0) + s[3:]]
    out = []
    for rep in reps:
        for i, k in enumerate(KEYS):
            vals = AX[k]
            j = vals.index(rep[i])
            for jj in (j - 1, j + 1):
                if 0 <= jj < len(vals):
                    t = list(rep); t[i] = vals[jj]
                    t[1], t[2] = canon(t[1], t[2])
                    t = tuple(t)
                    if t != s and t not in out:
                        out.append(t)
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    sha = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
    print(f"hr_pick.py sha256 {sha}")
    print(f"pick grid: {len(PICK_GRID)} settings | frontier universe adds the 960-setting grid (cd 24)\n")

    # ---------------------------------------------------------------- 1. run the pick grid
    res = {}
    for s in PICK_GRID:
        res[s] = run(s)[0]
    R = pd.DataFrame([res[s] for s in PICK_GRID])
    R.to_csv(os.path.join(OUT, "pick_grid.csv"), index=False)

    # ---------------------------------------------------------------- 2. select on 2021-23 only
    D = R[DESIGN_COLS].copy()                      # the selection sees no 2024-26 column
    ok = D[(D.y2022 >= MIN_2022) & (D.d_dd >= MAX_DD)]
    print(f"######## SELECTION (2021-23 only): {len(ok)} of {len(D)} settings have 2022 >= 0% and 2021-23 drop >= -50%")
    if not len(ok):
        print("no setting qualifies -> no pick"); return
    ok = ok.sort_values(["d_cagr", "d_dd"], ascending=[False, False])
    for _, r in ok.head(10).iterrows():
        print(f"  {name(tuple(r[k] for k in KEYS)):38s} {dline(r)}")
    p = ok.iloc[0]
    pick = (float(p.depth), int(p.N), float(p.fmax), float(p.tp), int(p.hm), int(p.cd))
    print(f"\nPICK: {name(pick)}")

    # ---------------------------------------------------------------- 3. one look at 2024-26
    pr, pe, plg = run(pick)
    lr, le, llg = run(LIVE)
    print("\n######## PICK vs LIVE, 2024-26 looked at once")
    for lab_, r, e in (("PICK", pr, pe), ("LIVE", lr, le)):
        print(f"{lab_}: {dline(r)}\n      {tline(r)}  trades {r.get('t_trades')} win {r.get('t_win', np.nan)*100:.0f}% "
              f"avg {r.get('t_avg', np.nan)*100:+.2f}% worst trade {r.get('t_worst', np.nan)*100:+.0f}%")
        print("      per year: " + " ".join(f"{y} {r[f'y{y}']*100:+.0f}%" for y in range(2021, 2027)))
        mr = month_returns(e[e.index >= E.DESIGN_END] / e[e.index >= E.DESIGN_END].iloc[0])
        print(f"      2024-26 months: {len(mr)}, up {int((mr > 0).sum())}, down {int((mr <= 0).sum())}, "
              f"worst {mr.min()*100:+.1f}% ({mr.idxmin():%Y-%m}), best {mr.max()*100:+.1f}%, "
              f"months below -10%: {int((mr < -0.10).sum())}")
        for per, ee in (("2021-23", e[e.index < E.DESIGN_END]), ("2024-26", e[e.index >= E.DESIGN_END])):
            u = underwater(ee)
            print(f"      {per} worst drop {u['dd']*100:.1f}% peak {u['peak']} trough {u['trough']} recovered {u['recovered']}; "
                  f"longest under a high {u['longest_under_days']} days; lowest vs period start {u['min_vs_start']*100:+.1f}%")
    print(f"  pick reaches 5%/month in 2024-26: {'YES' if pr['t_mo'] >= TARGET else 'NO'} ({pr['t_mo']*100:.2f}%/month)")
    mt = pd.DataFrame({"pick": month_returns(pe[pe.index >= E.DESIGN_END] / pe[pe.index >= E.DESIGN_END].iloc[0]),
                       "live": month_returns(le[le.index >= E.DESIGN_END] / le[le.index >= E.DESIGN_END].iloc[0])})
    mt.to_csv(os.path.join(OUT, "pick_months_2024_26.csv"))

    # ---------------------------------------------------------------- 4. plateau
    nb = neighbours(pick)
    print(f"\n######## PLATEAU: {len(nb)} one-step neighbours of the pick")
    rows = []
    for s in nb:
        r = res[s] if s in res else run(s)[0]
        rows.append(r)
        print(f"  {name(s):38s} {dline(r)} | {tline(r)}" + ("  >=5%/mo" if r["t_mo"] >= TARGET else ""))
    NB = pd.DataFrame(rows)
    NB.to_csv(os.path.join(OUT, "plateau.csv"), index=False)
    hit = NB[NB.t_mo >= TARGET]
    print(f"  {len(hit)} of {len(NB)} neighbours reach 5%/month in 2024-26 ({len(hit)/len(NB)*100:.0f}%); their 2024-26 drops: "
          + ", ".join(f"{x*100:.0f}%" for x in sorted(hit.t_dd)))
    print(f"  all neighbours: 2024-26 %/month median {NB.t_mo.median()*100:.2f}, min {NB.t_mo.min()*100:.2f}, max {NB.t_mo.max()*100:.2f}; "
          f"2024-26 drop median {NB.t_dd.median()*100:.0f}%, worst {NB.t_dd.min()*100:.0f}%; "
          f"2021-23 qualify under the rule: {int(((NB.y2022 >= 0) & (NB.d_dd >= -0.5)).sum())}")

    # ---------------------------------------------------------------- 5. stress
    print("\n######## STRESS")
    srows = []
    for lab_, s in (("PICK", pick), ("LIVE", LIVE)):
        cd1 = s[5] + 1
        for sc, dead, over in (("base", False, {}), ("+LUNA/FTT/SRM", True, {}), ("fills 0.2% worse", False, dict(pen=0.002)),
                               (f"bot timing (mtp_now, cd {cd1})", False, dict(mtp_now=True, cd=cd1)),
                               ("all three", True, dict(pen=0.002, mtp_now=True, cd=cd1))):
            r, e, _ = run(s, dead=dead, **over)
            r.update(who=lab_, scenario=sc)
            srows.append(r)
            print(f"  {lab_} {sc:30s} {dline(r)} | {tline(r)}")
            print(f"  {'':5s}{'':30s} per year: " + " ".join(f"{y} {r[f'y{y}']*100:+.0f}%" for y in range(2021, 2027)))
    pd.DataFrame(srows).to_csv(os.path.join(OUT, "stress.csv"), index=False)

    # ---------------------------------------------------------------- 6. risk frontier
    uni = list(dict.fromkeys(PICK_GRID + G960))
    for s in uni:
        if s not in res:
            res[s] = run(s)[0]
    U = pd.DataFrame([res[s] for s in uni])
    U.to_csv(os.path.join(OUT, "universe.csv"), index=False)
    g = pd.read_csv(os.path.join(HERE, "results", "grid960_19coins.csv"))
    m = U[U.cd == 24].merge(g, on=["depth", "N", "fmax", "tp", "hm"], suffixes=("", "_old"))
    print(f"\n######## RISK FRONTIER over {len(U)} settings (re-run grid960 vs saved CSV: {len(m)} matched, "
          f"max |t_mult diff| {np.abs(m.t_mult - m.t_mult_old).max():.2e}, max |d_mult diff| {np.abs(m.d_mult - m.d_mult_old).max():.2e})")
    frows = []
    for X in TARGETS:
        h = U[U.t_mo >= X].sort_values(["t_dd", "t_cagr"], ascending=[False, False])
        q = U[(U.d_mo >= X) & (U.y2022 >= 0)].sort_values(["d_dd", "d_cagr"], ascending=[False, False])
        fr = dict(target=X, n_hind=len(h), n_rank=len(q))
        if len(h):
            a = h.iloc[0]
            fr.update(hind=name(tuple(a[k] for k in KEYS)), hind_t_mo=a.t_mo, hind_t_dd=a.t_dd, hind_d_dd=a.d_dd, hind_d_mo=a.d_mo)
            print(f"  {X*100:.0f}%/mo hindsight ({len(h)} reach it): {name(tuple(a[k] for k in KEYS)):38s} "
                  f"24-26 {a.t_mo*100:+.2f}%/mo DD {a.t_dd*100:.0f}% (its 21-23: {a.d_mo*100:+.2f}%/mo DD {a.d_dd*100:.0f}%)")
        else:
            print(f"  {X*100:.0f}%/mo hindsight: no setting reaches it")
        if len(q):
            b = q.iloc[0]
            share = float((q.t_mo >= X).mean())
            fr.update(rank=name(tuple(b[k] for k in KEYS)), rank_d_mo=b.d_mo, rank_d_dd=b.d_dd, rank_t_mo=b.t_mo,
                      rank_t_dd=b.t_dd, rank_hit=bool(b.t_mo >= X), share_q_hit=share)
            print(f"  {X*100:.0f}%/mo ranked on 21-23 ({len(q)} qualify): {name(tuple(b[k] for k in KEYS)):38s} "
                  f"21-23 {b.d_mo*100:+.2f}%/mo DD {b.d_dd*100:.0f}% -> 24-26 {b.t_mo*100:+.2f}%/mo DD {b.t_dd*100:.0f}% "
                  f"{'reached' if b.t_mo >= X else 'MISSED'}; {share*100:.0f}% of the {len(q)} reached {X*100:.0f}% in 24-26")
        else:
            print(f"  {X*100:.0f}%/mo ranked on 21-23: no setting qualifies")
        frows.append(fr)
    pd.DataFrame(frows).to_csv(os.path.join(OUT, "frontier.csv"), index=False)

    # ---------------------------------------------------------------- 7. duplicate check (not used to pick)
    diffs = []
    for s in PICK_GRID:
        if s[1] >= 2:
            r = run((s[0], s[1], 1.0) + s[3:])[0]
            diffs.append((abs(r["d_mult"] / res[s]["d_mult"] - 1), abs(r["t_mult"] / res[s]["t_mult"] - 1)))
    dd_ = np.array(diffs)
    print(f"\nduplicate check, N>=2 fmax 1.0 vs 0.5 ({len(dd_)} pairs): max relative diff 21-23 {dd_[:, 0].max():.2e}, "
          f"24-26 {dd_[:, 1].max():.2e}; pairs differing by >0.1%: {int(((dd_ > 0.001).any(axis=1)).sum())}")


if __name__ == "__main__":
    main()
