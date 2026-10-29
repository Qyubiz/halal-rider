"""Adversarial check of the hr_pick.py pick (d12 N2 f0.5 tp5 hm48 cd24). Written after reading hr_pick.py's outputs.

1. Reproduce: re-run the 192-setting pick grid, re-apply the rule on 2021-23 columns only, compare with pick_grid.csv.
2. Reproduce pick / live headline numbers and the stress rows.
3. Trade concentration in 2024-26: top-5 trades' share of profit (dollar and log/compounded), by coin, by month,
   how many positions are held at once and the largest single position as a share of the account.
4. Leave-one-coin-out (exact engine re-runs, coin never traded): 2024-26 %/month with each coin removed.
5. Per calendar year vs the live settings (base and real-bot timing).
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import hr_pick as H  # noqa: E402
import study  # noqa: E402
from lab import month_returns  # noqa: E402

OUT = study.OUT
PICK_DIR = os.path.join(os.path.dirname(OUT.rstrip("/")), "pick")
PICK = (0.12, 2, 0.5, 0.05, 48, 24)
LIVE = H.LIVE
T0 = E.DESIGN_END


def mo_rate(mult, t0, t1):
    months = (t1 - t0).total_seconds() / (365.25 * 86400) * 12
    return mult ** (1 / months) - 1, months


def concentration(tag, e, lg, idx, syms):
    tt = lg[idx[lg.eb.astype(int).values] >= T0].copy()
    tt["entry_t"] = idx[tt.eb.astype(int).values]
    tt["exit_t"] = idx[tt.xb.astype(int).values]
    tt["coin"] = [syms[int(k)][:-4] for k in tt.k]
    eb = tt.eb.astype(int).values
    tt["eq_before"] = e.values[eb - 1]
    tt["pnl"] = tt.amt * tt.ret                       # account units (start of 2021 = 1.0)
    tt["c"] = tt.pnl / tt.eq_before                   # account-relative return of the trade
    tt["size"] = tt.amt / tt.eq_before
    tt["logc"] = np.log1p(tt.c)
    e24 = e[e.index >= T0]
    start_eq, end_eq = float(e24.iloc[0]), float(e24.iloc[-1])
    tot = tt.pnl.sum()
    mult = end_eq / start_eq
    rate, months = mo_rate(mult, e24.index[0], e24.index[-1])
    print(f"\n######## TRADE CONCENTRATION 2024-26: {tag}")
    print(f"  equity x{mult:.3f} ({rate*100:+.2f}%/mo over {months:.1f} months); {len(tt)} trades; sum of trade P&L "
          f"{tot:.3f} vs equity change {end_eq - start_eq:.3f} (account units, 2021-01-01 = 1.0)")
    print(f"  product of (1 + account-relative trade return) = x{np.exp(tt.logc.sum()):.3f} (check vs x{mult:.3f})")
    top = tt.sort_values("pnl", ascending=False)
    gross = tt.pnl[tt.pnl > 0].sum()
    for n in (1, 2, 5, 10, 20):
        s = top.head(n)
        print(f"  top {n:2d} trades by P&L: {s.pnl.sum()/tot*100:5.1f}% of net P&L, {s.pnl.sum()/gross*100:5.1f}% of gross profit")
    print("  top 10 trades:")
    for _, r in top.head(10).iterrows():
        print(f"    {r.coin:6s} {r.entry_t:%Y-%m-%d %H}h -> {r.exit_t:%Y-%m-%d %H}h ret {r.ret*100:+6.2f}% size {r['size']*100:4.0f}% "
              f"of account, account +{r.c*100:5.2f}%, P&L {r.pnl:.3f} ({r.pnl/tot*100:4.1f}% of net)")
    lt = tt.sort_values("logc", ascending=False)
    L = tt.logc.sum()
    print("  top 5 trades by account-relative return: " + "; ".join(
        f"{r.coin} {r.entry_t:%Y-%m-%d} ret {r.ret*100:+.1f}% size {r['size']*100:.0f}% -> account {r.c*100:+.2f}%"
        for _, r in lt.head(5).iterrows()))
    for n in (1, 2, 5, 10, 20):
        g = lt.head(n).logc.sum()
        rest = np.exp(L - g)
        rr, _ = mo_rate(rest, e24.index[0], e24.index[-1])
        rest2 = mult / np.exp(g)
        rr2, _ = mo_rate(rest2, e24.index[0], e24.index[-1])
        print(f"  top {n:2d} trades by account-relative return: {g/L*100:5.1f}% of the log growth; without them: "
              f"(a) compounding the other {len(tt)-n}: x{rest:.3f} = {rr*100:+.2f}%/mo; (b) actual x{mult:.3f} divided by "
              f"their growth factors: x{rest2:.3f} = {rr2*100:+.2f}%/mo")
    mr = month_returns(e24 / e24.iloc[0])
    for n in (1, 2, 3):
        b = mr.nlargest(n)
        rest = mult / np.prod(1 + b.values)
        rr, _ = mo_rate(rest, e24.index[0], e24.index[-1])
        print(f"  without the best {n} month(s) ({', '.join(f'{i:%Y-%m} {v*100:+.1f}%' for i, v in b.items())}): "
              f"x{rest:.3f} = {rr*100:+.2f}%/mo over the full {months:.1f} months")
    bc = tt.groupby("coin").agg(trades=("pnl", "size"), pnl=("pnl", "sum"), logc=("logc", "sum")).sort_values("pnl", ascending=False)
    bc["share_net"] = bc.pnl / tot
    bc["share_log"] = bc.logc / L
    print("  by coin (net P&L share / log-growth share):  " + "  ".join(
        f"{c} {r.share_net*100:.0f}%/{r.share_log*100:.0f}% ({int(r.trades)})" for c, r in bc.iterrows()))
    bm = tt.groupby(tt.exit_t.dt.to_period("M")).pnl.sum().sort_values(ascending=False)
    print("  top 5 months by P&L (by exit month): " + ", ".join(f"{p} {v/tot*100:.0f}%" for p, v in bm.head(5).items()))
    by_year = tt.groupby(tt.exit_t.dt.year).agg(n=("pnl", "size"), logc=("logc", "sum"))
    print("  log-growth share by exit year: " + ", ".join(f"{y} {r.logc/L*100:.0f}% ({int(r.n)} trades)" for y, r in by_year.iterrows()))
    # positions held at once (hour resolution) and size
    held = np.zeros(len(idx), np.int64)
    for a, b in zip(lg.eb.astype(int), lg.xb.astype(int)):
        held[a:b + 1] += 1
    h24 = held[idx >= T0]
    print(f"  positions held at once, 2024-26: max {h24.max()}, hours with >2 held {np.mean(h24 > 2)*100:.1f}%, with >=1 held "
          f"{np.mean(h24 >= 1)*100:.1f}%; largest single position {tt['size'].max()*100:.0f}% of the account, median {tt['size'].median()*100:.0f}%")
    print(f"  win rate {np.mean(tt.ret > 0)*100:.0f}%, avg {tt.ret.mean()*100:+.2f}%, worst trade {tt.ret.min()*100:+.1f}%, "
          f"worst account hit from one trade {tt.c.min()*100:+.1f}%, time-stop exits {np.mean(tt.why == 1)*100:.0f}%")
    tt.to_csv(os.path.join(OUT, f"trades_{tag.replace(' ', '_').replace(',', '')}.csv"), index=False)
    return tt


def main():
    os.makedirs(OUT, exist_ok=True)
    syms, idx, P = study.market(False)

    # ------------------------------------------------------------ 1. re-run grid, re-apply rule
    print("######## 1. re-run the 192-setting pick grid and re-apply the rule on 2021-23 columns only")
    rows = [H.run(s)[0] for s in H.PICK_GRID]
    R = pd.DataFrame(rows)
    old = pd.read_csv(os.path.join(PICK_DIR, "pick_grid.csv"))
    m = R.merge(old, on=list(H.KEYS), suffixes=("", "_old"))
    print(f"  {len(R)} settings; matched with saved pick_grid.csv: {len(m)}; max |d_mult diff| "
          f"{np.abs(m.d_mult - m.d_mult_old).max():.2e}, max |t_mult diff| {np.abs(m.t_mult - m.t_mult_old).max():.2e}, "
          f"max |d_dd diff| {np.abs(m.d_dd - m.d_dd_old).max():.2e}")
    D = R[H.DESIGN_COLS].copy()
    ok = D[(D.y2022 >= 0.0) & (D.d_dd >= -0.50)].sort_values(["d_cagr", "d_dd"], ascending=[False, False])
    p = ok.iloc[0]
    print(f"  {len(ok)} qualify; rule picks {H.name(tuple(p[k] for k in H.KEYS))}; 2021-23 drop {p.d_dd*100:.2f}%")
    print(f"  margin to the -50% limit: {(p.d_dd + 0.5)*100:.2f} points; the 2nd choice is "
          f"{H.name(tuple(ok.iloc[1][k] for k in H.KEYS))}")
    # what earlier rounds' rule (study/engine.pick_on_design: beats live in 21-23, 2022 >= -5%, drop >= -35%, best CAGR/|DD|)
    base = H.run(LIVE)[0]
    prev = [r for r in rows if E.beats(r, base, "design") and r["y2022"] >= -0.05 and r["d_dd"] >= -0.35]
    if prev:
        q = max(prev, key=lambda r: r["d_cagr"] / max(1e-9, -r["d_dd"]))
        print(f"  earlier rounds' pick rule (engine.pick_on_design) on the same grid: {len(prev)} qualify; picks "
              f"{H.name(tuple(q[k] for k in H.KEYS))} {H.dline(q)} | {H.tline(q)}")
    else:
        print("  earlier rounds' pick rule (engine.pick_on_design): nothing qualifies on this grid")

    # ------------------------------------------------------------ 2. headline + stress
    print("\n######## 2. headline numbers")
    scen = [("base", False, {}), ("bot timing (mtp_now, cd+1)", False, "T"), ("fills 0.2% worse", False, dict(pen=0.002)),
            ("+LUNA/FTT/SRM", True, {}), ("all three", True, "A")]
    res = {}
    for lab_, s in (("PICK", PICK), ("LIVE", LIVE)):
        for sc, dead, over in scen:
            if over == "T":
                over = dict(mtp_now=True, cd=s[5] + 1)
            elif over == "A":
                over = dict(pen=0.002, mtp_now=True, cd=s[5] + 1)
            r, e, lg = H.run(s, dead=dead, **over)
            res[(lab_, sc)] = (r, e, lg)
            print(f"  {lab_} {sc:28s} {H.dline(r)} | {H.tline(r)} trades {r.get('t_trades')} win {r.get('t_win')*100:.0f}% "
                  f"avg {r.get('t_avg')*100:+.2f}%")
    st = pd.read_csv(os.path.join(PICK_DIR, "stress.csv"))
    for who, sc_old, sc_new in (("PICK", "base", "base"), ("PICK", "bot timing (mtp_now, cd 25)", "bot timing (mtp_now, cd+1)"),
                                ("PICK", "all three", "all three"), ("LIVE", "base", "base")):
        o = st[(st.who == who) & (st.scenario == sc_old)].iloc[0]
        n = res[(who, sc_new)][0]
        print(f"  check vs stress.csv {who} {sc_old}: t_mult {o.t_mult:.4f} vs {n['t_mult']:.4f}, d_mult {o.d_mult:.4f} vs {n['d_mult']:.4f}")

    # ------------------------------------------------------------ 3. concentration
    for key in (("PICK", "base"), ("PICK", "bot timing (mtp_now, cd+1)"), ("LIVE", "base")):
        r, e, lg = res[key]
        concentration(f"{key[0]} {key[1]}", e, lg, idx, syms)

    # ------------------------------------------------------------ 4. leave one coin out
    print("\n######## 4. leave one coin out (coin never traded, exact engine re-run), 2024-26 %/month")
    K = len(syms)
    for lab_, over in (("PICK base", {}), ("PICK bot timing", dict(mtp_now=True, cd=25)), ("LIVE base", None)):
        s = LIVE if over is None else PICK
        over = over or {}
        out = []
        for k in range(K):
            sm = np.ones((len(idx), K)); sm[:, k] = 0.0
            r = H.run(s, smult=sm, **over)[0]
            out.append((syms[k][:-4], r["t_mo"], r["t_dd"], r["d_mo"], r["d_dd"]))
        out.sort(key=lambda x: x[1])
        below = [o for o in out if o[1] < 0.05]
        print(f"  {lab_}: min {out[0][1]*100:+.2f}%/mo (without {out[0][0]}), median {np.median([o[1] for o in out])*100:+.2f}, "
              f"max {out[-1][1]*100:+.2f}; coins whose removal drops it below 5%/mo: {len(below)} of {K}")
        print("    " + ", ".join(f"-{c} {t*100:+.2f}% ({dd*100:.0f}%)" for c, t, dd, _, _ in out))
    # remove the two coins whose single removal hurts most (exact)
    base_e = res[("PICK", "base")]
    tt = pd.read_csv(os.path.join(OUT, "trades_PICK_base.csv"))
    topc = tt.groupby("coin").logc.sum().sort_values(ascending=False).index[:2].tolist()
    sm = np.ones((len(idx), K))
    for c in topc:
        sm[:, syms.index(c + "USDT")] = 0.0
    for lab_, over in (("PICK base", {}), ("PICK bot timing", dict(mtp_now=True, cd=25))):
        r = H.run(PICK, smult=sm, **over)[0]
        print(f"  {lab_} without its top-2 profit coins {topc}: {H.tline(r)} ({r['t_mo']*100:+.2f}%/mo)")

    # ------------------------------------------------------------ 5. per year vs live
    print("\n######## 5. per calendar year, pick vs live (equity curve of the full run; %/mo = compounded monthly rate)")
    yrs = [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01")) for y in range(2021, 2026)]
    yrs.append((pd.Timestamp("2026-01-01"), idx[-1] + pd.Timedelta(hours=1)))
    yrs.append((pd.Timestamp("2025-01-01"), idx[-1] + pd.Timedelta(hours=1)))
    yrs.append((pd.Timestamp("2024-01-01"), idx[-1] + pd.Timedelta(hours=1)))
    keys = [("PICK", "base"), ("PICK", "bot timing (mtp_now, cd+1)"), ("PICK", "all three"), ("LIVE", "base"),
            ("LIVE", "bot timing (mtp_now, cd+1)"), ("LIVE", "all three")]
    rows = []
    for a, b in yrs:
        line = f"  {a:%Y-%m}..{min(b, idx[-1]):%Y-%m}:"
        for k in keys:
            e = res[k][1]
            seg = e[(e.index >= a) & (e.index < b)]
            prev = e[e.index < a]
            s0 = float(prev.iloc[-1]) if len(prev) else 1.0
            mult = float(seg.iloc[-1]) / s0
            rr, months = mo_rate(mult, a, seg.index[-1] + pd.Timedelta(hours=1))
            dd = float((seg / seg.cummax() - 1).min())
            rows.append(dict(start=a, who=k[0], scen=k[1], mult=mult, mo=rr, dd=dd, months=months))
            line += f" | {k[0]} {k[1][:10]} {(mult-1)*100:+6.0f}% {rr*100:+5.2f}%/mo DD {dd*100:4.0f}%"
        print(line)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_year.csv"), index=False)
    for k in (("PICK", "base"), ("LIVE", "base"), ("PICK", "bot timing (mtp_now, cd+1)")):
        e = res[k][1]
        mr = month_returns(e[e.index >= T0] / e[e.index >= T0].iloc[0])
        print(f"  {k[0]} {k[1]}: 2024-26 months {len(mr)}, best 3 " + ", ".join(f"{i:%Y-%m} {v*100:+.1f}%" for i, v in mr.nlargest(3).items())
              + "; worst 3 " + ", ".join(f"{i:%Y-%m} {v*100:+.1f}%" for i, v in mr.nsmallest(3).items())
              + f"; months >= +5%: {int((mr >= 0.05).sum())}, median month {mr.median()*100:+.2f}%")


if __name__ == "__main__":
    main()
