"""Lever "stack", POST-HOC (hindsight; written after x10_stack_grid.py showed the stacked grid tops out at 5.6%/month
in 2024-26, below plain 2 coins x 50% +3% cd 24 at 6.3%). Nothing here is a fair pick: it explains why the
boosters do not add up in 2024-26 and measures how far any stack can go with hindsight.

Part A: one booster at a time on the cd-24 concentrated bases (N2 x 50% +3%, N1 x 100% +3%), all periods.
Part B: anatomy 2024-26 vs 2021-23 for N2 x 50% +3% at cd 24 vs cd 2: trades, re-buys of the same coin within
        24h of its previous sale and what they earned, cash idle share.
Log: $PB_OUT/posthoc.log
"""
import numpy as np
import pandas as pd

import x10_stack_lib as X

BASES = {"N2 cap50 tp3 cd24": dict(depth=0.12, N=2, fmax=0.5, tp=0.03, hm=48, cd=24),
         "N1 cap100 tp3 cd24": dict(depth=0.12, N=1, fmax=1.0, tp=0.03, hm=48, cd=24)}
MODS = {"(base)": {}, "cd 0": dict(cd=0), "cd 2": dict(cd=2), "cd 6": dict(cd=6), "cd 12": dict(cd=12),
        "cd 48": dict(cd=48), "cd 72": dict(cd=72),
        "tpwait 2": dict(tpwait=2), "tpwait 4": dict(tpwait=4), "no rest tpwait 3": dict(tprest=False, tpwait=3),
        "ladder 50/18": dict(w=0.5, d2=0.18), "ladder 67/18": dict(w=2 / 3, d2=0.18), "ladder 67/16": dict(w=2 / 3, d2=0.16),
        "ladder 67/20": dict(w=2 / 3, d2=0.20),
        "tpf .2 mkt": dict(tpf=0.2, tpfm=1), "btc5 1 bid x 100%": dict(bx=0.05, bN=1, bf=1.0),
        "btc3 1 bid x 100%": dict(bx=0.03, bN=1, bf=1.0),
        "tp 2%": dict(tp=0.02), "tp 4%": dict(tp=0.04), "hm 36": dict(hm=36), "hm 72": dict(hm=72),
        "depth 11%": dict(depth=0.11), "depth 13%": dict(depth=0.13)}


def ln(r):
    return (f"21-23 {r['d_mo']*100:+6.2f}%/mo DD {r['d_dd']*100:5.1f}% | 24-26 {r['t_mo']*100:+6.2f}%/mo DD "
            f"{r['t_dd']*100:5.1f}% | 25-26 {r['s_mo']*100:+6.2f}%/mo DD {r['s_dd']*100:5.1f}% | 2024 {r['y2024']*100:+5.0f}% "
            f"2025 {r['y2025']*100:+4.0f}% 2026 {r['y2026']*100:+4.0f}%")


def anatomy(name, s):
    syms, idx, P = X.market(False)
    kw = X.spec_kw(P, s)
    e, lg = X.srun(P, idx, slip="auto", **kw)
    lg = lg.copy()
    for c in ("eb", "xb", "k"):
        lg[c] = lg[c].astype(int)
    lg = lg.sort_values("eb").reset_index(drop=True)
    # re-buy: same coin entered within 24h of its previous exit
    prev_x = {}
    rb = np.zeros(len(lg), bool)
    for i, r in lg.iterrows():
        px = prev_x.get(r.k)
        rb[i] = px is not None and r.eb - px < 24
        prev_x[r.k] = r.xb
    lg["rebuy"] = rb
    lg["con"] = lg.ret * lg.amt / lg.eq0
    # invested share of the account per hour (approx: sum of open budgets / equity)
    inv = np.zeros(len(idx))
    for r in lg.itertuples():
        inv[r.eb:r.xb + 1] += r.amt / r.eq0
    for p, a, b in (("21-23", idx[0], X.T_TEST0), ("24-26", X.T_TEST0, idx[-1] + pd.Timedelta(hours=1)),
                    ("25-26", X.T_25, idx[-1] + pd.Timedelta(hours=1))):
        m = (idx[lg.eb.values] >= a) & (idx[lg.eb.values] < b)
        g = lg[m]
        hm = (idx >= a) & (idx < b)
        months = hm.sum() / (24 * 365.25 / 12)
        rbg = g[g.rebuy]
        print(f"  {name:22s} {p}: {len(g)/months:5.1f} trades/mo, avg ret {g.ret.mean()*100:+.2f}%, sum of "
              f"trade P&L/account {g.con.sum()*100:+7.1f}% | re-buys <24h: {len(rbg)} avg {rbg.ret.mean()*100 if len(rbg) else np.nan:+.2f}% "
              f"sum {rbg.con.sum()*100:+6.1f}% | time stops {int((g.why == 1).sum())} avg {g.ret[g.why == 1].mean()*100:+.1f}% | "
              f"invested {inv[hm].mean()*100:4.1f}% of hours-weighted account, hours with 0 invested {np.mean(inv[hm] < 1e-9)*100:4.1f}%")


if __name__ == "__main__":
    print("POST-HOC / HINDSIGHT: normal runs (19 coins, normal costs)")
    for bn, b in BASES.items():
        print(f"\n== {bn}")
        for mn, m in MODS.items():
            r, _, _ = X.evaluate({**b, **m})
            print(f"  {mn:20s} {ln(r)}", flush=True)
    print("\n== anatomy")
    anatomy("N2 cap50 tp3 cd24", BASES["N2 cap50 tp3 cd24"])
    anatomy("N2 cap50 tp3 cd2", {**BASES["N2 cap50 tp3 cd24"], "cd": 2})
