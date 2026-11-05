"""Round 3, task "tphigh": motivation from 2021-23 ONLY (the panel is cut at 2024-01-01 before anything runs).

Question: would a take-profit set part of the way back to the reference 24h high (sell at entry + f x (ref - entry),
ref = highest close of the last 24 closed hourly candles at the entry hour) fit the bounces better than a fixed
+1.5%? For a limit fill entry = 0.88 ref, so ref/entry - 1 = 13.6% and f = 0.11 is +1.5%. The two rules only
differ in how they treat entries deeper than the trigger (market buys when the open is already under it).

Uses the live-settings trade log (engine.run on 2021-23 data) and, per trade, the 48 hourly candles after the fill
hour. A standalone path replay (no cash/cooldown interaction) applies the engine's exit rules to each target:
from the hour after the fill, open >= target -> sell at the open; else high >= target x (1+eps) -> sell at target;
else at the close of hour 48 sell at close x (1-slip). Net of 0.1% fees each side.
"""
import numpy as np
import pandas as pd

import engine as E
import study as S

syms, idx_all, P_all = S.market(False)
keep = idx_all < E.DESIGN_END
idx = idx_all[keep]
P = {f: v[keep] for f, v in P_all.items()}
e, lg = E.run(P, idx)
assert idx[-1] < E.DESIGN_END
print(f"panel {idx[0]} .. {idx[-1]} (2021-23 only), live settings x{e.iloc[-1]:.2f}")

O, H, L, C, R = (P[f].values for f in ("o", "h", "l", "cf", "ref"))
n, K = O.shape
lg = lg[lg.xb.astype(int) < n - 1].copy()            # trades closed inside 2021-23
eb = lg.eb.astype(int).values; kk = lg.k.astype(int).values; ent = lg.entry.values
ref = R[eb, kk]; lim = ref * (1 - E.LIVE["depth"]); op = O[eb, kk]
mkt = op <= lim
g = ref / ent - 1                                    # distance back to the reference high, as a fraction of entry
lg["mkt"] = mkt; lg["g"] = g
print(f"\n{len(lg)} trades closed in 2021-23: {int((lg.why == 0).sum())} take-profits, {int((lg.why == 1).sum())} time stops")
print(f"limit fills {int((~mkt).sum())} (g = {np.median(g[~mkt]) * 100:.2f}% median), market buys {int(mkt.sum())} "
      f"(g median {np.median(g[mkt]) * 100:.1f}%, p25 {np.percentile(g[mkt], 25) * 100:.1f}%, p75 "
      f"{np.percentile(g[mkt], 75) * 100:.1f}%, p90 {np.percentile(g[mkt], 90) * 100:.1f}%)")
for nm, m in (("limit", ~mkt), ("market", mkt)):
    gg = lg[m]
    print(f"  live {nm:6s}: n {len(gg):4d} avg ret {gg.ret.mean() * 100:+.2f}% TP {int((gg.why == 0).sum())} "
          f"time stops {int((gg.why == 1).sum())} (avg {gg.ret[gg.why == 1].mean() * 100:+.1f}%) "
          f"sum ret x size {(gg.ret * gg.amt).sum():+.2f} avg hours held {(gg.xb - gg.eb).mean():.1f}")

# ---- how far back toward the reference high does the bounce go within the 48h window?
HM = E.LIVE["hm"]
mx = np.full(len(lg), np.nan)
for i in range(len(lg)):
    a, k = eb[i] + 1, kk[i]
    b = min(eb[i] + HM, n - 1)
    mx[i] = np.nanmax(H[a:b + 1, k])
retr = (mx - ent) / (ref - ent)                      # fraction of the gap to ref reached (max high, 48h)
up = mx / ent - 1
print("\nShare of trades whose 48h high reaches entry + f x (ref - entry), by entry type and depth bucket:")
bk = pd.cut(g, [0, 0.1365, 0.16, 0.20, 0.30, 10], labels=["limit (13.6%)", "13.6-16%", "16-20%", "20-30%", ">30%"])
hdr = "  bucket            n   " + " ".join(f"f={f:<5}" for f in (0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40)) + \
      "  | +1.5%  +2%  +3%"
print(hdr)
for b in bk.categories:
    m = np.asarray(bk == b)
    if m.sum() == 0:
        continue
    s = " ".join(f"{(retr[m] >= f).mean() * 100:6.0f}%" for f in (0.08, 0.10, 0.125, 0.15, 0.20, 0.25, 0.30, 0.40))
    s2 = " ".join(f"{(up[m] >= x).mean() * 100:4.0f}%" for x in (0.015, 0.02, 0.03))
    print(f"  {b:15s} {m.sum():5d} {s}  | {s2}")


# ---- standalone replay of each trade under different targets
def replay(tgt_frac):
    """tgt_frac: per-trade take-profit as a fraction over entry. Returns (ret, hours held, why)."""
    ret = np.empty(len(lg)); hh = np.empty(len(lg)); why = np.empty(len(lg))
    for i in range(len(lg)):
        k, e0 = kk[i], ent[i]
        T = e0 * (1 + tgt_frac[i])
        px = np.nan
        for t in range(eb[i] + 1, min(eb[i] + HM, n - 1) + 1):
            if np.isnan(O[t, k]):
                continue
            if O[t, k] >= T:
                px = O[t, k]; w = 0
            elif H[t, k] >= T * (1 + E.EPS):
                px = T; w = 0
            elif t - eb[i] >= HM:
                px = C[t, k] * (1 - E.SLIP_MAJOR); w = 1
            if px == px:
                break
        if px != px:                                 # ran out of data
            t = min(eb[i] + HM, n - 1); px = C[t, k]; w = 1
        ret[i] = px / e0 * (1 - E.FEE) ** 2 - 1; hh[i] = t - eb[i]; why[i] = w
    return ret, hh, why


r0, h0, w0 = replay(np.full(len(lg), 0.015))
same = np.mean(np.abs(r0 - lg.ret.values) < 1e-9)
print(f"\nreplay of the live +1.5% matches the engine's trade returns on {same * 100:.1f}% of trades")
rules = {"fixed +1.5% (live)": np.full(len(lg), 0.015)}
for x in (0.02, 0.025, 0.03):
    rules[f"fixed +{x:.1%}"] = np.full(len(lg), x)
for f in (0.08, 0.10, 0.125, 0.15, 0.20, 0.25):
    rules[f"f={f}"] = f * g
for f in (0.06, 0.08, 0.10):
    rules[f"f={f} floor 1.5%"] = np.maximum(0.015, f * g)
print("Standalone replay (each live 2021-23 trade on its own; ignores cash, cooldown and the trades a new rule adds/drops)")
print(f"  {'rule':22s} {'avg ret':>8s} {'TP%':>5s} {'avg h':>6s} {'ret/h(bp)':>9s} | {'limit avg':>9s} "
      f"{'mkt avg':>8s} {'mkt TP%':>7s} {'mkt h':>6s}")
for nm, tf in rules.items():
    r, hh, w = replay(tf)
    print(f"  {nm:22s} {r.mean() * 100:+7.2f}% {(w == 0).mean() * 100:4.0f}% {hh.mean():6.1f} "
          f"{r.sum() / hh.sum() * 1e4:9.2f} | {r[~mkt].mean() * 100:+8.2f}% {r[mkt].mean() * 100:+7.2f}% "
          f"{(w[mkt] == 0).mean() * 100:6.0f}% {hh[mkt].mean():6.1f}")

# ---- per year (2021, 2022, 2023) for the market-buy subset: is the extra take-profit on deep entries stable?
yr = idx[eb].year
print("\nMarket buys only, avg return per trade by year:")
for nm in ("fixed +1.5% (live)", "f=0.1 floor 1.5%", "f=0.08 floor 1.5%", "f=0.15", "f=0.2"):
    r, hh, w = replay(rules[nm])
    print(f"  {nm:22s} " + "  ".join(f"{y}: {r[mkt & (yr == y)].mean() * 100:+.2f}% (n {int((mkt & (yr == y)).sum())}, "
                                       f"{hh[mkt & (yr == y)].mean():.1f}h)" for y in (2021, 2022, 2023)))
print("Limit fills only, avg return per trade by year:")
for nm in ("fixed +1.5% (live)", "f=0.125", "f=0.15", "f=0.2", "f=0.25"):
    r, hh, w = replay(rules[nm])
    print(f"  {nm:22s} " + "  ".join(f"{y}: {r[~mkt & (yr == y)].mean() * 100:+.2f}% ({hh[~mkt & (yr == y)].mean():.1f}h)"
                                       for y in (2021, 2022, 2023)))
