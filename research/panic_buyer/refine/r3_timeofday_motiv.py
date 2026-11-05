"""Round 3, task "timeofday": motivation from 2021-23 ONLY (the panel is cut at 2024-01-01 before anything runs).

Question: do dips bought in thin hours (low-volume UTC hours, weekends) behave differently from dips bought in busy
hours? The only attribute used is the UTC hour / weekday of the ORDER PLACEMENT, i.e. the start of the hourly cycle in
which the resting bid (or the market buy) was placed and filled: idx[eb]. That is known before the order goes out.

Parts:
  A. Which hours are thin: share of 2021-23 quote volume by UTC hour and by weekday (19 coins, summed).
  B. Live-settings trades (engine.run on the 2021-23 panel, trades closed before 2024) by entry hour block and weekday:
     count, win rate, avg return, time stops, sum of return x order size (equity units), hours held, market-buy share.
  C. "Dip events" without the portfolio's cash/N limits: each coin run alone through engine.run (one coin, so every
     trigger touch outside hold/cooldown is bought), same exit rules. More events, no crowding interaction. Grouped
     the same way, plus by year, to see if any hour/day effect is stable inside 2021-23.
  D. For the same standalone events: how much deeper the dip went (min low over the fill hour and the next 5 hours
     vs ref), and the standalone result if the trigger were deeper (14%, 16%) only in the given hour block. This tells
     whether "deepen in thin hours" is better than "skip in thin hours".
No 2024-26 number is computed or printed by this script.
"""
import numpy as np
import pandas as pd

import engine as E
import study as S

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

syms, idx_all, P_all = S.market(False)
keep = idx_all < E.DESIGN_END
idx = idx_all[keep]
P = {f: v[keep] for f, v in P_all.items()}
assert idx[-1] < E.DESIGN_END
n, K = P["o"].shape
print(f"panel {idx[0]} .. {idx[-1]} (2021-23 only), {K} coins")

# ------------------------------------------------------------------ A. thin hours by quote volume
qv = P["qv"].fillna(0).sum(axis=1)
byh = qv.groupby(qv.index.hour).sum()
byh = byh / byh.sum()
byd = qv.groupby(qv.index.dayofweek).sum()
byd = byd / byd.sum()
print("\nA. Share of 2021-23 quote volume (19 coins) by UTC hour (uniform = 4.17%):")
print("   " + " ".join(f"{h:02d}:{v * 100:4.2f}" for h, v in byh.items()))
order = byh.sort_values().index.tolist()
print(f"   thinnest 6 hours: {sorted(order[:6])}, busiest 6 hours: {sorted(order[-6:])}")
dn = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("   by weekday (uniform 14.3%): " + " ".join(f"{dn[d]}:{v * 100:4.1f}" for d, v in byd.items()))
for y in (2021, 2022, 2023):
    q = qv[qv.index.year == y]
    b = q.groupby(q.index.hour).sum(); b = b / b.sum()
    print(f"   {y} thinnest 6 hours: {sorted(b.sort_values().index[:6].tolist())}")

BLOCKS = {"00-03": range(0, 4), "04-07": range(4, 8), "08-11": range(8, 12), "12-15": range(12, 16),
          "16-19": range(16, 20), "20-23": range(20, 24)}


def blk(h):
    for b, r in BLOCKS.items():
        if h in r:
            return b


def table(lg, key, label):
    g = lg.groupby(key)
    T = pd.DataFrame({
        "n": g.size(),
        "win%": g.apply(lambda x: (x.ret > 0).mean() * 100),
        "avg%": g.ret.mean() * 100,
        "TS": g.apply(lambda x: int((x.why == 1).sum())),
        "TS%": g.apply(lambda x: (x.why == 1).mean() * 100),
        "TSavg%": g.apply(lambda x: x.ret[x.why == 1].mean() * 100 if (x.why == 1).any() else np.nan),
        "sum_rxs": g.contrib.sum(),
        "avg_hold_h": g.hold.mean(),
        "bp_per_h": g.apply(lambda x: x.ret.sum() / max(1, x.hold.sum()) * 1e4),
        "mkt%": g.mkt.mean() * 100,
    })
    print(f"\n{label}")
    print(T.round(2).to_string())
    return T


def enrich(lg, PP, ii, sy):
    lg = lg.copy()
    eb = lg.eb.astype(int).values; kk = lg.k.astype(int).values
    t0 = ii[eb]
    lg["hour"] = t0.hour; lg["dow"] = t0.dayofweek; lg["year"] = t0.year
    lg["block"] = [blk(h) for h in lg.hour]
    lg["wkend"] = np.where(lg.dow >= 5, "Sat-Sun", "Mon-Fri")
    lg["contrib"] = lg.ret * lg.amt
    lg["hold"] = lg.xb - lg.eb
    o = PP["o"].values[eb, kk]; ref = PP["ref"].values[eb, kk]
    lg["mkt"] = o <= ref * (1 - E.LIVE["depth"])
    lg["coin"] = [sy[k] for k in kk]
    return lg


# ------------------------------------------------------------------ B. live trades
e, lg = E.run(P, idx)
lg = lg[lg.xb.astype(int) < n - 1]
lg = enrich(lg, P, idx, syms)
print(f"\nB. live settings on 2021-23: x{e.iloc[-1]:.2f}, {len(lg)} trades closed before 2024, "
      f"{int((lg.why == 1).sum())} time stops, sum ret x size {lg.contrib.sum():+.2f}")
table(lg, "block", "B1. live trades by UTC hour block of the order placement")
table(lg, "hour", "B2. live trades by UTC hour")
lg["dname"] = [dn[d] for d in lg.dow]
table(lg, "dow", "B3. live trades by weekday (0=Mon)")
table(lg, "wkend", "B4. live trades weekend vs weekday")
for y in (2021, 2022, 2023):
    table(lg[lg.year == y], "block", f"B5. live trades by hour block, {y}")
table(lg, ["wkend", "block"], "B6. live trades by weekend x hour block")

# ------------------------------------------------------------------ C. standalone dip events (each coin alone)
parts = []
for j, s in enumerate(syms):
    Pj = {f: v[[s]] for f, v in P.items()}
    ej, lj = E.run(Pj, idx)
    lj = lj[lj.xb.astype(int) < n - 1].copy()
    lj["k"] = j
    parts.append(lj)
ev = enrich(pd.concat(parts, ignore_index=True), P, idx, syms)
ev["amt"] = 1.0; ev["contrib"] = ev.ret
print(f"\nC. standalone events (each coin alone, live rules, 2021-23): {len(ev)} trades, "
      f"{int((ev.why == 1).sum())} time stops, avg {ev.ret.mean() * 100:+.2f}%")
table(ev, "block", "C1. standalone events by UTC hour block")
table(ev, "hour", "C2. standalone events by UTC hour")
table(ev, "dow", "C3. standalone events by weekday (0=Mon)")
table(ev, "wkend", "C4. standalone events weekend vs weekday")
for y in (2021, 2022, 2023):
    table(ev[ev.year == y], "block", f"C5. standalone events by hour block, {y}")
    table(ev[ev.year == y], "wkend", f"C6. standalone events weekend vs weekday, {y}")
table(ev, ["wkend", "block"], "C7. standalone events by weekend x hour block")

# ------------------------------------------------------------------ D. how deep do dips go after the fill?
L = P["l"].values; R = P["ref"].values
eb = ev.eb.astype(int).values; kk = ev.k.astype(int).values
mn = np.array([np.nanmin(L[a:min(a + 6, n), k]) for a, k in zip(eb, kk)])
ev["mindd6"] = mn / R[eb, kk] - 1           # deepest low within the fill hour + next 5 hours, vs ref at entry
ev["reach14"] = ev.mindd6 <= (1 - 0.14) * (1 - E.EPS) - 1     # a limit at -14% would have filled
ev["reach16"] = ev.mindd6 <= (1 - 0.16) * (1 - E.EPS) - 1
g = ev.groupby("block")
print("\nD1. standalone events: deepest low in the fill hour + next 5h (vs ref) by hour block")
print(pd.DataFrame({"n": g.size(), "median_mindd6%": g.mindd6.median() * 100, "p25%": g.mindd6.quantile(0.25) * 100,
                    "reach-14%": g.reach14.mean() * 100, "reach-16%": g.reach16.mean() * 100}).round(2).to_string())
g = ev.groupby("wkend")
print(pd.DataFrame({"n": g.size(), "median_mindd6%": g.mindd6.median() * 100, "p25%": g.mindd6.quantile(0.25) * 100,
                    "reach-14%": g.reach14.mean() * 100, "reach-16%": g.reach16.mean() * 100}).round(2).to_string())

# D2: standalone runs with a deeper trigger only in one hour block / at weekends (each coin alone), vs live
hrs = idx.hour.values; wk = idx.dayofweek.values >= 5
masks = {**{b: np.isin(hrs, list(r)) for b, r in BLOCKS.items()}, "Sat-Sun": wk, "Mon-Fri": ~wk}


def standalone(depth_fn):
    out = []
    for j, s in enumerate(syms):
        Pj = {f: v[[s]] for f, v in P.items()}
        ej, lj = E.run(Pj, idx, depth=depth_fn())
        lj = lj[lj.xb.astype(int) < n - 1]
        out.append(lj)
    a = pd.concat(out, ignore_index=True)
    return a


print("\nD2. standalone (each coin alone), trades whose ORDER HOUR is in the mask: n, avg ret, sum ret, hours held;")
print("    live 12% everywhere vs the trigger deeper only inside the mask (sum ret counts only in-mask trades)")
base = ev
for mname, m in masks.items():
    row = [f"   {mname:8s} live: n {int(m[eb].sum()):4d} avg {base.ret[m[eb]].mean() * 100:+.2f}% "
           f"sum {base.ret[m[eb]].sum():+.2f} hold {base.hold[m[eb]].sum():6.0f}h"]
    for dd in (0.14, 0.16):
        dep = np.where(m, dd, 0.12)
        a = standalone(lambda: dep)
        ab = a.eb.astype(int).values
        mm = m[ab]
        row.append(f"| depth {dd:.0%}: n {int(mm.sum()):4d} avg {a.ret[mm].mean() * 100:+.2f}% sum {a.ret[mm].sum():+.2f} "
                   f"hold {(a.xb - a.eb)[mm].sum():6.0f}h (outside-mask sum {a.ret[~mm].sum():+.2f} vs "
                   f"{base.ret[~m[eb]].sum():+.2f})")
    print(" ".join(row))
