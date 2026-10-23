"""Lens "crowding", motivation part 2 (2021-23 trades only; nothing from 2024 onward is printed).

C. Order of the fills inside crowded hours (1-minute data, r2_crowding_fm): if the bot cancelled its other resting
   buys once F had filled in the hour, which trades would it have skipped, and how did they do? And what would a
   market buy of the same coin at the next hour's open have done instead (crude: the coin stays a candidate)?
D. Rolling cap: fills that came after F others in the trailing W hours.
E. Second leg as its own money: return of a leg-2 buy at ref*(1-d2) (sold with the position), and how many
   leg-2 buys are open at once (what a shared cash reserve would need).
"""
import numpy as np
import pandas as pd

import engine as E
import study as S
from r2_crowding_fm import fill_minutes

syms, idx, P = S.market(False)
e, lg = E.run(P, idx)
for col in ("eb", "xb", "k"):
    lg[col] = lg[col].astype(int)
d = lg[idx[lg.xb.values] < E.DESIGN_END].copy().reset_index(drop=True)
O, H, L, C, REF = (P[f].values for f in ("o", "h", "l", "cf", "ref"))
FM = fill_minutes(syms, idx, P, tag="h19")
d["fm"] = FM[d.eb.values, d.k.values]
print(f"2021-23 trades {len(d)}; fill minute known for {int(d.fm.notna().sum())}; at the hour open {int((d.fm == -1).sum())}")
d["x"] = d.ret * d.amt
d["tmin"] = d.eb * 60 + d.fm.clip(lower=-0.5)        # fill time in minutes (open fills first)
FEE, EPS, SL = E.FEE, E.EPS, E.SLIP_MAJOR


def crude_trade(t0, k, px, tp=0.015, hm=48):
    """One trade bought at px in hour t0 (hold rules of the live bot), return on the money."""
    for t in range(t0 + 1, len(idx)):
        if np.isnan(O[t, k]):
            continue
        tpp = px * (1 + tp)
        if O[t, k] >= tpp:
            return (1 - FEE) * (1 - FEE) * O[t, k] / px - 1
        if H[t, k] >= tpp * (1 + EPS):
            return (1 - FEE) * (1 - FEE) * tpp / px - 1
        if t - t0 >= hm:
            return (1 - FEE) * (1 - FEE) * C[t, k] * (1 - SL) / px - 1
    return np.nan


def nexthour(i):
    """Crude alternative for a skipped fill: market buy at the next hour's open if still under the trigger."""
    t, k = d.eb[i] + 1, d.k[i]
    lim = REF[t, k] * (1 - 0.12)
    if np.isnan(O[t, k]) or O[t, k] > lim:
        return 0.0, False
    return crude_trade(t, k, O[t, k] * (1 + SL)), True


print("\nC. per-hour cap: trades that filled after the F-th fill of their hour (same-minute ties allowed through)")
for F in (2, 3, 4, 5, 6):
    skip = np.zeros(len(d), bool)
    for t, g in d.groupby("eb"):
        if len(g) <= F:
            continue
        mF = np.sort(g.fm.values)[F - 1]
        skip[g.index[g.fm.values > mF]] = True
    s, a = d[skip], d[~skip]
    alt = [nexthour(i) for i in s.index]
    altr = np.array([r for r, _ in alt]); altb = np.array([b for _, b in alt])
    print(f"  F={F}: skipped {len(s):3d} trades (avg {s.ret.mean() * 100:+.2f}%, time stops {(s.why == 1).mean() * 100:4.1f}%, "
          f"sum ret*amt {s.x.sum():+.3f}) | kept {len(a)} (avg {a.ret.mean() * 100:+.2f}%, time stops "
          f"{(a.why == 1).mean() * 100:4.1f}%) | next-hour market buy instead: {int(altb.sum())} bought, "
          f"sum {(altr * s.amt.values).sum():+.3f}, avg {np.nanmean(altr[altb]) * 100 if altb.any() else np.nan:+.2f}%")

print("\nfill rank inside hours with >= 5 fills (rank 1 = first):")
big = d[d.eb.map(d.groupby("eb").size()) >= 5].copy()
big["rank"] = big.groupby("eb").fm.rank(method="min")
for lo, hi in ((1, 2), (3, 4), (5, 6), (7, 8)):
    s = big[(big["rank"] >= lo) & (big["rank"] <= hi)]
    print(f"  rank {lo}-{hi}: {len(s):3d} trades avg {s.ret.mean() * 100:+.2f}% time stops {(s.why == 1).mean() * 100:4.1f}% "
          f"sum {s.x.sum():+.3f}")

print("\nD. rolling cap: trades with >= F earlier fills in the trailing W hours (incl. earlier minutes of the hour)")
tm = np.sort(d.tmin.values)
for W in (6, 12, 24):
    prior = np.array([int(np.searchsorted(tm, x, "left") - np.searchsorted(tm, x - 60 * W, "left")) for x in d.tmin.values])
    for F in (4, 6, 8):
        s = d[prior >= F]
        print(f"  W={W:2d}h F={F}: {len(s):3d} trades avg {s.ret.mean() * 100:+.2f}% time stops {(s.why == 1).mean() * 100:4.1f}% "
              f"sum {s.x.sum():+.3f} | others avg {d[prior < F].ret.mean() * 100:+.2f}%")

print("\nE. second leg at ref*(1-d2) as its own money (sold with the position at the ladder's exit; crude)")


def leg2(i, d2, w=0.5, tp=0.015, hm=48):
    t0, k, en = d.eb[i], d.k[i], d.entry[i]
    L2 = REF[t0, k] * (1 - d2)
    f2 = -1.0; t2 = -1
    if O[t0, k] <= L2:
        f2 = O[t0, k] * (1 + SL); t2 = t0
    elif L[t0, k] <= L2 * (1 - EPS):
        f2 = L2; t2 = t0
    last = t0
    for t in range(t0 + 1, len(idx)):
        if np.isnan(O[t, k]):
            continue
        if f2 < 0 and L[t, k] <= L2 * (1 - EPS):
            f2 = L2; t2 = t; last = t
        elif t > last:
            avg = 1.0 / (w / en + ((1 - w) / f2 if f2 > 0 else 0)) if f2 > 0 else en
            tpp = avg * (1 + tp)
            px = O[t, k] if O[t, k] >= tpp else (tpp if H[t, k] >= tpp * (1 + EPS) else -1)
            if px > 0:
                return (px * (1 - FEE) ** 2 / f2 - 1 if f2 > 0 else np.nan), t2, t, 0
        if t - t0 >= hm:
            return (C[t, k] * (1 - SL) * (1 - FEE) ** 2 / f2 - 1 if f2 > 0 else np.nan), t2, t, 1
    return np.nan, t2, -1, -1


for d2 in (0.16, 0.18, 0.20, 0.24):
    res = [leg2(i, d2) for i in range(len(d))]
    r2 = np.array([r[0] for r in res]); t2 = np.array([r[1] for r in res]); tx = np.array([r[2] for r in res])
    why = np.array([r[3] for r in res])
    m = ~np.isnan(r2)
    # concurrency of open leg-2 positions (hour by hour)
    conc = np.zeros(len(idx), int)
    for a, b in zip(t2[m], tx[m]):
        conc[a:b] += 1
    print(f"  d2={d2:.0%}: leg-2 filled {int(m.sum()):3d} of {len(d)} | leg-2 avg {np.mean(r2[m]) * 100:+.2f}% "
          f"win {np.mean(r2[m] > 0) * 100:.0f}% | of these the position hit the time stop {np.mean(why[m] == 1) * 100:.0f}% "
          f"(leg-2 avg there {np.mean(r2[m & (why == 1)]) * 100:+.1f}%) | max leg-2 positions open at once {conc.max()}, "
          f"hours with >=3 open {int((conc >= 3).sum())}")
    s = d[m]
    print(f"           same trades, live leg-1 result: avg {s.ret.mean() * 100:+.2f}%, time stops {(s.why == 1).mean() * 100:.0f}%")
