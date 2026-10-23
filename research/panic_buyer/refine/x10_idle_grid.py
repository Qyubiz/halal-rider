"""Lever "idle cash": pre-registered grid, pick on 2021-23 only, then one look at 2024-26 / since 2025 + stress.

PRE-REGISTRATION (written 22 Oct 2026 before any parking result was seen; not changed after the run)
-------------------------------------------------------------------------------------------------------
Seen before writing this: the dip bot's cash share and fill distances (x10_idle_measure.py) and the buy-and-hold
price change of each of the 19 coins per period (BTC x1.96 in 2024-26, x0.88 since 2025; many alts x0.1-0.5 since
2025). Because of that, fixed parking assets are limited to the a-priori choices BTC and BTC+ETH (the task's
suggestion); no fixed BNB/TRX parking (that would be hindsight). Hindsight parking in single coins is reported
separately in x10_idle_hindsight.py and labelled as such.

GRID: 4 dip-bot bases x (no parking + 7 parking engines x 8 mechanics) = 228 runs, 19 coins, normal costs.
  bases (12% dip under the 24h high close, 48h time stop, 24h cooldown):
     live 8x20% +1.5% | 2x50% +5% | 2x50% +3% | 1x100% +3%
  parking engines (all weights decided at 00:00 UTC from closes up to 23:00, held for the day):
     BTC                 always BTC
     BTC f20 / BTC f50   BTC only while its last close > its 20-day / 50-day average of hourly closes, else cash
     BTC+ETH f50         50/50, each with its own 50-day filter
     mom1-30 / mom3-30   top-1 / top-3 of the coins in the panel by 30-day return, only if that return > 0
     mom1-7              top-1 by 7-day return, only if > 0
  mechanics: (mode 1 watcher, armx 1%) (mode 1, 3%) (mode 0 open-only, 5%) (mode 0, 10%)
             x rebuy_wait {0, 24} hours; band 5% of equity throughout.

SELECTION RULE (2021-23 columns only):
  keep runs with calendar-2022 return >= 0% and 2021-23 worst drop >= -50%;
  pick the highest 2021-23 %/month (tie: shallower 2021-23 drop). None qualifies -> no pick.
Then ONE look at the pick's 2024-26 and since-2025 numbers, plus stress (LUNA/FTT/SRM added + every fill,
dip and parking, buys and sells, 0.2% worse) and its worst trade as % of the account; the same for the pick's
base without parking. "Parking adds" = the paired comparison pick vs its base without parking, per period.
Also reported (labelled HINDSIGHT): the grid's best 2024-26 %/month, and per base the best parking variant by
2024-26, with all four reports.
"""
import os
import time

import numpy as np
import pandas as pd

import x10_idle_lib as X

PARKS = {
    "BTC": dict(kind="fixed", assets={"BTCUSDT": 1.0}),
    "BTC f20": dict(kind="fixed", assets={"BTCUSDT": 1.0}, filt=20),
    "BTC f50": dict(kind="fixed", assets={"BTCUSDT": 1.0}, filt=50),
    "BTC+ETH f50": dict(kind="fixed", assets={"BTCUSDT": 0.5, "ETHUSDT": 0.5}, filt=50),
    "mom1-30": dict(kind="mom", days=30, top=1, absmom=True),
    "mom3-30": dict(kind="mom", days=30, top=3, absmom=True),
    "mom1-7": dict(kind="mom", days=7, top=1, absmom=True),
}
MECH = [dict(mode=m, armx=x, band=0.05, rebuy_wait=w) for (m, x) in ((1, 0.01), (1, 0.03), (0, 0.05), (0, 0.10))
        for w in (0, 24)]
MIN_2022, MAX_DD = 0.0, -0.50
COLS = ["base", "park", "mode", "armx", "rebuy_wait", "d_mo", "d_dd", "y2022", "d_mult", "t_mo", "t_dd", "t_mult",
        "s_mo", "s_dd", "s_mult", "y2021", "y2023", "y2024", "y2025", "y2026", "worst_acct", "d_cash", "t_cash",
        "d_park", "t_park", "pk_forced", "pk_watch_sells", "pk_buys", "pk_turnover", "pk_costs", "trades"]


def mech_name(pk):
    return "none" if pk is None else f"m{pk['mode']} x{pk['armx']:.2f} w{pk['rebuy_wait']}"


def run_grid():
    syms, idx, P = X.market(False)
    W = {p: X.weights_for(P, idx, s) for p, s in PARKS.items()}
    rows = []
    t0 = time.time()
    for bname, b in X.BASES.items():
        combos = [(None, None)] + [(p, pk) for p in PARKS for pk in MECH]
        for p, pk in combos:
            kw = {k: b[k] for k in X.BOT_KEYS}
            if pk:
                kw.update(pk)
            e, lg, aux, st = X.irun(P, idx, pw=None if p is None else W[p], **kw)
            r = X.summarize(e, lg, idx, aux, st if p else None)
            r.update(base=bname, park=p or "none", mode=pk["mode"] if pk else -1, armx=pk["armx"] if pk else np.nan,
                     rebuy_wait=pk["rebuy_wait"] if pk else -1)
            rows.append(r)
    print(f"{len(rows)} runs in {time.time() - t0:.0f}s")
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(X.OUT, "grid.csv"), index=False)
    return R


def spec_of(row):
    b = X.BASES[row["base"]]
    park = None if row["park"] == "none" else PARKS[row["park"]]
    pk = None if park is None else dict(mode=int(row["mode"]), armx=float(row["armx"]), band=0.05,
                                        rebuy_wait=int(row["rebuy_wait"]))
    return b, park, pk


def full_report(label, row):
    b, park, pk = spec_of(row)
    r, e, lg, aux = X.evaluate(b, park, pk)
    rs, _, _, _ = X.evaluate(b, park, pk, stress=True)
    print(X.fmt4(f"{label}: {row['base']} | park {row['park']} | {mech_name(pk)}", r, rs))
    return r, rs


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    R = run_grid()
    D = R[["base", "park", "mode", "armx", "rebuy_wait", "d_mo", "d_dd", "y2022"]]    # selection sees only these
    ok = D[(D.y2022 >= MIN_2022) & (D.d_dd >= MAX_DD)]
    print(f"\nqualify on 2021-23 (2022 >= 0%, drop >= -50%): {len(ok)} of {len(D)}")
    if len(ok):
        pick = ok.sort_values(["d_mo", "d_dd"], ascending=[False, False]).iloc[0]
        print(f"HONEST PICK (2021-23 only): {pick.base} | park {pick.park} | mode {pick['mode']} armx {pick.armx} "
              f"wait {pick.rebuy_wait}: 2021-23 {pick.d_mo*100:+.2f}%/mo drop {pick.d_dd*100:.1f}% 2022 {pick.y2022*100:+.1f}%")
        print("top 10 qualifying on 2021-23:")
        print(ok.sort_values("d_mo", ascending=False).head(10).round(4).to_string(index=False))
        print("\n==== ONE LOOK: the honest pick and its base without parking")
        prow = R.loc[pick.name]
        full_report("PICK", prow)
        brow = R[(R.base == prow.base) & (R.park == "none")].iloc[0]
        full_report("ITS BASE, NO PARKING", brow)
    else:
        print("no pick")

    print("\n==== Paired table: each base without parking vs its parking variants (19 coins, normal costs)")
    show = ["base", "park", "mode", "armx", "rebuy_wait", "d_mo", "d_dd", "y2022", "t_mo", "t_dd", "s_mo", "s_dd",
            "t_cash", "t_park", "pk_turnover", "pk_costs"]
    for bname in X.BASES:
        S = R[R.base == bname]
        print(f"\n-- {bname}")
        print(S[show].round(4).to_string(index=False))

    print("\n==== HINDSIGHT: best of the grid by 2024-26 %/month, and the best parking variant of each base")
    hb = R.sort_values("t_mo", ascending=False).iloc[0]
    full_report("HINDSIGHT BEST OF GRID", hb)
    for bname in X.BASES:
        S = R[(R.base == bname) & (R.park != "none")].sort_values("t_mo", ascending=False).iloc[0]
        full_report(f"HINDSIGHT best parking for {bname}", S)
        full_report(f"   {bname} without parking", R[(R.base == bname) & (R.park == "none")].iloc[0])
    print(f"\nMAX 2024-26 %/month of any grid run: {R.t_mo.max()*100:.2f}%")
