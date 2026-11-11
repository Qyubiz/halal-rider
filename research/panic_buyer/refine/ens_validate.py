"""Agent 3 validation: the recommended architecture and its guardrails vs the two modes, (1) per period on the 19-coin
and 22-coin panels, normal fills and STRESS (real bot timing: market-buy take-profit at once, cooldown +1h; every buy
AND sell 0.2% worse; +LUNA/FTT/SRM), and (2) in the market's worst historical regimes. Regimes are picked from the data,
not by hand: the 6 worst non-overlapping 30-day returns of an equal-weight index of the 19 coins (daily closes,
windows >= 45 days apart), each window extended 14 days to show the rebound. Also prints guardrail invariants.

Usage (from research/panic_buyer/refine): PB_DATA=... PB_OUT=... python3 ens_validate.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.setdefault("PB_OUT", os.path.expanduser("~/pb_out_ens/research"))
os.makedirs(OUT, exist_ok=True)
import ens_lib as L  # noqa: E402

ARCH = ([L.MODE90, L.MODE10], dict(weights=[0.3, 0.7], rb=1))
PICK = dict(killd=0.15, killm=0.25, hs=0.25, cln=2, clT=72)            # ens_guard.py pick (2021-23 rule)
DESIGNS = {
    "90% mode": ([L.MODE90], {}),
    "10% mode": ([L.MODE10], {}),
    "ARCH 30/70 (no new guardrails)": ARCH,
    "ARCH + PICK X15 Y25 S25 N2 T72": (ARCH[0], dict(ARCH[1], **PICK)),
    "ARCH + strictest X15 Y20 S25 Z10 N2 T72": (ARCH[0], dict(ARCH[1], killd=0.15, killm=0.20, hs=0.25, zcap=0.10,
                                                                cln=2, clT=72)),
    "ARCH + kill 30d 30% only (post-hoc)": (ARCH[0], dict(ARCH[1], killm=0.30)),
    "ARCH + stop 20% + Z 10% only (post-hoc)": (ARCH[0], dict(ARCH[1], hs=0.20, zcap=0.10)),
}
STRESS = dict(mtp_now=True, cd_add=1, pen=0.002, pens=0.002)

rows, curves = [], {}
for name, (sv, kw) in DESIGNS.items():
    for tag, dead, extra in (("19", False, {}), ("22", True, {}), ("22 STRESS", True, STRESS)):
        syms, idx, P = L.market(dead)
        e, lg, nkd, nkm = L.erun(P, idx, sv, liquidate=dead, **dict(kw, **extra))
        r = L.summarize(e, lg, idx)
        r.update(design=name, panel=tag, kills_d=nkd, kills_m=nkm)
        hsv = kw.get("hs", 0.0)
        acct = lg.ret * lg.amt / lg.eq0
        r["inv_worst_trade_acct"] = float(acct.min()) * 100
        r["inv_worst_stopped_ret"] = float(lg.ret[lg.why == 4].min()) * 100 if (lg.why == 4).any() else np.nan
        r["inv_max_order_acct"] = float((lg.amt / lg.eq0).max()) * 100
        rows.append(r)
        if tag == "22":
            curves[name] = e
R = pd.DataFrame(rows)
R.to_csv(os.path.join(OUT, "validate.csv"), index=False)
pd.set_option("display.width", 300)
print("Per period (trades by entry hour; WR includes positions open at the end, marked at the last close)")
print(L.table(R))
print("\nGuardrail invariants (all periods): worst trade as % of account, worst stopped trade's own return, "
      "largest order as % of account, kill-switch firings")
print(R[["design", "panel", "inv_worst_trade_acct", "inv_worst_stopped_ret", "inv_max_order_acct", "kills_d",
         "kills_m"]].to_string(index=False, float_format=lambda v: f"{v:7.1f}"))

# ---------------- worst regimes from the data
syms, idx, P = L.market(False)
r1 = P["cf"].pct_change(fill_method=None)
ix = (1 + r1.mean(axis=1, skipna=True).fillna(0)).cumprod().resample("D").last()
r30 = ix / ix.shift(30) - 1
win = []
for d, v in r30.sort_values().items():
    if np.isnan(v):
        continue
    if all(abs((d - w[0]).days) >= 45 for w in win):
        win.append((d, v))
    if len(win) == 6:
        break
print("\nWorst regimes: 30-day windows of the equal-weight 19-coin index, +14 days after. Cells: design return over the "
      "window+14d / max drop inside it (22-coin panel, normal fills)")
out = []
for end, v in sorted(win):
    a, b = end - pd.Timedelta(days=30), end + pd.Timedelta(days=14)
    row = {"window": f"{a:%Y-%m-%d}..{end:%Y-%m-%d}", "index 30d": v * 100}
    for name, e in curves.items():
        s = e[(e.index >= a) & (e.index < b + pd.Timedelta(days=1))]
        s0 = e[e.index < a].iloc[-1] if (e.index < a).any() else 1.0
        x = np.concatenate([[s0], s.values])
        row[name] = f"{(x[-1]/x[0]-1)*100:+6.1f}/{((x/np.maximum.accumulate(x))-1).min()*100:6.1f}"
    out.append(row)
W = pd.DataFrame(out)
W.to_csv(os.path.join(OUT, "validate_regimes.csv"), index=False)
for _, rr in W.iterrows():
    print(f"\n{rr['window']}  index {rr['index 30d']:+.1f}%")
    for name in curves:
        print(f"   {name:44s} {rr[name]}")

# ---------------- untouched 2018-20 holdout (engine.load_pre, PB_HOLDOUT): never used for any choice above
import engine as E  # noqa: E402
try:
    HD = E.load_pre()
except Exception as ex:      # holdout data missing
    HD = None
    print(f"\nholdout skipped: {ex}")
if HD:
    hs_, hidx, HP = E.panel(HD)
    print(f"\n2018-20 holdout: {len(hs_)} coins, {hidx[0]} -> {hidx[-1]} (normal fills / buys+sells 0.2% worse with bot "
          f"timing). Cells: %/mo, max drop, win rate, PF, worst month, trades; plus the COVID crash 2020-02-20..2020-04-01")
    hrows = []
    for name, (sv, kw) in DESIGNS.items():
        for tag, extra in (("normal", {}), ("stress", STRESS)):
            e, lg, nkd, nkm = L.erun(HP, hidx, sv, **dict(kw, **extra))
            m, mo, dd, wm, wd = L.period(e, E.H_START, E.H_END)
            tb = hidx[lg.eb.astype(int).values]
            sel = (tb >= E.H_START) & (tb < E.H_END)
            x = lg[sel]
            acct = (x.ret * x.amt / x.eq0).values
            a, b = pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-02")
            s = e[(e.index >= a) & (e.index < b)]
            s0 = e[e.index < a].iloc[-1]
            xx = np.concatenate([[s0], s.values])
            hrows.append(dict(design=name, fills=tag, pct_mo=mo * 100, max_dd=dd * 100,
                              WR=float((x.ret > 0).mean()) * 100, PF=L.pf(acct), worst_mo=wm * 100, trades=len(x),
                              covid_ret=(xx[-1] / xx[0] - 1) * 100,
                              covid_dd=((xx / np.maximum.accumulate(xx)) - 1).min() * 100, kills=nkd + nkm))
    H = pd.DataFrame(hrows)
    H.to_csv(os.path.join(OUT, "validate_holdout.csv"), index=False)
    print(H.to_string(index=False, float_format=lambda v: f"{v:7.1f}"))
