"""Live settings on fresh data: reproduce the handoff numbers (19 coins: 2024-26 x2.61, DD -16%; 2021-23 DD -29%),
then the cost model (idea 9): BNB fee discount and worse fills / slippage."""
import os

import pandas as pd

import engine as E
import study as S

syms, idx, P = S.market()
print(f"19 coins, {idx[0]:%Y-%m-%d} -> {idx[-1]:%Y-%m-%d %H:%M} UTC")
base, e, lg = S.evaluate({})
print(E.line("LIVE SETTINGS", base))
print("  per year: " + " ".join(f"{y}:{v*100:+.0f}%" for y, v in E.years(e).items()))
print(f"  since 2024: {base['t_trades']} trades, win {base['t_win']*100:.0f}%, avg {base['t_avg']*100:+.2f}%, worst {base['t_worst']*100:+.0f}%")
t = e[e.index >= E.DESIGN_END]
print(f"  RM6,000 -> RM{6000 * t.iloc[-1] / t.iloc[0]:,.0f} since 2024")
e.to_pickle(os.path.join(S.OUT, "baseline_eq.pkl"))
lg.assign(bought=idx[lg.eb.astype(int)], sold=idx[lg.xb.astype(int)], coin=[syms[int(k)] for k in lg.k]).to_csv(
    os.path.join(S.OUT, "baseline_trades.csv"), index=False)
m = E.lab.month_returns(e)
pd.DataFrame({"month": m.index.strftime("%Y-%m"), "ret": m.values}).to_csv(os.path.join(S.OUT, "baseline_months.csv"), index=False)

old = pd.read_csv(os.path.join(os.path.dirname(E.HERE), "results", "monthly_results.csv"))
old = old[old.coins.str.startswith("19 halal")].set_index("month").bot_return_pct / 100
new = pd.Series(m.values, m.index.strftime("%Y-%m"))
j = pd.concat([old.rename("handoff"), new.rename("now")], axis=1).dropna()
print(f"  monthly returns vs the handoff's monthly_results.csv: {len(j)} months, mean abs diff "
      f"{(j.handoff - j.now).abs().mean()*100:.2f} pts, largest {(j.handoff - j.now).abs().max()*100:.2f} pts")

b2, _, _ = S.evaluate({}, dead=True)
print(E.line("LIVE + LUNA/FTT/SRM", b2))

print("\n######## idea 9: cost model")
for label, kw in {"BNB pays fees (0.075%/side)": dict(fee=0.00075),
                  "fills 0.1% worse": dict(pen=0.001), "fills 0.2% worse": dict(pen=0.002),
                  "fills 0.5% worse": dict(pen=0.005),
                  "market-order slippage 0.3%": dict(slip=0.003),
                  "BNB fees + fills 0.2% worse": dict(fee=0.00075, pen=0.002)}.items():
    r, _, _ = S.evaluate({}, **kw)
    print(E.line(label, r))
