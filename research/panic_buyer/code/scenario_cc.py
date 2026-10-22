"""Recompute the capital scenarios with the card as capital: owe capital + one-time 0.75% fee, no monthly interest.
Each month all profit above the starting capital is withdrawn, pays Claude (RM82) + server (RM20), and whatever is left
repays the card. Shortfalls in weak months come from your own pocket. Input: monthly_results.csv from big_backtest.py."""
import sys
import numpy as np, pandas as pd

OUT = sys.argv[1]
AI_RM, SERVER_RM, FEE_ONCE = 82.0, 20.0, 0.0075
CAPITALS = [1000, 2000, 3000, 4000, 5000, 6000]
mr = pd.read_csv(f"{OUT}/monthly_results.csv")

def scenario(r, cap):
    months = len(r)
    comp = cap * float(np.prod(1 + r))
    owed0 = cap * (1 + FEE_ONCE)
    eqv, owed, paid_month = float(cap), owed0, None
    wd, covered, pocket, free = [], 0, 0.0, 0.0
    for i, x in enumerate(r):
        eqv *= 1 + x
        w = max(0.0, eqv - cap); eqv -= w; wd.append(w)
        left = w - AI_RM - SERVER_RM
        if left >= 0:
            covered += 1
            rep = min(owed, left); owed -= rep; free += left - rep
            if owed <= 1e-9 and paid_month is None:
                paid_month = i + 1
        else:
            pocket += -left
    wd = np.array(wd)
    avg_left = wd.mean() - AI_RM - SERVER_RM
    if paid_month:
        repaid = f"month {paid_month}"
    elif avg_left > 0:
        repaid = f"not yet; about {months + int(np.ceil(owed / avg_left))} months at this pace"
    else:
        repaid = "never at this pace (profit below costs)"
    return {"capital_rm": cap, "card_fee_once_rm": round(cap * FEE_ONCE, 2), "card_to_repay_rm": round(owed0, 2),
            "reinvest_final_rm": round(comp), "reinvest_total_gain_rm": round(comp - cap),
            "reinvest_avg_gain_per_month_rm": round((comp - cap) / months),
            "reinvest_avg_gain_per_month_pct": round(((comp / cap) ** (1 / months) - 1) * 100, 2),
            "withdraw_total_rm": round(wd.sum()), "withdraw_avg_per_month_rm": round(wd.mean()),
            "withdraw_avg_per_month_pct": round(wd.mean() / cap * 100, 2),
            "costs_per_month_rm (Claude 82 + server 20)": AI_RM + SERVER_RM,
            "avg_left_per_month_after_costs_rm": round(avg_left),
            "months_profit_covered_costs": covered, "months": months,
            "paid_from_own_pocket_total_rm": round(pocket),
            "card_repaid_by_end_rm": round(owed0 - owed), "card_still_owed_end_rm": round(owed),
            "card_fully_repaid": repaid,
            "free_cash_to_you_total_rm": round(free), "money_still_in_bot_end_rm": round(eqv),
            "your_net_result_rm": round(eqv + free - owed - pocket)}

rows = []
for coins, g in mr.groupby("coins", sort=False):
    for label, start in (("2024-01 to 2026-10 (34 months, recent)", "2024-01"), ("2021-01 to 2026-10 (70 months, incl. 2021 bull run)", "2021-01")):
        r = g[g.month >= start].bot_return_pct.values / 100
        for cap in CAPITALS:
            rows.append({"coins": coins, "period": label, **scenario(r, cap)})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/results_by_capital.csv", index=False)

a = pd.read_csv(f"{OUT}/assumptions.csv")
a = a[~a.item.isin(["monthly costs", "card repayment", "your_net_result_rm"])]
a = pd.concat([a, pd.DataFrame([
    ("capital", "taken from the credit card: you owe the capital + a one-time 0.75% fee, no monthly interest"),
    ("monthly costs", "Claude Pro RM82 + Lightsail server RM20 (US$5 at ~4.08)"),
    ("card repayment", "profit left after the monthly costs repays the card; shortfalls in weak months are paid from your own pocket"),
    ("your_net_result_rm", "money still in the bot + free cash received - card still owed - own pocket paid"),
], columns=["item", "assumption"])])
a.to_csv(f"{OUT}/assumptions.csv", index=False)

show = ["capital_rm", "reinvest_total_gain_rm", "reinvest_avg_gain_per_month_rm", "withdraw_avg_per_month_rm",
        "avg_left_per_month_after_costs_rm", "months_profit_covered_costs", "paid_from_own_pocket_total_rm",
        "card_still_owed_end_rm", "card_fully_repaid", "your_net_result_rm"]
for (coins, per), g in res.groupby(["coins", "period"], sort=False):
    print(f"\n## {coins} | {per}")
    print(g[show].to_string(index=False))
