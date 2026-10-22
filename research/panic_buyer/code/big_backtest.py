"""Panic Buyer on a 150+ coin 'halal' universe: CSV results for capital RM1k-6k, compounding vs withdraw-all-profits,
with monthly costs (AI, server, credit-card charge) and credit-card payback."""
import os, sys, glob, json
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np, pandas as pd
from numba import njit
import universe as U

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
FEE = 0.001
AI_RM, SERVER_RM, CC_RATE = 82.0, 20.0, 0.0075
CAPITALS = [1000, 2000, 3000, 4000, 5000, 6000]
MIN_MEDIAN_DAILY_USD = float(os.environ.get("MIN_VOL", "1000000"))

@njit(cache=True)
def kernel(o, h, l, c, ref, depth, N, fmax, tp, hm, cd, eps, fee, slip):
    """Same logic as lab.dyn_kernel (validated against the live bot), with per-coin slippage."""
    n, K = c.shape
    units = np.zeros(K); entry = np.zeros(K); amt0 = np.zeros(K)
    ebar = np.full(K, -1, np.int64); okbar = np.zeros(K, np.int64)
    lastc = np.full(K, np.nan); dist = np.empty(K)
    cash = 1.0; eq = np.empty(n); eq_prev = 1.0
    maxlog = 600000; lg = np.zeros((maxlog, 7)); nl = 0
    for t in range(n):
        for k in range(K):
            dist[k] = 1e9
            if units[k] > 0.0 or t < okbar[k] or np.isnan(o[t, k]) or np.isnan(ref[t, k]): continue
            dist[k] = o[t, k] / (ref[t, k] * (1.0 - depth)) - 1.0
        order = np.argsort(dist)
        nsel = 0
        for q in range(K):
            if dist[order[q]] < 1e8 and nsel < N: nsel += 1
        if nsel > 0 and cash > 1e-9:
            per = min(fmax * eq_prev, cash / nsel)
            for q in range(nsel):
                k = order[q]
                lim = ref[t, k] * (1.0 - depth)
                fill = -1.0
                if o[t, k] <= lim: fill = o[t, k] * (1.0 + slip[k])
                elif l[t, k] <= lim * (1.0 - eps): fill = lim
                if fill > 0.0 and cash > 1e-9:
                    a = min(per, cash)
                    cash -= a; units[k] = a * (1.0 - fee) / fill
                    entry[k] = fill; amt0[k] = a; ebar[k] = t
        for k in range(K):
            if units[k] > 0.0 and ebar[k] < t and not np.isnan(o[t, k]):
                e = entry[k]; tpp = e * (1.0 + tp); px = -1.0; why = 0.0
                if o[t, k] >= tpp: px = o[t, k]
                elif h[t, k] >= tpp * (1.0 + eps): px = tpp
                elif t - ebar[k] >= hm: px = c[t, k] * (1.0 - slip[k]); why = 1.0
                if px > 0.0:
                    proceeds = units[k] * px * (1.0 - fee); cash += proceeds
                    if nl < maxlog:
                        lg[nl, 0] = ebar[k]; lg[nl, 1] = t; lg[nl, 2] = k; lg[nl, 3] = why
                        lg[nl, 4] = proceeds / amt0[k] - 1.0; lg[nl, 5] = amt0[k]; lg[nl, 6] = e; nl += 1
                    units[k] = 0.0; okbar[k] = t + cd
        v = cash
        for k in range(K):
            if not np.isnan(c[t, k]): lastc[k] = c[t, k]
            if units[k] > 0.0: v += units[k] * lastc[k]
        eq[t] = v; eq_prev = v
    return eq, lg[:nl].copy(), units, lastc

# ------------------------------------------------------------------ universe + liquidity filter
pm = U.pass_map(); ex = json.load(open("/home/claude/bt/universe/exinfo.json"))
status = {s["baseAsset"]: s["status"] for s in ex["symbols"] if s["quoteAsset"] == "USDT"}
rows, D = [], {}
for p in sorted(glob.glob(f"{DATA}/*.csv")):
    sym = os.path.basename(p)[:-4]; base = sym[:-4]
    if base not in pm: continue
    d = pd.read_csv(p, index_col=0, parse_dates=True)
    d = d[d.index >= "2021-01-01"]
    if len(d) < 24 * 60:
        rows.append(dict(symbol=sym, category=pm[base], binance_status=status.get(base, "?"), first=None, last=None,
                         median_daily_usd=0, used="no", reason="less than 60 days of data since 2021")); continue
    dv = d.qv.resample("1D").sum(); dv = dv[dv > 0]
    med = float(dv.median())
    ok = med >= MIN_MEDIAN_DAILY_USD
    rows.append(dict(symbol=sym, category=pm[base], binance_status=status.get(base, "?"), first=d.index[0], last=d.index[-1],
                     median_daily_usd=round(med), used="yes" if ok else "no",
                     reason="" if ok else f"median daily volume below ${MIN_MEDIAN_DAILY_USD/1e6:.1f}M"))
    if ok: D[sym] = d
coins = pd.DataFrame(rows)
excl = pd.DataFrame([dict(symbol=t + "USDT", reason=w) for t, w in U.excluded_map().items() if t in status])
print(f"coins tested: {len(D)} (of {len(coins)} screened with data); delisted among them: {sum(status[s[:-4]] != 'TRADING' for s in D)}")

# ------------------------------------------------------------------ run
H19 = {"BTC", "ETH", "SOL", "XRP", "BNB", "DOGE", "ADA", "TRX", "LINK", "AVAX", "SUI", "LTC", "BCH", "HBAR", "DOT", "NEAR",
       "ARB", "FET", "RENDER", "RNDR"}
liq_rank = coins[coins.used == "yes"].set_index("symbol").median_daily_usd.rank(ascending=False)

def run(symset):
    syms = sorted(symset)
    idx = pd.date_range(min(D[s].index[0] for s in syms), max(D[s].index[-1] for s in syms), freq="1h")
    P = {f: np.column_stack([D[s][f].reindex(idx).values for s in syms]).astype(np.float64) for f in ("o", "h", "l", "c")}
    Cf = pd.DataFrame(P["c"], index=idx).ffill().values
    ref = pd.DataFrame(P["c"], index=idx).rolling(24, min_periods=24).max().shift(1).values
    slip = np.array([0.0002 if liq_rank[s] <= 50 else 0.001 for s in syms])
    eq, lg, units, lastc = kernel(P["o"], P["h"], P["l"], Cf, ref, 0.12, 8, 0.20, 0.015, 48, 24, 0.001, FEE, slip)
    stuck = [(syms[k], round(units[k] * lastc[k] / eq[-1] * 100, 1)) for k in range(len(syms)) if units[k] > 0]
    tr = pd.DataFrame(lg, columns=["eb", "xb", "k", "why", "ret", "amt", "entry"])
    prev = np.concatenate([[1.0], eq[:-1]])
    tr = pd.DataFrame({"bought_utc": idx[tr.eb.astype(int)], "sold_utc": idx[tr.xb.astype(int)], "coin": [syms[int(k)] for k in tr.k],
                       "buy_price": tr.entry, "result_pct": (tr.ret * 100).round(3),
                       "exit": np.where(tr.why > 0, "time limit 48h", "take-profit +1.5%"),
                       "order_share_of_account_pct": (tr.amt / prev[tr.eb.astype(int)] * 100).round(2)})
    return pd.Series(eq, idx), tr, stuck

by_vol = list(coins[coins.used == "yes"].sort_values("median_daily_usd", ascending=False).symbol)
UNIVERSES = {"Top 150 screened halal coins by volume": set(by_vol[:150]),
             f"All {len(D)} screened halal coins": set(D),
             "19 halal-rider coins (live bot list)": {s for s in D if s[:-4] in H19}}
RUNS = {}
for name, ss in UNIVERSES.items():
    E, tr, stuck = run(ss)
    RUNS[name] = (E, tr)
    y = E.resample("YE").last(); y = pd.concat([pd.Series([1.0], [E.index[0]]), y]).pct_change().dropna()
    tt = tr[tr.bought_utc >= "2024-01-01"]
    d24 = E[E.index >= "2024-01-01"]; d21 = E[E.index < "2024-01-01"]
    print(f"\n== {name}: {len(ss)} coins | still holding at end: {stuck}")
    print("   per year:", " ".join(f"{k.year}:{v*100:+.0f}%" for k, v in y.items()))
    print(f"   trades {len(tr)} total, since 2024 {len(tt)/34:.0f}/month, win {np.mean(tt.result_pct>0)*100:.0f}%, avg {tt.result_pct.mean():+.2f}%, worst {tt.result_pct.min():+.0f}%")
    print(f"   biggest drop 2024-26 {(d24/d24.cummax()-1).min()*100:.0f}%, 2021-23 {(d21/d21.cummax()-1).min()*100:.0f}%")
RUNS["Top 150 screened halal coins by volume"][1].to_csv(f"{OUT}/trades_top150_coins.csv", index=False)
RUNS[f"All {len(D)} screened halal coins"][1].to_csv(f"{OUT}/trades_all_{len(D)}_coins.csv", index=False)
RUNS["19 halal-rider coins (live bot list)"][1].to_csv(f"{OUT}/trades_19_coins.csv", index=False)
cmp_rows = []
for n in (20, 50, 100, 150, len(D)):
    E_, tr_, stuck_ = run(set(by_vol[:n]))
    y_ = E_.resample("YE").last(); y_ = pd.concat([pd.Series([1.0], [E_.index[0]]), y_]).pct_change().dropna()
    t_ = E_[E_.index >= "2024-01-01"]
    row = {"coins": f"top {n} by volume" if n < len(D) else f"all {n}", **{f"year_{k.year}_pct": round(v * 100, 1) for k, v in y_.items()},
           "jan2024_oct2026_multiple": round(t_.iloc[-1] / t_.iloc[0], 3), "biggest_drop_2024_26_pct": round((t_ / t_.cummax() - 1).min() * 100, 1),
           "trades_per_month_2024_26": round(len(tr_[tr_.bought_utc >= "2024-01-01"]) / 34)}
    cmp_rows.append(row)
for name in ("19 halal-rider coins (live bot list)",):
    E_, tr_ = RUNS[name]
    y_ = E_.resample("YE").last(); y_ = pd.concat([pd.Series([1.0], [E_.index[0]]), y_]).pct_change().dropna()
    t_ = E_[E_.index >= "2024-01-01"]
    cmp_rows.append({"coins": "19 halal-rider coins", **{f"year_{k.year}_pct": round(v * 100, 1) for k, v in y_.items()},
                     "jan2024_oct2026_multiple": round(t_.iloc[-1] / t_.iloc[0], 3), "biggest_drop_2024_26_pct": round((t_ / t_.cummax() - 1).min() * 100, 1),
                     "trades_per_month_2024_26": round(len(tr_[tr_.bought_utc >= "2024-01-01"]) / 34)})
cmpdf = pd.DataFrame(cmp_rows); cmpdf.to_csv(f"{OUT}/coin_count_comparison.csv", index=False)
print("\n" + cmpdf.to_string(index=False))
coins.to_csv(f"{OUT}/coins_used.csv", index=False)
excl.to_csv(f"{OUT}/coins_excluded.csv", index=False)

def monthly(e, start):
    e = e[e.index >= start]
    m = e.resample("ME").last()
    m = pd.concat([pd.Series([e.iloc[0]], [e.index[0] - pd.Timedelta(seconds=1)]), m])
    r = m.pct_change().dropna(); r.index = r.index.strftime("%Y-%m")
    return r

def scenario(r, cap, card=True):
    """Compounding vs withdraw-all-profit (profits above the starting capital are taken out each month; after a losing
    month nothing is taken until the account is back to the starting capital). Costs: AI + server every month; with
    card=True the capital is borrowed on the credit card at 0.75% of the borrowed amount per month while any of it is
    owed, and profit left after costs repays the card. Shortfalls are paid from your own pocket."""
    months = len(r)
    comp = cap * float(np.prod(1 + r.values))
    eqv, debt, paid_month = float(cap), float(cap) if card else 0.0, None
    wd, covered, pocket, free, charges = [], 0, 0.0, 0.0, 0.0
    for i, x in enumerate(r.values):
        eqv *= 1 + x
        w = max(0.0, eqv - cap); eqv -= w; wd.append(w)
        charge = CC_RATE * cap if debt > 1e-9 else 0.0
        charges += charge
        left = w - (AI_RM + SERVER_RM + charge)
        if left >= 0:
            covered += 1
            rep = min(debt, left); debt -= rep; free += left - rep
            if card and debt <= 1e-9 and paid_month is None: paid_month = i + 1
        else:
            pocket += -left
    wd = np.array(wd)
    net = eqv - debt - pocket + free - (0.0 if card else cap)
    return {"funding": "credit card 0.75%/month" if card else "own money",
            "capital_rm": cap,
            "reinvest_final_rm": round(comp), "reinvest_total_gain_rm": round(comp - cap),
            "reinvest_total_gain_pct": round((comp / cap - 1) * 100, 1),
            "reinvest_avg_gain_per_month_pct": round(((comp / cap) ** (1 / months) - 1) * 100, 2),
            "reinvest_avg_gain_per_month_rm": round((comp - cap) / months),
            "withdraw_total_rm": round(wd.sum()), "withdraw_avg_per_month_rm": round(wd.mean()),
            "withdraw_avg_per_month_pct": round(wd.mean() / cap * 100, 2),
            "withdraw_best_month_rm": round(wd.max()), "months_with_nothing_to_withdraw": int((wd < 0.5).sum()),
            "months": months,
            "monthly_costs_rm": round(AI_RM + SERVER_RM + (CC_RATE * cap if card else 0.0), 2),
            "months_profit_covered_all_costs": covered,
            "paid_from_own_pocket_total_rm": round(pocket),
            "card_charges_paid_total_rm": round(charges),
            "card_still_owed_end_rm": round(debt),
            "card_fully_repaid_in_month": (paid_month if paid_month else "not repaid in this period") if card else "-",
            "free_cash_to_you_total_rm": round(free),
            "money_still_in_bot_end_rm": round(eqv),
            "your_net_result_rm": round(net)}

out_rows, months_tables = [], []
for uname, (E, tr) in RUNS.items():
    for label, start in (("2024-01 to 2026-10 (34 months, recent)", "2024-01-01"), ("2021-01 to 2026-10 (70 months, incl. 2021 bull run)", "2021-01-01")):
        r = monthly(E, start)
        for card in (True, False):
            for cap in CAPITALS:
                out_rows.append({"coins": uname, "period": label, **scenario(r, cap, card)})
    r = monthly(E, "2021-01-01")
    mt = pd.DataFrame({"coins": uname, "month": r.index, "bot_return_pct": (r.values * 100).round(2)})
    for cap in CAPITALS:
        eqv, col = float(cap), []
        for x in r.values:
            eqv *= 1 + x; w = max(0.0, eqv - cap); eqv -= w; col.append(round(w, 2))
        mt[f"withdraw_rm_capital_{cap}"] = col
    months_tables.append(mt)
res = pd.DataFrame(out_rows)
res.to_csv(f"{OUT}/results_by_capital.csv", index=False)
pd.concat(months_tables).to_csv(f"{OUT}/monthly_results.csv", index=False)
show = ["coins", "funding", "capital_rm", "reinvest_final_rm", "reinvest_avg_gain_per_month_pct", "withdraw_avg_per_month_rm",
        "monthly_costs_rm", "months_profit_covered_all_costs", "paid_from_own_pocket_total_rm", "card_still_owed_end_rm",
        "card_fully_repaid_in_month", "your_net_result_rm"]
for per in res.period.unique():
    print("\n##", per); print(res[res.period == per][show].to_string(index=False))

notes = [
    ("strategy", "Panic Buyer as running live: hourly; buy 12% under the 24h highest hourly close on the 8 closest coins; max 20% per order; all orders fully funded; sell +1.5% or after 48h; 24h cooldown"),
    ("data", "Binance spot hourly candles 2021-01-01 to 2026-10-21; delisted coins included until they stopped trading (holdings at delisting stay at the last price)"),
    ("costs per trade", "0.1% fee each side; market orders extra 0.02% (top-50 liquidity) or 0.1% (other coins)"),
    ("fills", "limit buy only if the price traded 0.1% through it; no take-profit in the same hour as the buy"),
    ("coin screen", "approximation of the user's screener methodology v1, not its database verdicts: see coins_used.csv / coins_excluded.csv"),
    ("liquidity filter", f"median daily trading volume at least US${MIN_MEDIAN_DAILY_USD/1e6:.1f}M"),
    ("money", "results in % are the same for any capital; RM values = capital x %; the bot trades USDT"),
    ("reinvest", "profits stay in the bot and compound"),
    ("withdraw", "each month-end, everything above the starting capital is withdrawn; after a losing month nothing is withdrawn until it is back to the starting capital"),
    ("monthly costs", "Claude Pro RM82 + Lightsail server RM20 (US$5 at ~4.08) + credit-card charge 0.75% of the borrowed capital per month while any is owed"),
    ("card repayment", "profit left after the monthly costs repays the card; shortfalls in weak months are paid from your own pocket"),
    ("your_net_result_rm", "money still in the bot - card still owed - own pocket paid + free cash received (own money: also minus the capital you put in)"),
    ("warning", "backtest, not a promise: live results are usually somewhat worse; losses on borrowed money are still owed"),
]
pd.DataFrame(notes, columns=["item", "assumption"]).to_csv(f"{OUT}/assumptions.csv", index=False)
