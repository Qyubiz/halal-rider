# Panic Buyer: research handoff (22 Oct 2026)

Everything we know about the Panic Buyer bot, so a new session can refine it without repeating work.

## 1. What exists

**Live bot**, in repo `Qyubiz/halal-rider` (PR #1 merged to `main`, commit `cba637c`):

- `panic_buyer.py`: the live Binance spot bot. Commands: `check | run | status | pause | panic`.
- `binance_spot.py`: halal-rider's minimal Binance client, unchanged. It can't withdraw.
- `deploy/panic-install.sh`, `deploy/panic-buyer-setup`, `deploy/panic-buyer` (CLI) and `deploy/panic-buyer.service` (systemd).
- `tests/test_panic.py` and `tests/fake_binance.py`. `python3 -m unittest discover -s tests` passes 25 tests (15 halal-rider, 10 panic buyer).
- `PANIC_BUYER.md`: user-facing install and command guide.
- Settings can be overridden with `panic_config.json` next to the script. Keys: `symbols`, `depth_pct`, `n_orders`, `max_order_pct`, `take_profit_pct`, `max_hours`, `cooldown_hours`, `fee_pct`.
- Environment file: `/etc/panic-buyer.env` with `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `CAPITAL_USDT`, `COMPOUND` (default no), optional `TELEGRAM_BOT_TOKEN`/`TELEGRAM_CHAT_ID`, and `PANIC_DATA`.

**Status:** built and tested, **not yet live**. The plan is to install it on the user's AWS Lightsail server next to halal-rider, start with 300 USDT for 2–4 weeks, then scale to RM5–6k if live results track the backtest.

**Research code** (this folder, `code/`) was written in a temporary workspace. Paths are hard-coded to `/home/claude/bt/...`; see section 9.

## 2. The strategy (live settings)

The thesis: in crypto, leveraged futures traders get force-liquidated in sharp drops, which pushes spot prices below fair value for a short time. A funded limit order already waiting lower catches that overshoot and sells into the bounce.

- **Universe:** halal-rider's 19 halal coins: BTC ETH SOL XRP BNB DOGE ADA TRX LINK AVAX SUI LTC BCH HBAR DOT NEAR ARB FET RENDER (USDT pairs).
- **Cycle:** every hour, 10 s after the hourly candle closes.
- **Trigger** per coin = highest close of the last 24 closed hourly candles × (1 − 12%).
- **Entry:** limit BUYs at the trigger on the **8 coins closest to their trigger**, ranked by `price / trigger − 1`.
  - Each order is at most 20% of bot capital, and all orders together never exceed free cash (fully funded; Binance locks the USDT).
  - A coin already under its trigger is bought at market.
  - Unfilled orders are cancelled and re-placed every hour.
- **Exit:** a limit SELL at +1.5% is placed right after the fill is seen. If it hasn't sold after 48 h, the coin is sold at market.
- **Cooldown:** 24 h after selling a coin.
- **Money:** profits are set aside by default (`COMPOUND=no`) and losses shrink the budget. The bot only touches orders and coins it created, because it shares the account with halal-rider.

## 3. Headline results

Binance hourly data. Costs: 0.1% fee per side, plus 0.02% slippage on market orders (majors) or 0.1% (other coins). Fill rules: a limit buy fills only if price trades 0.1% through it, and a take-profit can't fill in the same hour as the buy.

**19 halal coins** (the live list). Settings were chosen on 2021–2023 data only (see section 4):

| Measure | Value |
|---|---|
| Jan 2024 – Oct 2026 (out of sample) | ×2.61 (RM6,000 → RM15,653 with profits reinvested), **+2.86%/month compounded** |
| Months positive | 30 of 34 |
| Worst month | −6% |
| Max drawdown 2024–26 | −16% |
| Max drawdown 2021–23 | −29% |
| Per year | 2021 +255% · 2022 +19% · 2023 +33% · 2024 +71% · 2025 +28% · 2026 to Oct +19% |
| Trades (since 2024) | about 26/month, 93% wins, average +1.09%, worst −22% |
| With LUNA/FTT/SRM collapses added | still positive: 2024–26 RM6,000 → RM14,753; 2022 +1%; worst single trade −100% (LUNA), about −13% of the account |
| If every fill were 0.5% worse (21-coin set) | still about +1.7%/month |

**Adding coins makes it worse.** These are 2024–26 multiples on the same rules, using screened halal coins ranked by median daily volume. Delisted coins are included:

| Coins | 2024–26 multiple | Max drawdown |
|---|---|---|
| Top 20 | ×2.13 | −18% |
| Top 50 | ×2.30 | −26% |
| Top 100 | ×0.93 | −44% |
| Top 150 | ×0.63 | −64% |
| All 263 | ×0.50 | −63% |
| 19 halal-rider coins | ×2.61 | −16% |

Since 2024, coins ranked 51–150 lost 23% of the account and coins that were later delisted lost 23.5%. The top 50 made +33%. Small coins crash for real reasons (delistings, token unlocks, hacks) and don't bounce back. Worst trades were real events: GFT and BAKE around their delistings, and sharp drops in HEMI, TRB, MASK and MANTA.

## 4. Research path: what failed and what worked

All "test" numbers are out of sample: settings chosen on 2021-01 to 2023-12, results on 2024-01 to 2026-10, unless noted.

**Failed (don't repeat):**
- **Indicator-confluence entries** (10 of 12, up to all of 40 indicators, "no trade if any conflict"): win rate 63–78%, but negative expectancy after the 0.24% round-trip cost. Average wins were small and average losses large. The full-agreement version almost never traded.
- **1h five-indicator rule + reward/risk ≥ 2:** +0.8%/trade over 90 days, about 0 over 2 years.
- **4h EMA20/50 cross + 3×ATR trailing stop:** +0.93%/trade over 2 years, but lumpy and sensitive to settings.
- **Volatility harvesting** (RM6,000 →, 2024–26): trailing grid RM6.4k (max drawdown −34%); rebalancing RM5.9k.
- **Rotation and regime switches:** cross-coin momentum rotation RM5.8k (max drawdown −66%); breadth on/off switch RM7.4k (max drawdown −59%).
- **BTC→alt lead-lag:** about 0 per trade after costs.
- **Funding rate as a signal:** the contrarian basket lost (RM4.7k), and using funding as a dip-buyer filter added nothing.
- **Checking every 15 minutes instead of hourly:** no improvement (21-coin set: ×2.02 vs ×2.49).
- **Shallow 8% trigger:** fails. Triggers of 10–15% work.

**What worked, step by step:**
1. **The edge.** Buying 10–12% under the 24h high and selling at +1.5–2% wins about 91% of trades with +0.85% average. Verified by an independent plain-Python event study that reproduced the engine (4,192 vs 4,194 trades). Positive every year except 2022, which was flat.
2. **Realism fix.** The first high-return version (×8 on test) placed more limit orders than the account could fund, which Binance doesn't allow.
   - Fully funded static orders on every coin: about +1.1%/month.
   - Hourly market buys only: about +2.3%/month, but −47% in 2022.
3. **Chosen design.** Dynamic, fully funded resting bids: each hour, all free cash goes into orders on the N coins closest to their trigger. This captures most of the edge with real money.
4. **Picking the setting** (960 combinations, 21 coins plus LUNA/FTT/SRM):
   - Rule, using 2021–2023 data only: keep settings that stayed within −5% in 2022, then take the best return/drawdown with drawdown better than −35%. Result: depth 12%, 8 orders, +1.5%, 48 h.
   - All 64 settings that passed the 2022 rule were profitable in 2024–26 at 97%, median +27%/year.
5. **Engine bug fixed.** Cash from a sale is now usable only from the next candle; it had been usable within the same candle.

## 5. Validation done

- **Event study vs engine:** identical trade counts and returns, as above.
- **Paper bot replay:** the paper bot's own code, run hour by hour over history, matches the engine within 0.1%. Sep–Dec 2025: 1.1052 vs 1.1051. Apr–Jul 2022: 1.1753 vs 1.1749.
- **Live bot replay:** the live `panic_buyer.py`, driven through a fake Binance (`tests/fake_binance.py`) on real history, compared with the engine:

| Period | Live bot | Engine |
|---|---|---|
| Sep–Dec 2025 | 1.047 | 1.050 |
| Mar–Jun 2024 | 1.130 | 1.134 |
| May–Jul 2022 | 1.132 | 1.174 |
| May–Oct 2026 | 1.163 | 1.134 |

  Differences come from small timing details: market buy at the last close vs the open, and the time-stop hour.
- **Unit tests:** funded orders on the 8 closest coins; forming candle ignored; shared USDT and minimum order size respected; market buy under trigger; fill → take-profit → 24 h cooldown; 48 h time stop; partial fills; Binance -2013 (unknown order) handling; panic touches only the bot's own orders and coins.
- **Server install** simulated end to end (install, update, setup reusing halal-rider's key, env permissions, status).
- **halal-rider fix:** `install.sh` got `git -c safe.directory=...` so `halal-rider update` no longer fails with "dubious ownership".

## 6. Capital and costs (the user's plan)

Assumptions:
- Capital comes from the credit card with a **one-time 0.75% fee** (no monthly interest assumed).
- Monthly costs: Claude Pro **RM82** + Lightsail **RM20**.
- All profit above the starting capital is withdrawn monthly. After a loss, nothing is withdrawn until the account is back to the starting capital.

19 coins, Jan 2024 – Oct 2026 (`results/results_by_capital.csv`):

| Capital | Profit/month | Left after costs | Net after 34 months |
|---|---|---|---|
| RM3,500 | RM102 | RM0 (break-even) | about −RM36 |
| RM5,000 | RM145 | +RM43 | +RM1,435 |
| RM6,000 | RM174 | +RM72 | +RM2,416 |

- Repaying the card from profit alone takes about 71 months at RM6k. At any size it can't be faster than about 35 months, because the bot makes about 2.9%/month.
- A −29% drawdown (the worst, in 2021–23) on RM6,000 is about −RM1,740 still owed.
- Agreed sweet spot: **RM5–6k**, only if the card bill is paid from salary when due.

## 7. Known limitations and risks

1. Hourly candles hide the order of moves inside the hour, so the fill rules are conservative. Live could be better (take-profit inside the fill hour) or worse (queue position, fast wicks).
2. The 19 coins are survivors chosen today. The 263-coin test, which includes delisted coins, shows the risk outside large caps.
3. Only 2021–2026 data: one bull run, one bear market and a recovery. 2021 dominates the full-period numbers.
4. Many settings were tried (960 for the dynamic bot plus earlier families), so there is multiple-testing risk. The design/test split and plateau checks mitigate it but don't remove it.
5. The bot doesn't check Binance delisting announcements. Low risk for the 19 large coins; essential before widening the universe.
6. It shares the account with halal-rider: the account must hold USDT for both bots' capital, because resting buys lock USDT.
7. Fees assume 0.1% with no BNB discount; the BNB discount would improve results. Time-stop market sells assume small slippage.
8. Shariah: spot only, no leverage or interest. The coin screen follows the user's methodology v1 (SC Malaysia SAC + AAOIFI) but is an approximation, not the verdicts in their screener database.

## 8. Refinement ideas (highest value first)

1. **Live vs backtest reconciliation.** After 2–4 weeks live, replay the exact live period with `code/replay_live_bot.py` and compare trade by trade: fills missed or extra, slippage, take-profit timing.
2. **Faster exits.** Watch fills via Binance's user-data websocket (or poll every 1–5 min) and place the take-profit immediately. Test with 1m/5m candles; the backtest currently forbids a take-profit in the fill hour.
3. **Adaptive trigger depth.** Scale depth per coin by volatility (k × 24h ATR%, or a rolling percentile of 24h drawdowns) instead of a fixed 12%.
4. **Stress-aware sizing.** Bigger or more orders when BTC itself is falling fast (liquidation cascades), fewer in calm markets.
5. **Dynamic universe.** Re-rank the halal list monthly by volume and keep the top 20–50 (top-50 tested ×2.30 vs ×2.61 for the fixed 19; check stability).
6. **Exit variants.** Partial take-profit plus trailing remainder; a shorter time stop when market breadth is falling.
7. **Guards.** Binance delisting schedule (`GET /sapi/v1/spot/delist-schedule`), monitoring-tag coins, exchange maintenance.
8. **Statistical rigor.** Anchored yearly walk-forward, parameter heatmaps, block-bootstrap Monte Carlo of monthly returns, deflated Sharpe / probability of backtest overfitting (CSCV).
9. **Cost model.** BNB fee discount; slippage from real order-book depth.

## 9. How to reproduce

- Python 3.11+ with `pip install numpy pandas numba`. Market data is public with no API key: `https://data-api.binance.vision/api/v3/klines`. Funding history comes from `https://data.binance.vision`.
- Fetch command: `THREADS=8 python3 code/fetch_klines.py 1h 2021-01-01 BTCUSDT,ETHUSDT,... DATA_DIR`.
- Data folders the scripts expect, under `/home/claude/bt/`:

| Folder | Contents |
|---|---|
| `data_long/` | 1h, 22 large coins |
| `data_halal/` | 1h: SUI, BCH, HBAR, ARB, FET, RENDER, RNDR |
| `data_dead/` | 1h: LUNA, FTT, SRM. Cut LUNA at 2022-05-13 (the later data is LUNA 2.0), FTT at 2023-09-22 |
| `data_15m/`, `data_dead15/` | 15-minute candles |
| `data_funding/` | funding rates |
| `data_univ/` | 1h, 322 screened coins |

- Scripts in `code/`:

| Script | What it does |
|---|---|
| `lab.py` | Numba engines. `dyn_kernel` is the live strategy; `wick_kernel`/`wick2`, `grid_kernel` and `rebalance_kernel` are the earlier families. Also has metrics. |
| `verify4.py` | The 960-setting grid for the live strategy (2021–23 design / 2024–26 test) → `dyn_grid.csv` |
| `final_report.py` | Detail for the chosen (steady) and aggressive settings |
| `halal19.py` | The 19-coin backtest |
| `universe.py` + `big_backtest.py` | 150+ coin universe and all result CSVs (`MIN_VOL` env sets the liquidity filter) |
| `scenario_cc.py` | Capital, cost and credit-card scenarios from `monthly_results.csv` |
| `verify_wick.py` | Independent event study, realism variants, stress test |
| `verify2.py`, `verify3.py` | Hourly-only and fully funded variants, bear-market filters |
| `verify5.py` | Funding-rate tests |
| `verify6.py` | The 15-minute version |
| `run_vol.py`, `run_rot.py` | Grid, rebalance, wick, rotation, breadth and lead-lag families |
| `replay_bot.py` | Paper bot replayed hour by hour vs the engine |
| `replay_live_bot.py` | Live bot through the fake Binance vs the engine. Needs the repo's `panic_buyer.py`, `binance_spot.py` and `tests/fake_binance.py` on the path. |
| `early/` | The first indicator-based experiments (all failed; kept for reference) |

- `results/`: result CSVs (`results_by_capital.csv`, `monthly_results.csv`, `coin_count_comparison.csv`, `coins_used.csv`, `coins_excluded.csv`, `assumptions.csv`). Trade logs aren't included; regenerate them with `big_backtest.py`.
- `snapshot/binance_usdt_pairs_2026-10-21.csv`: Binance USDT pairs with status (TRADING/BREAK) and 24h volume, as used by the universe screen.

## 10. User context

- Shariah-compliant spot trading only: no leverage, margin, futures, shorting or interest products.
- Binance spot, on an account verified before Malaysia's block. Do not suggest Malaysian exchanges (Luno, HATA) for this.
- Not a programmer. Prefers copy-paste steps. GitHub `Qyubiz/halal-rider`; AWS Lightsail Ubuntu server.
- halal-rider (a pump-rider bot, capital US$300) runs on the same server and account.
- Goals:
  - Withdraw profit monthly to pay Claude Pro (RM82) and the server (RM20), and repay the card used as capital.
  - Original target was 4–6%/month; the backtest gives about 2.9%/month.
