# Panic Buyer (Binance spot, live)

Buys sharp crashes and sells the bounce. Leveraged futures traders get force-liquidated in fast drops, which
pushes prices below fair value for a short time; buy orders already waiting lower catch that overshoot.
Spot only: no leverage, no borrowing, no interest.

**It cannot withdraw or move funds.** It uses the same kind of key as halal-rider: "Spot trading" only,
withdrawals off, restricted to your server's IP. Keys live only in `/etc/panic-buyer.env` on your server.
It only touches orders and coins it bought itself, so it can run next to halal-rider on the same account.

## Rules
1. Every hour (10 seconds after the hourly candle closes) each coin's trigger = its highest hourly close of the last 24 hours minus 12%.
2. Limit buy orders wait on Binance at the trigger of the 8 coins closest to it. Each order is at most 20% of the
   bot's money and all orders together never exceed its free money. A coin already under its trigger is bought at market.
3. Every buy immediately gets a limit sell on Binance at +1.5%.
4. Not sold after 48 hours: sold at market, and that coin is skipped for 24 hours.
5. Profits are set aside as USDT you can withdraw (set `COMPOUND=yes` to reinvest them).

## Backtest (same 19 halal coins as halal-rider, hourly Binance data, 0.1% fees)
Settings were picked on 2021-2023 only, then tested on Jan 2024 - Oct 2026:
RM6,000 -> RM15,653 with profits reinvested (about +2.9% a month), 30 of 34 months positive,
worst month -6%, biggest drop -16%. The 2022 crash year (LUNA, FTX) was +19%.
The live code was replayed hour by hour on real history through a simulated Binance and matched the backtest
(e.g. Sep-Dec 2025: +4.7% live vs +5.0% backtest). Past results do not guarantee future results; live trading
is usually somewhat worse. Not financial advice.

Biggest risk: a coin collapsing within 48 hours. In a stress test with LUNA added, one trade lost 100% of its
order (about 13% of the bot's money). Keep the coin list to large, screened coins.

## Optional high-return modes (high risk, off by default)
Both put all the bot's money on one coin at a time and skip any coin already more than 35% under its 7-day
high (crash guard). Backtest per month, win rate and worst drop:

| Period | Normal mode | 90% mode | 10% mode |
|---|---|---|---|
| Settings | 8 coins, 12% dip, +1.5%, 48h | 1 coin, 14% dip, +2.5%, 96h, 3h cooldown | 1 coin, 14% dip, +5%, 48h |
| 2018-2020 (never used to pick settings) | +2.3%, 91% wins, -45% | **+10.9%, 92% wins**, -66% | +11.1%, 80% wins, -61% |
| 2021-2023 | +5.0%, 94% wins, -29% | +13.2%, 94% wins, -64% | +15.0%, 81% wins, -61% |
| 2024 - Oct 2026 | +2.9%, 93% wins, -16% | +6.9%, 93% wins, -35% | +8.1%, 77% wins, -30% |
| Jan 2025 - Oct 2026 | +2.0%, 92% wins, -15% | +5.8%, 93% wins, -23% | +6.1%, 75% wins, -29% |

The real bot code was replayed with both settings and matches the backtest (90% mode: 93-94% wins in every
period). With fills 0.2% worse, 2024-2026 is about +6% (90% mode) and +7% (10% mode) a month.

The cost is risk. Both modes were 61-66% down at their worst point (the DOGE spike of January 2021, the March
2020 crash): RM6,000 -> about RM2,000-2,350. One trade lost about half the account (DOGE, Jan 2021); since 2024
the worst single trade lost 22-27%. Without the crash guard, one LUNA-type collapse would have wiped out the
whole account. Use only money you can afford to lose; not with borrowed money.

To switch one on (and back off by deleting the file):

    sudo cp /opt/panic-buyer/app/panic_config.90wr.example.json /opt/panic-buyer/app/panic_config.json
    sudo systemctl restart panic-buyer

(for 10% mode, copy `panic_config.10pct.example.json` instead)

## Install on the halal-rider server (Ubuntu)
    curl -sL https://raw.githubusercontent.com/Qyubiz/halal-rider/main/deploy/panic-install.sh | sudo bash
    sudo panic-buyer-setup

Setup can reuse halal-rider's Binance key and Telegram. Keep enough USDT in the account for both bots
(this bot's capital plus halal-rider's), because this bot's waiting buy orders lock its USDT.

## Everyday commands
    panic-buyer status    open trades, waiting buys, profit
    panic-buyer log       live log
    panic-buyer stop      stop buying (coins it holds keep their sell order on Binance)
    panic-buyer start     start again
    panic-buyer panic     cancel its orders and sell its coins at market
    panic-buyer update    get the newest version
