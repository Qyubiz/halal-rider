# Halal pump-rider (Binance spot, live)

A small bot that buys the start of strong pumps in 19 halal coins and rides them with a trailing stop.
It only trades while Bitcoin is above its 50-day average.

**It cannot withdraw or move funds.** Use an API key with only "Spot trading" enabled, withdrawals off,
restricted to your server's IP. Keys live only in `/etc/halal-rider.env` on your server.

## Rules
1. Bitcoin above its 50-day average (daily closes), otherwise no new trades.
2. Signal on a closed 5-minute candle: +2% over 15 minutes on 3x the 4-hour average volume, green.
3. Stop-buy 1% above the signal close, valid 60 minutes (it only buys if the pump keeps going).
4. Stop-loss on Binance 5% under the entry, raised every 5 minutes to 8% under the highest price.
5. Sold after 24 hours at most. Max 5 trades, each 1/5 of CAPITAL_USDT. -10% in a week pauses new trades.

## Install on an Ubuntu server
    curl -sL https://raw.githubusercontent.com/Qyubiz/halal-rider/main/deploy/install.sh | sudo bash
    sudo halal-rider-setup

## Everyday commands
    halal-rider status    open trades and results
    halal-rider log       live log
    halal-rider stop      stop new trading (open trades keep their Binance stop-loss)
    halal-rider panic     cancel the bot's orders and sell the bot's coins at market
    halal-rider update    get the newest version

Backtest (research in the luno-halal-bot repo): one year of 5-minute data, 19 coins, RM6,000 -> about
RM8,860 with a 12.5% worst drop. Past results do not guarantee future results; live trading is usually
somewhat worse. Not financial advice.
