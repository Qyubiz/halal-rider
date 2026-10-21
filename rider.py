#!/usr/bin/env python3
"""Halal pump-rider: live Binance SPOT bot.

Rules (from the research, see scalper/README.md)
  1. Only when Bitcoin is above its 50-day average (daily closes).
  2. Signal on a closed 5-minute candle: price up >= 2% over 15 minutes, volume >= 3x the 4-hour
     average, green candle.
  3. Don't chase: place a stop-buy 1% above the signal close, valid 60 minutes. It only fills if the
     pump keeps going. Limit 0.3% above the trigger caps slippage.
  4. After a fill: a stop-loss order sits ON BINANCE 5% under the entry (protects even if this server
     is down). Every 5 minutes the bot raises it to 8% under the highest price since entry.
  5. Sold after 24 hours at most. At most 5 trades at once, each 1/5 of the bot's capital.
  6. If the account falls 10% in a week, no new trades until next week (open trades keep their stops).

Commands
  python rider.py check     test keys, balances, symbols and Bitcoin trend. Places no orders.
  python rider.py run       trade, forever (run it as the systemd service)
  python rider.py status    print open trades, pending buys and results
  python rider.py panic     cancel the bot's orders and sell the bot's coins at market, then stop

Environment (in /etc/halal-rider.env on the server, never in git)
  BINANCE_API_KEY, BINANCE_API_SECRET   spot-trading key, withdrawals OFF, IP-restricted
  CAPITAL_USDT                          the most the bot may use (e.g. 300)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  optional phone alerts
"""

import argparse
import csv
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from binance_spot import BinanceError, BinanceSpot, round_step  # noqa: E402

MYT = timezone(timedelta(hours=8))
FIVE_MS = 5 * 60 * 1000
DEFAULT_CFG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT",
                "LINKUSDT", "AVAXUSDT", "SUIUSDT", "LTCUSDT", "BCHUSDT", "HBARUSDT", "DOTUSDT", "NEARUSDT",
                "ARBUSDT", "FETUSDT", "RENDERUSDT"],
    "rise_pct": 2.0, "vol_mult": 3.0, "lookback": 48,
    "confirm_pct": 1.0, "confirm_minutes": 60, "buy_limit_slip_pct": 0.3,
    "stop_pct": 5.0, "trail_pct": 8.0, "max_hours": 24,
    "max_positions": 5, "btc_trend_days": 50,
    "weekly_halt_pct": 10.0, "restep_stop_min_pct": 0.3,
}


def now_ms():
    return int(time.time() * 1000)


def myt(ms):
    return datetime.fromtimestamp(ms / 1000, MYT).strftime("%Y-%m-%d %H:%M")


def ema(values, n):
    out, k, cur = [], 2 / (n + 1), None
    for v in values:
        cur = v if cur is None else v * k + cur * (1 - k)
        out.append(cur)
    return out


class Rider:
    def __init__(self, client, cfg, capital, root=None, out=print, notify=None, clock=now_ms):
        root = root or os.environ.get("RIDER_DATA") or HERE
        os.makedirs(root, exist_ok=True)
        self.c = client
        self.cfg = cfg
        self.capital = float(capital)
        self.root = root
        self.out = out
        self.notify = notify or (lambda msg: None)
        self.clock = clock
        self.state = self.load_state()
        self.info = {}
        self._btc = (0, None)

    # ------------------------------------------------------------- state
    @property
    def state_path(self):
        return os.path.join(self.root, "rider_state.json")

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {"positions": {}, "armed": {}, "last_signal": {}, "week": None, "week_start": None,
                "halted_week": None, "realized_usdt": 0.0}

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, sort_keys=True)
        os.replace(tmp, self.state_path)

    def log(self, msg, alert=False):
        line = f"{myt(self.clock())} {msg}"
        self.out(line)
        try:
            with open(os.path.join(self.root, "rider.log"), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if alert:
            self.notify(msg)

    def record(self, sym, entry, exit_px, qty, reason):
        pnl = (exit_px - entry) * qty
        self.state["realized_usdt"] = self.state.get("realized_usdt", 0.0) + pnl
        path = os.path.join(self.root, "rider_trades.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["closed_myt", "symbol", "qty", "entry", "exit", "pnl_usdt_before_fees", "reason"])
            w.writerow([myt(self.clock()), sym, f"{qty:.8g}", f"{entry:.8g}", f"{exit_px:.8g}", f"{pnl:.4f}", reason])
        self.log(f"SOLD {sym} {qty:.8g} @ {exit_px:.8g} ({reason}), P&L {pnl:+.2f} USDT "
                 f"({(exit_px / entry - 1) * 100:+.2f}%)", alert=True)

    # ------------------------------------------------------------- helpers
    def load_info(self):
        self.info = self.c.exchange_info(self.cfg["symbols"])

    def qty_str(self, sym, qty):
        return round_step(qty, self.info[sym]["step"])

    def px_str(self, sym, px, up=False):
        return round_step(px, self.info[sym]["tick"], up=up)

    def btc_uptrend(self):
        t = self.clock()
        if t - self._btc[0] < 30 * 60 * 1000 and self._btc[1] is not None:
            return self._btc[1]
        d = self.c.klines("BTCUSDT", "1d", self.cfg["btc_trend_days"] + 30)
        closed = [k["close"] for k in d if k["close_ms"] < t]
        avg = ema(closed, self.cfg["btc_trend_days"])[-1]
        ok = d[-1]["close"] > avg
        self._btc = (t, ok)
        return ok

    def equity(self, prices, balances):
        usdt = balances.get("USDT", {"free": 0.0, "locked": 0.0})
        total = usdt["free"] + usdt["locked"]
        for sym, p in self.state["positions"].items():
            total += p["qty"] * prices.get(sym, p["entry"])
        return total

    def bot_capital(self, prices, balances):
        return min(self.capital, self.equity(prices, balances))

    def avg_fill(self, order):
        qty = float(order.get("executedQty") or 0)
        quote = float(order.get("cummulativeQuoteQty") or 0)
        return qty, (quote / qty if qty else 0.0)

    # ------------------------------------------------------------- orders
    def place_stop(self, sym, qty, stop_px, price_now):
        """Stop-loss sell that lives on Binance. Sells at market right away if already under the stop."""
        q = self.qty_str(sym, qty)
        if float(q) <= 0:
            return None
        if price_now <= stop_px:
            self.market_sell(sym, qty, "stop-loss (price already under the stop)")
            return "SOLD"
        types = self.info[sym]["order_types"]
        if "STOP_LOSS" in types:
            o = self.c.new_order(symbol=sym, side="SELL", type="STOP_LOSS", quantity=q,
                                 stopPrice=self.px_str(sym, stop_px))
        else:
            o = self.c.new_order(symbol=sym, side="SELL", type="STOP_LOSS_LIMIT", timeInForce="GTC", quantity=q,
                                 stopPrice=self.px_str(sym, stop_px), price=self.px_str(sym, stop_px * 0.99))
        return o["orderId"]

    def market_sell(self, sym, qty, reason):
        p = self.state["positions"].get(sym)
        q = self.qty_str(sym, qty)
        if float(q) <= 0:
            self.state["positions"].pop(sym, None)
            return
        o = self.c.new_order(symbol=sym, side="SELL", type="MARKET", quantity=q)
        sold, px = self.avg_fill(o)
        if p:
            self.record(sym, p["entry"], px, sold, reason)
            self.state["positions"].pop(sym, None)
        self.save()

    def open_position(self, sym, qty_exec, entry, price_now):
        base = self.info[sym]["base"]
        free = self.c.balances().get(base, {"free": 0.0})["free"]
        qty = min(qty_exec, free)  # if Binance took the fee in coins, sell what we really hold
        self.state["positions"][sym] = {"qty": qty, "entry": entry, "opened_ms": self.clock(), "peak": entry,
                                        "stop": entry * (1 - self.cfg["stop_pct"] / 100), "stop_id": None,
                                        "last_ms": (self.clock() // FIVE_MS) * FIVE_MS - FIVE_MS}
        self.save()
        self.log(f"BOUGHT {sym} {qty:.8g} @ {entry:.8g} (stop {self.state['positions'][sym]['stop']:.8g})", alert=True)
        sid = self.place_stop(sym, qty, self.state["positions"][sym]["stop"], price_now)
        if sid != "SOLD" and sym in self.state["positions"]:
            self.state["positions"][sym]["stop_id"] = sid
        self.save()

    # ------------------------------------------------------------- the cycle
    def check_armed(self, prices):
        t = self.clock()
        for sym in list(self.state["armed"]):
            a = self.state["armed"][sym]
            try:
                o = self.c.get_order(sym, a["order_id"])
                status = o.get("status")
                if status == "FILLED":
                    qty, px = self.avg_fill(o)
                    del self.state["armed"][sym]
                    self.open_position(sym, qty, px, prices.get(sym, px))
                elif t >= a["expires_ms"] or status in ("CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"):
                    if status not in ("CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH"):
                        try:
                            self.c.cancel_order(sym, a["order_id"])
                        except BinanceError:
                            pass
                        o = self.c.get_order(sym, a["order_id"])
                    qty, px = self.avg_fill(o)
                    del self.state["armed"][sym]
                    if qty > 0:
                        self.open_position(sym, qty, px, prices.get(sym, px))
                    else:
                        self.log(f"{sym}: pump did not continue, stop-buy cancelled (no trade)")
                self.save()
            except BinanceError as err:
                self.log(f"ALERT {sym}: checking the stop-buy failed: {err}", alert=True)

    def manage_positions(self, prices):
        t = self.clock()
        cfg = self.cfg
        for sym in list(self.state["positions"]):
            p = self.state["positions"][sym]
            try:
                if p.get("stop_id"):
                    o = self.c.get_order(sym, p["stop_id"])
                    if o.get("status") == "FILLED":
                        qty, px = self.avg_fill(o)
                        self.record(sym, p["entry"], px, qty, "stop on Binance")
                        del self.state["positions"][sym]
                        self.save()
                        continue
                    if o.get("status") in ("CANCELED", "EXPIRED", "REJECTED"):
                        p["stop_id"] = None
                if t - p["opened_ms"] >= cfg["max_hours"] * 3600 * 1000:
                    self.cancel_stop(sym, p)
                    if sym in self.state["positions"]:
                        self.market_sell(sym, p["qty"], f"time limit {cfg['max_hours']}h")
                    continue
                for k in self.c.klines(sym, "5m", 300):
                    if k["open_ms"] > p["last_ms"] and k["close_ms"] < t:
                        p["peak"] = max(p["peak"], k["high"])
                        p["last_ms"] = k["open_ms"]
                price = prices.get(sym, p["peak"])
                p["peak"] = max(p["peak"], price)
                want = max(p["stop"], p["peak"] * (1 - cfg["trail_pct"] / 100))
                if not p.get("stop_id") or want >= p["stop"] * (1 + cfg["restep_stop_min_pct"] / 100):
                    self.cancel_stop(sym, p)
                    if sym not in self.state["positions"]:
                        continue
                    p["stop"] = want
                    sid = self.place_stop(sym, p["qty"], want, price)
                    if sid != "SOLD" and sym in self.state["positions"]:
                        p["stop_id"] = sid
                        self.log(f"{sym}: stop raised to {want:.8g} (peak {p['peak']:.8g})")
                self.save()
            except BinanceError as err:
                self.log(f"ALERT {sym}: managing the trade failed: {err}", alert=True)

    def cancel_stop(self, sym, p):
        if not p.get("stop_id"):
            return
        try:
            self.c.cancel_order(sym, p["stop_id"])
        except BinanceError:
            pass
        o = self.c.get_order(sym, p["stop_id"])
        p["stop_id"] = None
        if o.get("status") == "FILLED":  # it triggered while we were replacing it
            qty, px = self.avg_fill(o)
            self.record(sym, p["entry"], px, qty, "stop on Binance")
            self.state["positions"].pop(sym, None)
        self.save()

    def weekly_ok(self, prices, balances):
        now = datetime.fromtimestamp(self.clock() / 1000, MYT)
        y, w, _ = now.isocalendar()
        key = f"{y}-W{w:02d}"
        eq = self.equity(prices, balances)
        if self.state.get("week") != key:
            self.state["week"], self.state["week_start"] = key, eq
        if self.state.get("halted_week") == key:
            return False
        if eq < self.state["week_start"] * (1 - self.cfg["weekly_halt_pct"] / 100):
            self.state["halted_week"] = key
            self.log(f"ALERT: account down more than {self.cfg['weekly_halt_pct']}% this week: no new trades "
                     f"until next week. Open trades keep their stops.", alert=True)
            return False
        return True

    def scan(self, prices):
        cfg = self.cfg
        t = self.clock()
        balances = self.c.balances()
        if not self.weekly_ok(prices, balances):
            return
        if not self.btc_uptrend():
            return
        slots = cfg["max_positions"] - len(self.state["positions"]) - len(self.state["armed"])
        if slots <= 0:
            return
        capital = self.bot_capital(prices, balances)
        for sym in cfg["symbols"]:
            if slots <= 0:
                break
            if sym in self.state["positions"] or sym in self.state["armed"] or sym not in self.info:
                continue
            if self.info[sym]["status"] != "TRADING":
                continue
            try:
                ks = [k for k in self.c.klines(sym, "5m", cfg["lookback"] + 6) if k["close_ms"] < t]
            except BinanceError as err:
                self.log(f"{sym}: price read failed: {err}")
                continue
            if len(ks) < cfg["lookback"] + 4:
                continue
            last = ks[-1]
            if self.state["last_signal"].get(sym) == last["open_ms"]:
                continue
            avg_vol = sum(k["volume"] for k in ks[-1 - cfg["lookback"]:-1]) / cfg["lookback"]
            rise = last["close"] / ks[-4]["close"] - 1
            if not (rise >= cfg["rise_pct"] / 100 and avg_vol > 0 and last["volume"] >= cfg["vol_mult"] * avg_vol
                    and last["close"] > last["open"]):
                continue
            self.state["last_signal"][sym] = last["open_ms"]
            usdt_free = self.c.balances().get("USDT", {"free": 0.0})["free"]
            stake = min(capital / cfg["max_positions"], usdt_free * 0.98)
            trigger = last["close"] * (1 + cfg["confirm_pct"] / 100)
            limit = trigger * (1 + cfg["buy_limit_slip_pct"] / 100)
            qty = float(self.qty_str(sym, stake / limit))
            if qty * limit < max(self.info[sym]["min_notional"], 5.0) or qty < self.info[sym]["min_qty"]:
                self.log(f"{sym}: signal, but {stake:.2f} USDT is below Binance's minimum order")
                continue
            price = prices.get(sym, last["close"])
            self.log(f"SIGNAL {sym}: +{rise * 100:.1f}% in 15 min on {last['volume'] / avg_vol:.1f}x volume")
            try:
                if price >= trigger:
                    if price > limit:
                        self.log(f"{sym}: already {price / last['close'] * 100 - 100:.1f}% above the signal, too late")
                        continue
                    o = self.c.new_order(symbol=sym, side="BUY", type="MARKET", quoteOrderQty=f"{qty * price:.2f}")
                    q, px = self.avg_fill(o)
                    if q > 0:
                        self.open_position(sym, q, px, price)
                else:
                    o = self.c.new_order(symbol=sym, side="BUY", type="STOP_LOSS_LIMIT", timeInForce="GTC",
                                         quantity=self.qty_str(sym, qty), stopPrice=self.px_str(sym, trigger, up=True),
                                         price=self.px_str(sym, limit))
                    self.state["armed"][sym] = {"order_id": o["orderId"], "trigger": trigger, "qty": qty,
                                                "expires_ms": t + cfg["confirm_minutes"] * 60 * 1000}
                    self.log(f"{sym}: stop-buy set at {trigger:.8g} for {qty:.8g} (valid {cfg['confirm_minutes']} min)")
                slots -= 1
                self.save()
            except BinanceError as err:
                self.log(f"ALERT {sym}: order failed: {err}", alert=True)

    def cycle(self):
        prices = self.c.prices(self.cfg["symbols"])
        self.check_armed(prices)
        self.manage_positions(prices)
        self.scan(prices)
        self.save()

    # ------------------------------------------------------------- commands
    def panic(self):
        for sym, a in list(self.state["armed"].items()):
            try:
                self.c.cancel_order(sym, a["order_id"])
            except BinanceError as err:
                self.log(f"{sym}: cancel stop-buy: {err}")
            del self.state["armed"][sym]
        for sym in list(self.state["positions"]):
            p = self.state["positions"][sym]
            try:
                self.cancel_stop(sym, p)
                if sym in self.state["positions"]:
                    self.market_sell(sym, p["qty"], "panic")
            except BinanceError as err:
                self.log(f"ALERT {sym}: panic sell failed, sell it in the Binance app: {err}", alert=True)
        self.save()


def telegram():
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return None

    def send(msg):
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": "Halal rider: " + msg}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10)
        except Exception:  # noqa: BLE001 - alerts must never stop trading
            pass
    return send


def load_cfg():
    path = os.path.join(HERE, "rider_config.json")
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Halal pump-rider (Binance spot)")
    ap.add_argument("command", choices=["check", "run", "status", "panic"])
    a = ap.parse_args(argv)
    cfg = load_cfg()
    client = BinanceSpot()
    capital = float(os.environ.get("CAPITAL_USDT", "0") or 0)
    bot = Rider(client, cfg, capital, notify=telegram())

    if a.command == "status":
        s = bot.state
        print(f"Open trades: {len(s['positions'])}, waiting stop-buys: {len(s['armed'])}, "
              f"realized P&L (before fees): {s.get('realized_usdt', 0):+.2f} USDT")
        for sym, p in s["positions"].items():
            print(f"  {sym}: {p['qty']:.8g} @ {p['entry']:.8g}, stop {p['stop']:.8g}, peak {p['peak']:.8g}, "
                  f"opened {myt(p['opened_ms'])}")
        for sym, x in s["armed"].items():
            print(f"  waiting {sym}: buy if it reaches {x['trigger']:.8g} before {myt(x['expires_ms'])}")
        return 0

    client.sync_time()
    bot.load_info()
    if a.command == "check":
        print(f"Binance reachable, clock offset {client.offset_ms} ms, {len(bot.info)} of {len(cfg['symbols'])} coins found")
        bal = client.balances()
        print(f"Key works. USDT free {bal.get('USDT', {}).get('free', 0):.2f}, BNB {bal.get('BNB', {}).get('free', 0):.4f}")
        print(f"CAPITAL_USDT (the most the bot may use): {capital:.2f}")
        print(f"Bitcoin above its {cfg['btc_trend_days']}-day average: {'YES, trading allowed' if bot.btc_uptrend() else 'NO, bot waits'}")
        if capital <= 0:
            print("CAPITAL_USDT is not set: the bot will not trade.")
        print("CHECK PASSED" if capital > 0 else "CHECK FAILED")
        return 0 if capital > 0 else 1
    if a.command == "panic":
        bot.panic()
        print("All bot orders cancelled and bot coins sold. Stop the service: sudo systemctl stop halal-rider")
        return 0
    if capital <= 0:
        print("CAPITAL_USDT is not set in /etc/halal-rider.env, refusing to trade.")
        return 1
    bot.log(f"Started. Capital limit {capital:.2f} USDT, {len(bot.info)} coins.", alert=True)
    last_info = time.time()
    while True:
        try:
            if time.time() - last_info > 6 * 3600:
                bot.load_info()
                client.sync_time()
                last_info = time.time()
            bot.cycle()
        except BinanceError as err:
            bot.log(f"ALERT: cycle error: {err}", alert=True)
        except Exception as err:  # noqa: BLE001 - keep running; stops are on Binance anyway
            bot.log(f"ALERT: unexpected error: {err!r}", alert=True)
        wait = 300 - (time.time() % 300) + 8  # a few seconds after each 5-minute candle closes
        time.sleep(wait)


if __name__ == "__main__":
    sys.exit(main())
