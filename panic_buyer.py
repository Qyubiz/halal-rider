#!/usr/bin/env python3
"""Panic Buyer: live Binance SPOT bot that buys sharp crashes and sells the bounce.

Why it works: leveraged futures traders get force-liquidated in sharp drops, pushing prices below fair
value for a short time. Buy orders already waiting 12% lower catch that overshoot.

Rules (backtest on these 19 halal coins, hourly candles, Jan 2024 - Oct 2026: RM6,000 -> RM15,653 with
profits reinvested, 30 of 34 months positive, worst month -6%, biggest drop -16%)
  1. Every hour, 10 seconds after the hourly candle closes: each coin's trigger price is
     its highest hourly close of the last 24 hours minus 12%.
  2. Limit BUY orders wait on Binance at the trigger of the 8 coins closest to it. Each order is at most
     20% of the bot's money, and all orders together never exceed the bot's free money.
     A coin already under its trigger is bought at market.
  3. Each buy immediately gets a limit SELL on Binance at +1.5%.
  4. Not sold after 48 hours: sold at market. That coin is then skipped for 24 hours.
  5. The bot only touches orders and coins it bought itself (halal-rider's trades are safe).
  6. Profits are not reinvested unless COMPOUND=yes: they stay as USDT you can withdraw.

Commands
  python panic_buyer.py check    test keys, balances and coins, show the nearest triggers. No orders.
  python panic_buyer.py run      trade, forever (run it as the systemd service)
  python panic_buyer.py status   open trades, waiting buys, results
  python panic_buyer.py pause    cancel the bot's waiting buy orders (coins it holds keep their sell order)
  python panic_buyer.py panic    cancel the bot's orders and sell the bot's coins at market

Environment (in /etc/panic-buyer.env on the server, never in git)
  BINANCE_API_KEY, BINANCE_API_SECRET   spot-trading key, withdrawals OFF, IP-restricted
  CAPITAL_USDT                          the most the bot may use (e.g. 300)
  COMPOUND                              "yes" reinvests profits (default: no)
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID  optional phone alerts
  PANIC_DATA                            folder for state, log and trade list
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
HOUR_MS = 3600 * 1000
DONE = ("FILLED", "CANCELED", "EXPIRED", "REJECTED", "EXPIRED_IN_MATCH")
DEFAULT_CFG = {
    "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "ADAUSDT", "TRXUSDT",
                "LINKUSDT", "AVAXUSDT", "SUIUSDT", "LTCUSDT", "BCHUSDT", "HBARUSDT", "DOTUSDT", "NEARUSDT",
                "ARBUSDT", "FETUSDT", "RENDERUSDT"],
    "depth_pct": 12.0,        # trigger = 24h highest hourly close minus this
    "n_orders": 8,            # waiting buy orders on the coins closest to their trigger
    "max_order_pct": 20.0,    # one order is at most this share of the bot's money
    "take_profit_pct": 1.5,
    "max_hours": 48,
    "cooldown_hours": 24,
    "fee_pct": 0.1,           # used for the P&L numbers only
}


def now_ms():
    return int(time.time() * 1000)


def myt(ms):
    return datetime.fromtimestamp(ms / 1000, MYT).strftime("%Y-%m-%d %H:%M")


class PanicBuyer:
    def __init__(self, client, cfg, capital, compound=False, root=None, out=print, notify=None, clock=now_ms):
        root = root or os.environ.get("PANIC_DATA") or HERE
        os.makedirs(root, exist_ok=True)
        self.c = client
        self.cfg = cfg
        self.capital = float(capital)
        self.compound = bool(compound)
        self.root = root
        self.out = out
        self.notify = notify or (lambda msg: None)
        self.clock = clock
        self.state = self.load_state()
        self.info = {}

    # ------------------------------------------------------------- state
    @property
    def state_path(self):
        return os.path.join(self.root, "panic_state.json")

    def load_state(self):
        if os.path.exists(self.state_path):
            with open(self.state_path) as f:
                return json.load(f)
        return {"positions": {}, "orders": {}, "cooldown": {}, "realized_usdt": 0.0, "trades": 0, "wins": 0}

    def save(self):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.state, f, indent=2, sort_keys=True)
        os.replace(tmp, self.state_path)

    def log(self, msg, alert=False):
        line = f"{myt(self.clock())} {msg}"
        self.out(line)
        try:
            with open(os.path.join(self.root, "panic.log"), "a") as f:
                f.write(line + "\n")
        except OSError:
            pass
        if alert:
            self.notify(msg)

    def record(self, sym, p, qty, exit_px, reason):
        fee = self.cfg["fee_pct"] / 100
        pnl = (exit_px * (1 - fee) - p["entry"] * (1 + fee)) * qty
        st = self.state
        st["realized_usdt"] = st.get("realized_usdt", 0.0) + pnl
        st["trades"] = st.get("trades", 0) + 1
        st["wins"] = st.get("wins", 0) + (1 if pnl > 0 else 0)
        st["cooldown"][sym] = self.clock() + self.cfg["cooldown_hours"] * HOUR_MS
        path = os.path.join(self.root, "panic_trades.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["closed_myt", "symbol", "qty", "entry", "exit", "pnl_usdt_after_fees", "reason", "opened_myt"])
            w.writerow([myt(self.clock()), sym, f"{qty:.8g}", f"{p['entry']:.8g}", f"{exit_px:.8g}", f"{pnl:.4f}",
                        reason, myt(p["opened_ms"])])
        self.log(f"SOLD {sym} @ {exit_px:.8g} ({reason}): {(exit_px / p['entry'] - 1) * 100:+.2f}%, "
                 f"{pnl:+.2f} USDT after fees. Total {st['realized_usdt']:+.2f} USDT", alert=True)

    # ------------------------------------------------------------- helpers
    def load_info(self):
        self.info = self.c.exchange_info(self.cfg["symbols"])

    def qty_str(self, sym, qty):
        return round_step(qty, self.info[sym]["step"])

    def px_str(self, sym, px, up=False):
        return round_step(px, self.info[sym]["tick"], up=up)

    def min_notional(self, sym):
        return max(self.info[sym]["min_notional"], 5.0)

    def cid(self, kind, sym):
        return f"pb-{kind}-{sym}-{self.clock() // 1000}"[:36]

    def order_status(self, sym, oid):
        """Binance order, or a cancelled placeholder if Binance no longer knows it (code -2013)."""
        try:
            return self.c.get_order(sym, oid)
        except BinanceError as err:
            if err.code == -2013:
                self.log(f"{sym}: Binance no longer knows order {oid}, treating it as cancelled")
                return {"status": "CANCELED", "executedQty": "0", "cummulativeQuoteQty": "0"}
            raise

    @staticmethod
    def avg_fill(order):
        qty = float(order.get("executedQty") or 0)
        quote = float(order.get("cummulativeQuoteQty") or 0)
        return qty, (quote / qty if qty else 0.0)

    def budget(self):
        """(bot money, money still free for new buys)."""
        realized = self.state.get("realized_usdt", 0.0)
        total = self.capital + (realized if self.compound else min(0.0, realized))
        in_use = sum(p["entry"] * p["qty"] for p in self.state["positions"].values())
        in_use += sum(o["limit"] * o["qty"] for o in self.state["orders"].values())
        return max(0.0, total), max(0.0, total - in_use)

    def triggers(self, t):
        """Distance of every tradable coin to its trigger: list of (distance, sym, trigger, price)."""
        prices = self.c.prices(self.cfg["symbols"])
        out = []
        for sym in self.cfg["symbols"]:
            if sym not in self.info or self.info[sym]["status"] != "TRADING" or sym not in prices:
                continue
            ks = [k for k in self.c.klines(sym, "1h", 30) if k["close_ms"] < t]
            if len(ks) < 24:
                continue
            trigger = max(k["close"] for k in ks[-24:]) * (1 - self.cfg["depth_pct"] / 100)
            out.append((prices[sym] / trigger - 1, sym, trigger, prices[sym]))
        out.sort()
        return out

    # ------------------------------------------------------------- trading steps
    def open_position(self, sym, qty_exec, px, fill_ms):
        base = self.info[sym]["base"]
        free = self.c.balances().get(base, {"free": 0.0})["free"]
        # if Binance took the fee in coins we hold a bit less; never count coins of another bot
        qty = min(qty_exec * (1 - self.cfg["fee_pct"] / 100), free)
        p = {"qty": qty, "entry": px, "opened_ms": int(fill_ms), "tp_id": None, "tp_price": None}
        self.state["positions"][sym] = p
        self.save()
        self.log(f"BOUGHT {sym} {qty:.8g} @ {px:.8g} ({qty * px:.2f} USDT)", alert=True)
        self.place_tp(sym, p)
        self.save()

    def place_tp(self, sym, p):
        price = self.px_str(sym, p["entry"] * (1 + self.cfg["take_profit_pct"] / 100), up=True)
        qty = self.qty_str(sym, p["qty"])
        if float(qty) <= 0 or float(qty) * float(price) < self.min_notional(sym):
            self.log(f"{sym}: {float(qty) * float(price):.2f} USDT is too small for a sell order, sold at the time limit")
            return
        try:
            o = self.c.new_order(symbol=sym, side="SELL", type="LIMIT", timeInForce="GTC", quantity=qty,
                                 price=price, newClientOrderId=self.cid("tp", sym))
            p["tp_id"], p["tp_price"] = o["orderId"], float(price)
            self.log(f"{sym}: take-profit sell waiting at {price}")
        except BinanceError as err:
            self.log(f"ALERT {sym}: take-profit order failed, retry next hour: {err}", alert=True)

    def close_position(self, sym, reason):
        """Cancel the take-profit (keeping anything it already sold) and sell the rest at market."""
        p = self.state["positions"][sym]
        sold_q, sold_quote = 0.0, 0.0
        if p.get("tp_id"):
            try:
                self.c.cancel_order(sym, p["tp_id"])
            except BinanceError:
                pass
            o = self.order_status(sym, p["tp_id"])
            q, px = self.avg_fill(o)
            sold_q, sold_quote = q, q * px
            p["tp_id"] = None
            if o.get("status") == "FILLED":
                self.record(sym, p, q, px, f"take-profit +{self.cfg['take_profit_pct']}%")
                del self.state["positions"][sym]
                self.save()
                return
        left = self.qty_str(sym, p["qty"] - sold_q)
        price_now = self.c.prices([sym]).get(sym, p["entry"])
        if float(left) > 0 and float(left) * price_now >= self.min_notional(sym):
            o = self.c.new_order(symbol=sym, side="SELL", type="MARKET", quantity=left,
                                 newClientOrderId=self.cid("out", sym))
            q, px = self.avg_fill(o)
            sold_q, sold_quote = sold_q + q, sold_quote + q * px
        elif float(left) > 0:
            self.log(f"{sym}: {float(left):.8g} left is below Binance's minimum order, left in the account")
        if sold_q > 0:
            self.record(sym, p, sold_q, sold_quote / sold_q, reason)
        else:
            self.state["cooldown"][sym] = self.clock() + self.cfg["cooldown_hours"] * HOUR_MS
        del self.state["positions"][sym]
        self.save()

    def check_orders(self):
        """Buy orders placed last hour: keep what filled, cancel the rest."""
        for sym in list(self.state["orders"]):
            od = self.state["orders"][sym]
            try:
                o = self.order_status(sym, od["order_id"])
                if o.get("status") not in DONE:
                    try:
                        self.c.cancel_order(sym, od["order_id"])
                    except BinanceError:
                        pass  # it may have filled a moment ago: read it again
                    o = self.order_status(sym, od["order_id"])
                    if o.get("status") not in DONE:
                        self.log(f"ALERT {sym}: could not cancel buy order {od['order_id']}, retry next hour", alert=True)
                        continue
                qty, px = self.avg_fill(o)
                del self.state["orders"][sym]
                self.save()
                if qty > 0:
                    self.open_position(sym, qty, px, int(o.get("updateTime") or self.clock()))
            except BinanceError as err:
                self.log(f"ALERT {sym}: checking the buy order failed: {err}", alert=True)

    def manage_positions(self):
        t = self.clock()
        for sym in list(self.state["positions"]):
            p = self.state["positions"][sym]
            try:
                if p.get("tp_id"):
                    o = self.order_status(sym, p["tp_id"])
                    status = o.get("status")
                    if status == "FILLED":
                        q, px = self.avg_fill(o)
                        self.record(sym, p, q, px, f"take-profit +{self.cfg['take_profit_pct']}%")
                        del self.state["positions"][sym]
                        self.save()
                        continue
                    if status in DONE:          # cancelled outside the bot: keep what it sold, re-place
                        q, px = self.avg_fill(o)
                        if q > 0:
                            self.record(sym, dict(p), q, px, "take-profit (part)")
                            p["qty"] = max(0.0, p["qty"] - q)
                        p["tp_id"] = None
                if t - p["opened_ms"] >= self.cfg["max_hours"] * HOUR_MS:
                    self.close_position(sym, f"time limit {self.cfg['max_hours']}h")
                    continue
                if not p.get("tp_id"):
                    self.place_tp(sym, p)
                self.save()
            except BinanceError as err:
                self.log(f"ALERT {sym}: managing the trade failed: {err}", alert=True)

    def place_orders(self):
        t = self.clock()
        total, free_budget = self.budget()
        cand = [x for x in self.triggers(t) if x[1] not in self.state["positions"] and x[1] not in self.state["orders"]
                and self.state["cooldown"].get(x[1], 0) <= t]
        sel = cand[: self.cfg["n_orders"]]
        if not sel:
            return
        usdt_free = self.c.balances().get("USDT", {"free": 0.0})["free"]
        cash = min(free_budget, usdt_free * 0.98)
        per = min(self.cfg["max_order_pct"] / 100 * total, cash / len(sel))
        for dist, sym, trigger, price in sel:
            if per < self.min_notional(sym):
                if cash >= self.min_notional(sym):
                    self.log(f"{per:.2f} USDT per order is below Binance's minimum: raise CAPITAL_USDT")
                break
            try:
                if dist <= 0:
                    o = self.c.new_order(symbol=sym, side="BUY", type="MARKET", quoteOrderQty=f"{per:.2f}",
                                         newClientOrderId=self.cid("buy", sym))
                    q, px = self.avg_fill(o)
                    if q > 0:
                        self.log(f"{sym}: {abs(dist) * 100:.1f}% under its trigger, bought at market")
                        self.open_position(sym, q, px, t)
                else:
                    limit = self.px_str(sym, trigger)
                    qty = self.qty_str(sym, per / float(limit))
                    if float(qty) < self.info[sym]["min_qty"] or float(qty) * float(limit) < self.min_notional(sym):
                        continue
                    o = self.c.new_order(symbol=sym, side="BUY", type="LIMIT", timeInForce="GTC", quantity=qty,
                                         price=limit, newClientOrderId=self.cid("buy", sym))
                    self.state["orders"][sym] = {"order_id": o["orderId"], "limit": float(limit), "qty": float(qty),
                                                 "placed_ms": t}
                self.save()
            except BinanceError as err:
                self.log(f"ALERT {sym}: buy order failed: {err}", alert=True)

    def cycle(self):
        self.check_orders()
        self.manage_positions()
        self.place_orders()
        self.save()

    # ------------------------------------------------------------- commands
    def summary(self):
        s = self.state
        total, free_budget = self.budget()
        n = s.get("trades", 0)
        lines = [f"Bot money {total:.2f} USDT ({'profits reinvested' if self.compound else 'profits set aside'}), "
                 f"free for new buys {free_budget:.2f}",
                 f"Closed trades {n}, wins {s.get('wins', 0)} ({(s.get('wins', 0) / n * 100 if n else 0):.0f}%), "
                 f"profit after fees {s.get('realized_usdt', 0.0):+.2f} USDT"]
        for sym, p in s["positions"].items():
            left = (p["opened_ms"] + self.cfg["max_hours"] * HOUR_MS - self.clock()) / HOUR_MS
            lines.append(f"  holding {sym}: {p['qty']:.8g} @ {p['entry']:.8g}, sell at {p.get('tp_price') or '-'}, "
                         f"time limit in {max(0, left):.0f}h")
        for sym, o in s["orders"].items():
            lines.append(f"  waiting to buy {sym}: {o['qty']:.8g} @ {o['limit']:.8g}")
        return "\n".join(lines)

    def cancel_buys(self):
        """Cancel the bot's waiting buy orders (anything already bought keeps its take-profit on Binance)."""
        for sym, od in list(self.state["orders"].items()):
            try:
                self.c.cancel_order(sym, od["order_id"])
            except BinanceError:
                pass
            try:
                o = self.order_status(sym, od["order_id"])
                q, px = self.avg_fill(o)
                del self.state["orders"][sym]
                self.save()
                if q > 0:
                    self.open_position(sym, q, px, int(o.get("updateTime") or self.clock()))
            except BinanceError as err:
                self.log(f"ALERT {sym}: cancel buy order failed, cancel it in the Binance app: {err}", alert=True)
        self.save()

    def panic(self):
        self.cancel_buys()
        for sym in list(self.state["positions"]):
            try:
                self.close_position(sym, "panic")
            except BinanceError as err:
                self.log(f"ALERT {sym}: panic sell failed, sell it in the Binance app: {err}", alert=True)
        self.save()


def telegram():
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        return None

    def send(msg):
        try:
            data = urllib.parse.urlencode({"chat_id": chat, "text": "Panic buyer: " + msg}).encode()
            urllib.request.urlopen(f"https://api.telegram.org/bot{token}/sendMessage", data=data, timeout=10)
        except Exception:  # noqa: BLE001 - alerts must never stop trading
            pass
    return send


def load_cfg():
    path = os.path.join(HERE, "panic_config.json")
    cfg = dict(DEFAULT_CFG)
    if os.path.exists(path):
        with open(path) as f:
            cfg.update(json.load(f))
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description="Panic buyer (Binance spot)")
    ap.add_argument("command", choices=["check", "run", "status", "pause", "panic"])
    a = ap.parse_args(argv)
    cfg = load_cfg()
    client = BinanceSpot()
    capital = float(os.environ.get("CAPITAL_USDT", "0") or 0)
    compound = os.environ.get("COMPOUND", "no").strip().lower() in ("yes", "y", "true", "1")
    bot = PanicBuyer(client, cfg, capital, compound=compound, notify=telegram())

    if a.command == "status":
        print(bot.summary())
        return 0

    try:
        client.sync_time()
        bot.load_info()
    except BinanceError as err:
        print(f"Binance not reachable or key rejected: {err}")
        print("CHECK FAILED" if a.command == "check" else "Not started, will retry.")
        return 1
    if a.command == "check":
        print(f"Binance reachable, clock offset {client.offset_ms} ms, {len(bot.info)} of {len(cfg['symbols'])} coins found")
        try:
            bal = client.balances()
        except BinanceError as err:
            print(f"The API key does not work: {err}")
            print("Check: Spot trading enabled, this server's IP allowed on the key, key and secret pasted fully.")
            print("CHECK FAILED")
            return 1
        print(f"Key works. USDT free {bal.get('USDT', {}).get('free', 0):.2f}, BNB {bal.get('BNB', {}).get('free', 0):.4f}")
        print(f"CAPITAL_USDT (the most this bot may use): {capital:.2f}, profits {'reinvested' if compound else 'set aside'}")
        near = bot.triggers(now_ms())[: cfg["n_orders"]]
        print("Coins closest to their buy trigger now: " + ", ".join(f"{s[:-4]} {d * 100:+.1f}%" for d, s, _, _ in near))
        if capital < 5 * cfg["n_orders"]:
            print(f"CAPITAL_USDT must be at least {5 * cfg['n_orders']} (8 orders of 5 USDT minimum).")
            print("CHECK FAILED")
            return 1
        print("CHECK PASSED")
        return 0
    if a.command == "pause":
        bot.cancel_buys()
        print("Waiting buy orders cancelled. Coins already bought keep their take-profit sell on Binance.")
        return 0
    if a.command == "panic":
        bot.panic()
        print("All panic-buyer orders cancelled and its coins sold.")
        return 0
    if capital < 5 * cfg["n_orders"]:
        print("CAPITAL_USDT is missing or too small in /etc/panic-buyer.env, refusing to trade.")
        return 1
    bot.log(f"Started. Capital {capital:.2f} USDT, {len(bot.info)} coins, profits "
            f"{'reinvested' if compound else 'set aside'}.", alert=True)
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
        except Exception as err:  # noqa: BLE001 - keep running; sells wait on Binance anyway
            bot.log(f"ALERT: unexpected error: {err!r}", alert=True)
        time.sleep(3600 - (time.time() % 3600) + 10)   # 10 s after each hourly candle closes


if __name__ == "__main__":
    sys.exit(main())
