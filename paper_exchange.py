"""Paper trading for the Panic Buyer: real Binance prices, pretend money and pretend orders. No API key needed,
nothing is ever sent to your account.

Prices, candles, coin rules and the order book come from Binance's public market data. Orders are simulated
with the same rules as the backtest:
  - a MARKET buy fills at the current ask, a MARKET sell at the current bid;
  - a resting LIMIT order fills when a later 1-hour candle trades 0.1% through its price (a buy when the low is
    at least 0.1% under it, a sell when the high is at least 0.1% over it), at the limit or the candle's open
    if that is better;
  - fee 0.1% per side: in the coin on buys, in USDT on sells (like Binance without BNB).
The pretend account (balances and orders) lives in paper_account.json in the bot's data folder.
"""
import json
import os
import time

from binance_spot import BinanceError, BinanceSpot

HOUR_MS = 3600 * 1000
OPEN = ("NEW", "PARTIALLY_FILLED")


class PaperExchange:
    def __init__(self, root, usdt, market=None, fee=0.001, eps=0.001, clock=None):
        self.m = market or BinanceSpot(key="", secret="")     # public endpoints only
        self.fee, self.eps = fee, eps
        self.clock = clock or (lambda: int(time.time() * 1000))
        self.path = os.path.join(root, "paper_account.json")
        os.makedirs(root, exist_ok=True)
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.a = json.load(f)
        else:
            self.a = {"bal": {"USDT": {"free": float(usdt), "locked": 0.0}}, "orders": {}, "next_id": 1,
                      "start_usdt": float(usdt), "started_ms": self.clock()}
            self._save()
        self.offset_ms = 0

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.a, f, indent=1)
        os.replace(tmp, self.path)

    def _b(self, asset):
        return self.a["bal"].setdefault(asset, {"free": 0.0, "locked": 0.0})

    # --------------------------------------------------------------- public market data (real)
    def sync_time(self):
        return self.m.sync_time()

    def exchange_info(self, symbols):
        return self.m.exchange_info(symbols)

    def klines(self, symbol, interval="1h", limit=30):
        return self.m.klines(symbol, interval, limit)

    def prices(self, symbols):
        return self.m.prices(symbols)

    def book_ticker(self, symbol):
        return self.m.book_ticker(symbol)

    # --------------------------------------------------------------- pretend account
    def balances(self):
        self._match()
        return {k: dict(v) for k, v in self.a["bal"].items()}

    def get_order(self, symbol, oid):
        self._match()
        o = self.a["orders"].get(str(oid))
        if o is None:
            raise BinanceError("PAPER: Order does not exist.", -2013)
        return dict(o)

    def get_order_by_cid(self, symbol, cid):
        for o in self.a["orders"].values():
            if o.get("clientOrderId") == cid:
                return dict(o)
        raise BinanceError("PAPER: Order does not exist.", -2013)

    def cancel_order(self, symbol, oid):
        self._match()
        o = self.a["orders"].get(str(oid))
        if o is None or o["status"] not in OPEN:
            raise BinanceError("PAPER: Unknown order sent.", -2011)
        left = float(o["origQty"]) - float(o["executedQty"])
        if o["side"] == "BUY":
            amt = left * float(o["price"])
            self._b("USDT")["locked"] -= amt
            self._b("USDT")["free"] += amt
        else:
            self._b(o["symbol"][:-4])["locked"] -= left
            self._b(o["symbol"][:-4])["free"] += left
        o["status"], o["updateTime"] = "CANCELED", self.clock()
        self._save()
        return dict(o)

    def _fill(self, o, qty, px, when):
        base = o["symbol"][:-4]
        if o["side"] == "BUY":
            if o["type"] == "LIMIT":
                self._b("USDT")["locked"] -= qty * float(o["price"])
                self._b("USDT")["free"] += qty * (float(o["price"]) - px)
            else:
                self._b("USDT")["free"] -= qty * px
            self._b(base)["free"] += qty * (1 - self.fee)
        else:
            if o["type"] == "LIMIT":
                self._b(base)["locked"] -= qty
            else:
                self._b(base)["free"] -= qty
            self._b("USDT")["free"] += qty * px * (1 - self.fee)
        ex = float(o["executedQty"]) + qty
        o["executedQty"] = repr(ex)
        o["cummulativeQuoteQty"] = repr(float(o["cummulativeQuoteQty"]) + qty * px)
        o["status"] = "FILLED" if ex >= float(o["origQty"]) - 1e-12 else "PARTIALLY_FILLED"
        o["updateTime"] = int(when)

    def new_order(self, symbol, side, type, quantity=None, price=None, quoteOrderQty=None, timeInForce=None,
                  newClientOrderId=None, **kw):
        self._match()
        bid, ask = self.m.book_ticker(symbol)
        now = self.clock()
        if type == "MARKET" and side == "BUY":
            quantity = repr(float(quoteOrderQty) / ask)
        qty = float(quantity)
        ref = float(price) if type == "LIMIT" else (ask if side == "BUY" else bid)
        if qty * ref < 5.0:
            raise BinanceError("PAPER: Filter failure: NOTIONAL", -1013)
        base = symbol[:-4]
        if side == "BUY":
            need = qty * ref
            if self._b("USDT")["free"] + 1e-9 < need:
                raise BinanceError("PAPER: Account has insufficient balance for requested action.", -2010)
            if type == "LIMIT":
                self._b("USDT")["free"] -= need
                self._b("USDT")["locked"] += need
        else:
            if self._b(base)["free"] + 1e-12 < qty:
                raise BinanceError("PAPER: Account has insufficient balance for requested action.", -2010)
            if type == "LIMIT":
                self._b(base)["free"] -= qty
                self._b(base)["locked"] += qty
        oid = self.a["next_id"]
        self.a["next_id"] += 1
        o = {"orderId": oid, "symbol": symbol, "side": side, "type": type, "status": "NEW", "origQty": quantity,
             "price": price or "0", "executedQty": "0", "cummulativeQuoteQty": "0", "updateTime": now,
             "clientOrderId": newClientOrderId, "placed_ms": now}
        self.a["orders"][str(oid)] = o
        if type == "MARKET":
            self._fill(o, qty, ask if side == "BUY" else bid, now)
        elif side == "BUY" and float(price) >= ask:
            self._fill(o, qty, ask, now)                 # marketable limit: fills at once
        elif side == "SELL" and float(price) <= bid:
            self._fill(o, qty, bid, now)
        self._save()
        return dict(o)

    def _match(self):
        """Fill resting limit orders against the 1-hour candles that closed since they were placed."""
        now = self.clock()
        changed = False
        for o in self.a["orders"].values():
            if o["status"] not in OPEN or o["type"] != "LIMIT":
                continue
            start = o.get("matched_to", o["placed_ms"] - o["placed_ms"] % HOUR_MS)
            if now - start < HOUR_MS:
                continue
            ks = [k for k in self.m.klines(o["symbol"], "1h", min(1000, int((now - start) / HOUR_MS) + 2))
                  if k["open_ms"] >= start and k["close_ms"] < now]
            p = float(o["price"])
            for k in ks:
                left = float(o["origQty"]) - float(o["executedQty"])
                if o["side"] == "BUY" and k["low"] <= p * (1 - self.eps):
                    self._fill(o, left, min(p, k["open"]), k["open_ms"] + 30 * 60 * 1000)
                elif o["side"] == "SELL" and k["high"] >= p * (1 + self.eps):
                    self._fill(o, left, max(p, k["open"]), k["open_ms"] + 30 * 60 * 1000)
                o["matched_to"] = k["close_ms"] + 1
                changed = True
                if o["status"] == "FILLED":
                    break
        if changed:
            self.a["orders"] = {k: v for k, v in self.a["orders"].items()
                                if v["status"] in OPEN or now - v["updateTime"] < 14 * 24 * HOUR_MS}
            self._save()

    def value(self):
        """Pretend account value in USDT at current prices."""
        bal = self.a["bal"]
        coins = [a + "USDT" for a, v in bal.items() if a != "USDT" and v["free"] + v["locked"] > 0]
        px = self.m.prices(coins) if coins else {}
        tot = bal["USDT"]["free"] + bal["USDT"]["locked"]
        for a, v in bal.items():
            if a != "USDT":
                tot += (v["free"] + v["locked"]) * px.get(a + "USDT", 0.0)
        return tot
