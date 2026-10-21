"""A fake Binance spot exchange driven by hourly candles (no network, no real orders).

The bot runs 10 s after each candle opens; its market orders fill at that candle's open. Resting limit orders
are then matched against the candle: a buy fills if the low trades `eps` through its price, a sell if the high
does (same rule as the backtest). Fees are taken like Binance without BNB: buys in the coin, sells in USDT.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from binance_spot import BinanceError  # noqa: E402

OPEN = ("NEW", "PARTIALLY_FILLED")


class FakeBinance:
    def __init__(self, candles, start=0, usdt=1000.0, fee=0.001, eps=0.001, tick="0.0001", step="0.0001",
                 min_notional=5.0):
        self.k = candles                    # {symbol: [{"open_ms","open","high","low","close","close_ms"}, ...]}
        self.i = start
        self.fee, self.eps = fee, eps
        self.tick, self.step, self.min_notional = tick, step, min_notional
        self.bal = {"USDT": {"free": float(usdt), "locked": 0.0}}
        self.orders = {}
        self.next_id = 1
        self.status = {s: "TRADING" for s in candles}
        self.partial = {}                   # orderId -> share of the order to fill (tests)
        self.calls = []
        self.ref = next(iter(candles))

    # --------------------------------------------------------------- time
    def clock(self):
        return self.k[self.ref][self.i]["open_ms"] + 10_000

    def price(self, sym):
        return self.k[sym][self.i]["open"]

    # --------------------------------------------------------------- public endpoints
    def sync_time(self):
        return 0

    def exchange_info(self, symbols):
        return {s: {"status": self.status[s], "base": s[:-4], "quote": "USDT", "tick": self.tick, "step": self.step,
                    "min_qty": float(self.step), "min_notional": self.min_notional,
                    "order_types": ["LIMIT", "MARKET", "STOP_LOSS_LIMIT"]} for s in symbols if s in self.k}

    def klines(self, symbol, interval="1h", limit=30):
        assert interval == "1h"
        rows = self.k[symbol][: self.i + 1]            # includes the candle still forming, like Binance
        return [dict(r) for r in rows[-limit:]]

    def prices(self, symbols):
        return {s: self.price(s) for s in symbols if s in self.k}

    # --------------------------------------------------------------- account
    def _a(self, asset):
        return self.bal.setdefault(asset, {"free": 0.0, "locked": 0.0})

    def balances(self):
        return {a: dict(v) for a, v in self.bal.items()}

    def get_order(self, symbol, oid):
        self.calls.append(("get", symbol, oid))
        if oid not in self.orders:
            raise BinanceError("HTTP 400: Order does not exist.", -2013)
        return dict(self.orders[oid])

    def cancel_order(self, symbol, oid):
        self.calls.append(("cancel", symbol, oid))
        o = self.orders.get(oid)
        if o is None or o["status"] not in OPEN:
            raise BinanceError("HTTP 400: Unknown order sent.", -2011)
        left = float(o["origQty"]) - float(o["executedQty"])
        if o["side"] == "BUY":
            self._a("USDT")["locked"] -= left * float(o["price"])
            self._a("USDT")["free"] += left * float(o["price"])
        else:
            self._a(symbol[:-4])["locked"] -= left
            self._a(symbol[:-4])["free"] += left
        o["status"] = "CANCELED"
        o["updateTime"] = self.clock()
        return dict(o)

    def _fill(self, o, qty, px, when=None):
        sym, base = o["symbol"], o["symbol"][:-4]
        if o["side"] == "BUY":
            if o["type"] == "LIMIT":
                self._a("USDT")["locked"] -= qty * float(o["price"])
                self._a("USDT")["free"] += qty * (float(o["price"]) - px)     # price improvement back
            else:
                self._a("USDT")["free"] -= qty * px
            self._a(base)["free"] += qty * (1 - self.fee)
        else:
            if o["type"] == "LIMIT":
                self._a(base)["locked"] -= qty
            else:
                self._a(base)["free"] -= qty
            self._a("USDT")["free"] += qty * px * (1 - self.fee)
        ex = float(o["executedQty"]) + qty
        o["executedQty"] = repr(ex)
        o["cummulativeQuoteQty"] = repr(float(o["cummulativeQuoteQty"]) + qty * px)
        o["status"] = "FILLED" if ex >= float(o["origQty"]) - 1e-12 else "PARTIALLY_FILLED"
        o["updateTime"] = when if when is not None else self.clock()
        self.calls.append(("fill", sym, o["orderId"], qty, px))

    def new_order(self, symbol, side, type, quantity=None, price=None, quoteOrderQty=None, timeInForce=None,
                  newClientOrderId=None, **kw):
        self.calls.append(("new", symbol, side, type, quantity, price, quoteOrderQty))
        if self.status.get(symbol) != "TRADING":
            raise BinanceError("HTTP 400: market closed", -1013)
        now_px = self.price(symbol)
        base = symbol[:-4]
        oid = self.next_id
        self.next_id += 1
        if type == "MARKET" and side == "BUY":
            quantity = repr(float(quoteOrderQty) / now_px)
        o = {"orderId": oid, "symbol": symbol, "side": side, "type": type, "status": "NEW", "origQty": quantity,
             "price": price or "0", "executedQty": "0", "cummulativeQuoteQty": "0", "updateTime": self.clock(),
             "clientOrderId": newClientOrderId}
        qty = float(quantity)
        if qty * (float(price) if price else now_px) < self.min_notional:
            raise BinanceError("HTTP 400: Filter failure: NOTIONAL", -1013)
        if side == "BUY":
            need = qty * (float(price) if type == "LIMIT" else now_px)
            if self._a("USDT")["free"] + 1e-9 < need:
                raise BinanceError("HTTP 400: Account has insufficient balance for requested action.", -2010)
            if type == "LIMIT":
                self._a("USDT")["free"] -= need
                self._a("USDT")["locked"] += need
        else:
            if self._a(base)["free"] + 1e-12 < qty:
                raise BinanceError("HTTP 400: Account has insufficient balance for requested action.", -2010)
            if type == "LIMIT":
                self._a(base)["free"] -= qty
                self._a(base)["locked"] += qty
        self.orders[oid] = o
        if type == "MARKET":
            self._fill(o, qty, now_px)
        elif (side == "BUY" and float(price) >= now_px) or (side == "SELL" and float(price) <= now_px):
            self._fill(o, qty, now_px)                                     # marketable limit: fills at once
        return dict(o)

    # --------------------------------------------------------------- the market moves
    def advance(self):
        """Match resting orders against the current candle, then move to the next candle."""
        for o in list(self.orders.values()):
            if o["status"] not in OPEN:
                continue
            k = self.k[o["symbol"]][self.i]
            p = float(o["price"])
            left = float(o["origQty"]) - float(o["executedQty"])
            share = self.partial.pop(o["orderId"], 1.0)
            mid = k["open_ms"] + 30 * 60 * 1000          # resting orders fill somewhere inside the hour
            if o["side"] == "BUY" and k["low"] <= p * (1 - self.eps):
                self._fill(o, left * share, min(p, k["open"]), mid)
            elif o["side"] == "SELL" and k["high"] >= p * (1 + self.eps):
                self._fill(o, left * share, max(p, k["open"]), mid)
        self.i += 1

    def value(self):
        """Account value in USDT at the current price."""
        tot = self.bal["USDT"]["free"] + self.bal["USDT"]["locked"]
        for a, v in self.bal.items():
            if a != "USDT" and a + "USDT" in self.k:
                tot += (v["free"] + v["locked"]) * self.price(a + "USDT")
        return tot
