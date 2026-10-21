"""Tests for the live pump-rider with a fake Binance (no network, no real orders)."""

import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import rider as R  # noqa: E402
from binance_spot import BinanceError, round_step  # noqa: E402

FIVE = R.FIVE_MS
T0 = (1_790_000_000_000 // FIVE) * FIVE + 8000  # 8 s after a candle closed
SYM = "AAAUSDT"


def kl(i, o, h, l, c, v):
    start = T0 - 8000 - (60 - i) * FIVE
    return {"open_ms": start, "open": o, "high": h, "low": l, "close": c, "volume": v, "close_ms": start + FIVE - 1}


def pump_candles():
    rows = [kl(i, 100, 100.2, 99.8, 100, 10) for i in range(57)]
    rows += [kl(57, 100, 100.6, 99.9, 100.5, 12), kl(58, 100.5, 101.3, 100.4, 101.2, 15),
             kl(59, 101.2, 102.6, 101.1, 102.5, 50)]
    return rows


class FakeBinance:
    def __init__(self):
        self.now = T0
        self.px = {SYM: 102.5, "BTCUSDT": 70000.0}
        self.bal = {"USDT": {"free": 1000.0, "locked": 0.0}}
        self.k5 = {SYM: pump_candles()}
        self.k1d = [{"open_ms": T0 - (80 - d) * 86400000, "open": 60000 + d * 100, "high": 0, "low": 0,
                     "close": 60000 + d * 100, "volume": 1, "close_ms": T0 - (79 - d) * 86400000 - 1} for d in range(80)]
        self.orders, self.next_id, self.log = {}, 1, []

    def exchange_info(self, symbols):
        info = {s: {"status": "TRADING", "base": s[:-4], "quote": "USDT", "tick": "0.0001", "step": "0.001",
                    "min_qty": 0.001, "min_notional": 5.0, "order_types": ["LIMIT", "MARKET", "STOP_LOSS",
                                                                           "STOP_LOSS_LIMIT"]} for s in symbols}
        return {s: v for s, v in info.items() if s in (SYM, "BTCUSDT")}

    def klines(self, symbol, interval="5m", limit=100):
        if interval == "1d":
            return self.k1d[-limit:]
        return self.k5.get(symbol, [])[-limit:]

    def prices(self, symbols):
        return {s: p for s, p in self.px.items() if s in symbols}

    def balances(self):
        return {k: dict(v) for k, v in self.bal.items()}

    def get_order(self, symbol, oid):
        return dict(self.orders[oid])

    def cancel_order(self, symbol, oid):
        o = self.orders[oid]
        if o["status"] != "NEW":
            raise BinanceError("Unknown order", -2011)
        o["status"] = "CANCELED"
        if o["side"] == "BUY":
            self.bal["USDT"]["free"] += float(o["quantity"]) * float(o["price"])
        return o

    def new_order(self, **p):
        oid = self.next_id
        self.next_id += 1
        o = dict(p, orderId=oid, status="NEW", executedQty="0", cummulativeQuoteQty="0")
        self.orders[oid] = o
        self.log.append(o)
        base = p["symbol"][:-4]
        if p["type"] == "MARKET":
            price = self.px[p["symbol"]]
            if p["side"] == "BUY":
                qty = float(p["quoteOrderQty"]) / price
                self.bal["USDT"]["free"] -= qty * price
                self.bal.setdefault(base, {"free": 0.0, "locked": 0.0})["free"] += qty
            else:
                qty = float(p["quantity"])
                self.bal[base]["free"] -= qty
                self.bal["USDT"]["free"] += qty * price
            o.update(status="FILLED", executedQty=str(qty), cummulativeQuoteQty=str(qty * price))
        elif p["side"] == "BUY":
            self.bal["USDT"]["free"] -= float(p["quantity"]) * float(p["price"])
        return {"orderId": oid, **o}

    def fill(self, oid, price):
        o = self.orders[oid]
        qty = float(o["quantity"])
        base = o["symbol"][:-4]
        o.update(status="FILLED", executedQty=str(qty), cummulativeQuoteQty=str(qty * price))
        if o["side"] == "BUY":
            self.bal["USDT"]["free"] += qty * float(o["price"]) - qty * price
            self.bal.setdefault(base, {"free": 0.0, "locked": 0.0})["free"] += qty
        else:
            self.bal[base]["free"] -= qty
            self.bal["USDT"]["free"] += qty * price


class RiderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fake = FakeBinance()
        self.lines, self.alerts = [], []
        self.bot = self.make()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, capital=500, **cfg):
        c = dict(R.DEFAULT_CFG, symbols=[SYM, "BTCUSDT"], **cfg)
        bot = R.Rider(self.fake, c, capital, root=self.tmp.name, out=self.lines.append,
                      notify=self.alerts.append, clock=lambda: self.fake.now)
        bot.load_info()
        return bot

    def stop_orders(self):
        return [o for o in self.fake.log if o["side"] == "SELL" and o["type"].startswith("STOP")]

    def test_signal_sets_stop_buy_one_percent_higher(self):
        self.bot.cycle()
        a = self.bot.state["armed"][SYM]
        self.assertAlmostEqual(a["trigger"], 102.5 * 1.01)
        o = self.fake.orders[a["order_id"]]
        self.assertEqual((o["side"], o["type"]), ("BUY", "STOP_LOSS_LIMIT"))
        self.assertEqual(o["stopPrice"], "103.525")
        self.assertEqual(o["price"], round_step(102.5 * 1.01 * 1.003, "0.0001"))
        self.assertLessEqual(float(o["quantity"]) * float(o["price"]), 100.0 + 1e-9)  # 500 / 5

    def test_no_trade_when_bitcoin_downtrend(self):
        for d in self.fake.k1d:
            d["close"] = 90000 - d["open_ms"] / 1e9
        self.fake.k1d[-1]["close"] = 1000.0
        self.bot.cycle()
        self.assertEqual(self.bot.state["armed"], {})

    def test_fill_places_stop_on_binance(self):
        self.bot.cycle()
        oid = self.bot.state["armed"][SYM]["order_id"]
        self.fake.fill(oid, 103.6)
        self.fake.now += FIVE
        self.fake.px[SYM] = 103.8
        self.bot.cycle()
        p = self.bot.state["positions"][SYM]
        self.assertAlmostEqual(p["entry"], 103.6)
        stop = self.stop_orders()[-1]
        self.assertEqual(stop["type"], "STOP_LOSS")
        self.assertAlmostEqual(float(stop["stopPrice"]), 103.6 * 0.95, places=3)
        self.assertTrue(any("BOUGHT" in a for a in self.alerts))

    def open_trade(self):
        self.bot.cycle()
        oid = self.bot.state["armed"][SYM]["order_id"]
        self.fake.fill(oid, 103.6)
        self.fake.now += FIVE
        self.fake.px[SYM] = 103.8
        self.bot.cycle()
        return self.bot.state["positions"][SYM]

    def test_trailing_stop_raised_and_old_one_cancelled(self):
        self.open_trade()
        first = self.stop_orders()[-1]
        self.fake.now += FIVE
        start = self.fake.now - 8000 - FIVE
        self.fake.k5[SYM].append({"open_ms": start, "open": 104, "high": 125, "low": 103.9, "close": 124,
                                  "volume": 30, "close_ms": start + FIVE - 1})
        self.fake.px[SYM] = 124
        self.bot.cycle()
        self.assertEqual(self.fake.orders[first["orderId"]]["status"], "CANCELED")
        new = self.stop_orders()[-1]
        self.assertAlmostEqual(float(new["stopPrice"]), 125 * 0.92, places=3)
        self.assertAlmostEqual(self.bot.state["positions"][SYM]["stop"], 125 * 0.92)

    def test_stop_filled_records_trade(self):
        self.open_trade()
        stop = self.stop_orders()[-1]
        self.fake.fill(stop["orderId"], 98.4)
        self.fake.now += FIVE
        self.fake.px[SYM] = 98
        self.bot.cycle()
        self.assertNotIn(SYM, self.bot.state["positions"])
        with open(os.path.join(self.tmp.name, "rider_trades.csv")) as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(rows[-1]["reason"], "stop on Binance")
        self.assertLess(float(rows[-1]["pnl_usdt_before_fees"]), 0)

    def test_stop_buy_expires_without_trade(self):
        self.bot.cycle()
        oid = self.bot.state["armed"][SYM]["order_id"]
        self.fake.now += 61 * 60 * 1000
        self.fake.k5[SYM] = [dict(k, volume=10) for k in self.fake.k5[SYM]]  # no new signal
        self.bot.cycle()
        self.assertEqual(self.fake.orders[oid]["status"], "CANCELED")
        self.assertEqual(self.bot.state["armed"], {})
        self.assertEqual(self.bot.state["positions"], {})

    def test_time_limit_sells_at_market(self):
        self.open_trade()
        self.fake.now += 25 * 3600 * 1000
        self.fake.px[SYM] = 104.5
        self.bot.cycle()
        self.assertNotIn(SYM, self.bot.state["positions"])
        sells = [o for o in self.fake.log if o["side"] == "SELL" and o["type"] == "MARKET"]
        self.assertEqual(len(sells), 1)

    def test_already_above_trigger_buys_at_market(self):
        self.fake.px[SYM] = 103.6  # above 103.525 trigger, under the 0.3% limit
        self.bot.cycle()
        self.assertIn(SYM, self.bot.state["positions"])
        self.assertEqual(self.fake.log[0]["type"], "MARKET")

    def test_too_late_does_not_chase(self):
        self.fake.px[SYM] = 105.0
        self.bot.cycle()
        self.assertEqual(self.bot.state["positions"], {})
        self.assertEqual(self.bot.state["armed"], {})

    def test_same_signal_not_used_twice(self):
        self.bot.cycle()
        oid = self.bot.state["armed"][SYM]["order_id"]
        self.fake.now += 61 * 60 * 1000
        self.bot.cycle()  # same last candle in the fake data: must not re-arm
        self.assertEqual(self.fake.orders[oid]["status"], "CANCELED")
        self.assertEqual(self.bot.state["armed"], {})

    def test_weekly_halt(self):
        self.bot.cycle()
        self.bot.state["armed"] = {}
        self.bot.state["last_signal"] = {}
        self.bot.state["week_start"] = 2000.0  # account is 1000 now: -50%
        self.fake.now += FIVE
        self.bot.cycle()
        self.assertEqual(self.bot.state["armed"], {})
        self.assertTrue(any("no new trades" in a for a in self.alerts))

    def test_panic_cancels_and_sells(self):
        self.open_trade()
        self.bot.panic()
        self.assertEqual(self.bot.state["positions"], {})
        self.assertTrue(any(o["type"] == "MARKET" and o["side"] == "SELL" for o in self.fake.log))

    def test_capital_limit_respected(self):
        bot = self.make(capital=100)
        bot.cycle()
        o = self.fake.orders[bot.state["armed"][SYM]["order_id"]]
        self.assertLessEqual(float(o["quantity"]) * float(o["price"]), 20.0 + 1e-9)

    def test_round_step(self):
        self.assertEqual(round_step(1.23456, "0.001"), "1.234")
        self.assertEqual(round_step(1.23456, "0.001", up=True), "1.235")
        self.assertEqual(round_step(123.9, "10"), "120")
        self.assertEqual(round_step(0.000123456, "0.00000100"), "0.000123")

    def test_client_has_no_withdraw(self):
        import binance_spot
        names = " ".join(dir(binance_spot.BinanceSpot)).lower()
        for bad in ("withdraw", "transfer", "margin", "futures", "convert"):
            self.assertNotIn(bad, names)


if __name__ == "__main__":
    unittest.main()
