"""Tests for the live Panic Buyer against a fake Binance (no network, no real orders)."""
import csv
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import panic_buyer as PB  # noqa: E402
from fake_binance import FakeBinance  # noqa: E402

H = 3600 * 1000
T0 = (1_790_000_000_000 // H) * H


def series(rows):
    return [{"open_ms": T0 + i * H, "open": o, "high": h, "low": lo, "close": c, "close_ms": T0 + (i + 1) * H - 1}
            for i, (o, h, lo, c) in enumerate(rows)]


def flat(p, n):
    return [(p, p * 1.001, p * 0.999, p)] * n


def make(candles, start, usdt=1000.0, capital=1000.0, compound=False, **cfg_over):
    fake = FakeBinance(candles, start=start, usdt=usdt)
    cfg = dict(PB.DEFAULT_CFG)
    cfg["symbols"] = list(candles)
    cfg.update(cfg_over)
    root = tempfile.mkdtemp()
    logs = []
    bot = PB.PanicBuyer(fake, cfg, capital, compound=compound, root=root, out=logs.append, clock=fake.clock)
    bot.load_info()
    bot.logs = logs
    return fake, bot


def ten_coins(n=40, start=26):
    """10 coins: flat at 100, then at `start` coin j opens at 100-j (closer to its 88 trigger as j grows)."""
    out = {}
    for j, name in enumerate("ABCDEFGHIJ"):
        rows = flat(100.0, start) + [(100.0 - j, 100.5 - j, 99.5 - j, 100.0 - j)] * (n - start)
        out[name * 3 + "USDT"] = series(rows)
    return out


class Placement(unittest.TestCase):
    def test_eight_funded_orders_on_closest_coins(self):
        fake, bot = make(ten_coins(), 26)
        bot.cycle()
        orders = bot.state["orders"]
        self.assertEqual(set(orders), {c * 3 + "USDT" for c in "CDEFGHIJ"})      # the 8 closest
        total = sum(o["qty"] * o["limit"] for o in orders.values())
        self.assertLessEqual(total, 1000.0)
        for o in orders.values():
            self.assertAlmostEqual(o["limit"], 88.0, places=3)               # 100 * (1 - 12%)
            self.assertLessEqual(o["qty"] * o["limit"], 200.0 + 1e-6)       # <= 20% of capital
        self.assertAlmostEqual(fake.bal["USDT"]["locked"], total, places=6)

    def test_forming_candle_is_ignored(self):
        c = {"AAAUSDT": series(flat(100.0, 26) + [(100.0, 300.0, 99.0, 250.0)] * 4)}
        fake, bot = make(c, 26)
        bot.cycle()
        self.assertAlmostEqual(bot.state["orders"]["AAAUSDT"]["limit"], 88.0, places=3)

    def test_respects_shared_usdt_and_minimum(self):
        fake, bot = make(ten_coins(), 26, usdt=300.0, capital=1000.0)
        bot.cycle()
        self.assertLessEqual(sum(o["qty"] * o["limit"] for o in bot.state["orders"].values()), 300 * 0.98 + 1e-6)
        fake, bot = make(ten_coins(), 26, usdt=30.0, capital=30.0)
        bot.cycle()
        self.assertEqual(bot.state["orders"], {})                           # 3.7 USDT per order < Binance minimum

    def test_market_buy_when_already_under_trigger(self):
        c = {"AAAUSDT": series(flat(100.0, 26) + [(85.0, 85.5, 84.5, 85.0)] * 4)}
        fake, bot = make(c, 26)
        bot.cycle()
        self.assertIn("AAAUSDT", bot.state["positions"])
        p = bot.state["positions"]["AAAUSDT"]
        self.assertAlmostEqual(p["entry"], 85.0, places=6)
        tp = fake.orders[p["tp_id"]]
        self.assertEqual((tp["side"], tp["type"], tp["status"]), ("SELL", "LIMIT", "NEW"))
        self.assertGreaterEqual(float(tp["price"]), 85.0 * 1.015)


class TradeLife(unittest.TestCase):
    def crash_then(self, after_rows):
        rows = flat(100.0, 26) + [(100.0, 100.2, 87.0, 88.5)] + after_rows
        return {"AAAUSDT": series(rows)}

    def test_fill_then_take_profit_then_cooldown(self):
        fake, bot = make(self.crash_then([(88.5, 90.0, 88.4, 89.9)] + flat(95.0, 40)), 26)
        bot.cycle(); fake.advance()                       # order at 88 fills in the crash candle
        bot.cycle()
        p = bot.state["positions"]["AAAUSDT"]
        self.assertAlmostEqual(p["entry"], 88.0, places=6)
        self.assertAlmostEqual(p["qty"], float(fake.orders[1]["executedQty"]) * 0.999, places=9)
        fake.advance()                                    # next candle high 90 > 88*1.015*1.001: take-profit
        bot.cycle()
        self.assertNotIn("AAAUSDT", bot.state["positions"])
        self.assertEqual(bot.state["trades"], 1)
        self.assertGreater(bot.state["realized_usdt"], 0)
        self.assertNotIn("AAAUSDT", bot.state["orders"])  # cooldown: no new order yet
        for _ in range(23):
            fake.advance(); bot.cycle()
            self.assertNotIn("AAAUSDT", bot.state["orders"])
        fake.advance(); bot.cycle()
        self.assertIn("AAAUSDT", bot.state["orders"])     # 24 h later it is back

    def test_time_stop_after_48h(self):
        fake, bot = make(self.crash_then(flat(87.0, 60)), 26)
        bot.cycle(); fake.advance(); bot.cycle()
        self.assertIn("AAAUSDT", bot.state["positions"])
        for _ in range(47):
            fake.advance(); bot.cycle()
            self.assertIn("AAAUSDT", bot.state["positions"])
        fake.advance(); bot.cycle()
        self.assertNotIn("AAAUSDT", bot.state["positions"])
        with open(os.path.join(bot.root, "panic_trades.csv")) as f:
            row = list(csv.DictReader(f))[-1]
        self.assertEqual(row["reason"], "time limit 48h")
        self.assertLess(float(row["pnl_usdt_after_fees"]), 0)
        self.assertEqual(fake.bal["AAA"]["locked"], 0.0)
        self.assertLess(fake.bal["AAA"]["free"], 0.01)                      # only fee dust left

    def test_partial_fill_cancels_the_rest(self):
        fake, bot = make(self.crash_then(flat(88.6, 10)), 26)
        bot.cycle()
        oid = bot.state["orders"]["AAAUSDT"]["order_id"]
        fake.partial[oid] = 0.4
        fake.advance(); bot.cycle()
        self.assertEqual(fake.orders[oid]["status"], "CANCELED")
        p = bot.state["positions"]["AAAUSDT"]
        self.assertAlmostEqual(p["qty"], float(fake.orders[oid]["executedQty"]) * 0.999, places=9)
        self.assertAlmostEqual(fake.bal["USDT"]["locked"], 0.0, places=6)

    def test_unknown_order_is_dropped(self):
        fake, bot = make(ten_coins(), 26)
        bot.state["orders"]["AAAUSDT"] = {"order_id": 999, "limit": 88.0, "qty": 1.0, "placed_ms": 0}
        bot.cycle()
        self.assertTrue(any("no longer knows order 999" in m for m in bot.logs))


class Money(unittest.TestCase):
    def test_profits_set_aside_unless_compound(self):
        for compound in (False, True):
            fake, bot = make({"AAAUSDT": series(flat(100.0, 30))}, 26, compound=compound)
            bot.state["realized_usdt"] = 50.0
            total, _ = bot.budget()
            self.assertEqual(total, 1050.0 if compound else 1000.0)
            bot.state["realized_usdt"] = -50.0
            self.assertEqual(bot.budget()[0], 950.0)                       # losses always shrink the budget

    def test_panic_touches_only_its_own_orders_and_coins(self):
        fake, bot = make(self.__class__.crash(), 26)
        # another bot (halal-rider) holds 1 AAA with its own stop order, and has a buy order on BBB
        fake.bal["AAA"] = {"free": 1.0, "locked": 0.0}
        foreign_sell = fake.new_order(symbol="AAAUSDT", side="SELL", type="LIMIT", quantity="1", price="200")["orderId"]
        bot.cycle(); fake.advance(); bot.cycle()
        self.assertIn("AAAUSDT", bot.state["positions"])
        foreign_buy = fake.new_order(symbol="BBBUSDT", side="BUY", type="LIMIT", quantity="0.5", price="50")["orderId"]
        bot.panic()
        self.assertEqual(bot.state["positions"], {})
        self.assertEqual(bot.state["orders"], {})
        self.assertEqual(fake.orders[foreign_buy]["status"], "NEW")
        self.assertEqual(fake.orders[foreign_sell]["status"], "NEW")
        self.assertAlmostEqual(fake.bal["AAA"]["locked"], 1.0, places=9)  # other bot's coin untouched

    @staticmethod
    def crash():
        rows = flat(100.0, 26) + [(100.0, 100.2, 87.0, 88.5)] + flat(88.6, 10)
        return {"AAAUSDT": series(rows), "BBBUSDT": series(flat(100.0, 40))}


if __name__ == "__main__":
    unittest.main()
