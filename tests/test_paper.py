"""Paper mode: the real bot logic on a pretend account (market data from a stub instead of Binance)."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import panic_buyer as PB  # noqa: E402
from fake_binance import FakeBinance  # noqa: E402
from paper_exchange import PaperExchange  # noqa: E402
from test_panic import flat, series, ten_coins  # noqa: E402


class Market:
    """Public market data only, from candles; clock() is the bot's time (10 s into the current candle)."""

    def __init__(self, candles, start):
        self.f = FakeBinance(candles, start=start)

    def sync_time(self):
        return 0

    def exchange_info(self, symbols):
        return self.f.exchange_info(symbols)

    def klines(self, symbol, interval="1h", limit=30):
        return self.f.klines(symbol, interval, limit)

    def prices(self, symbols):
        return self.f.prices(symbols)

    def book_ticker(self, symbol):
        return self.f.book_ticker(symbol)

    def clock(self):
        return self.f.clock()

    def advance(self):
        self.f.i += 1

    def signed(self, *a, **k):
        raise AssertionError("paper mode must never call a signed (account) endpoint")

    balances = get_order = cancel_order = new_order = signed


def setup(candles, start, root=None, **cfg_over):
    m = Market(candles, start)
    root = root or tempfile.mkdtemp()
    px = PaperExchange(root, 1000.0, market=m, clock=m.clock)
    cfg = dict(PB.DEFAULT_CFG)
    cfg["symbols"] = list(candles)
    cfg.update(cfg_over)
    e = PB.Ensemble(px, cfg, 1000.0, compound=True, root=root, out=lambda s: None, clock=m.clock)
    e.load_info()
    return m, px, e


class Paper(unittest.TestCase):
    def test_orders_are_placed_on_the_pretend_account(self):
        m, px, e = setup(ten_coins(), 26)
        e.cycle()
        self.assertEqual(len(e.bots[0].state["orders"]), 8)
        self.assertAlmostEqual(px.a["bal"]["USDT"]["free"] + px.a["bal"]["USDT"]["locked"], 1000.0, places=6)
        self.assertGreater(px.a["bal"]["USDT"]["locked"], 900.0)

    def test_fill_take_profit_and_profit(self):
        rows = flat(100.0, 26) + [(100.0, 100.2, 87.0, 88.5), (88.5, 90.0, 88.4, 89.9)] + flat(95.0, 30)
        m, px, e = setup({"AAAUSDT": series(rows)}, 26, n_orders=1, max_order_pct=100.0)
        e.cycle()                                  # buy order at 88 waits
        m.advance(); e.cycle()                     # crash candle closed: filled, take-profit placed
        b = e.bots[0]
        self.assertIn("AAAUSDT", b.state["positions"])
        m.advance(); e.cycle()                     # next candle reached +1.5%: sold
        self.assertEqual(b.state["trades"], 1)
        self.assertEqual(b.state["wins"], 1)
        self.assertGreater(px.value(), 1000.0)

    def test_pretend_account_survives_a_restart(self):
        root = tempfile.mkdtemp()
        m, px, e = setup(ten_coins(), 26, root=root)
        e.cycle()
        px2 = PaperExchange(root, 99999.0, market=m, clock=m.clock)
        self.assertAlmostEqual(px2.a["start_usdt"], 1000.0)
        self.assertEqual(len([o for o in px2.a["orders"].values() if o["status"] == "NEW"]), 8)

    def test_paper_switch_uses_its_own_folder_and_no_key(self):
        env = dict(os.environ)
        root = tempfile.mkdtemp()
        os.environ.update({"PAPER": "yes", "PAPER_DATA": root, "PANIC_DATA": "/nonexistent/live", "PAPER_USDT": "650",
                           "BINANCE_API_KEY": "", "BINANCE_API_SECRET": ""})
        try:
            PB.main(["status"])
        finally:
            os.environ.clear()
            os.environ.update(env)
        self.assertTrue(os.path.exists(os.path.join(root, "paper_account.json")))


if __name__ == "__main__":
    unittest.main()
