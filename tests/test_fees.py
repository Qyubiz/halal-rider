"""Paying Binance fees with BNB (BNB_FEES=yes): the bot reads each buy's real fee from Binance instead of assuming
it was taken in the coin, so nothing is left unsold and it never sells more than it bought."""
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import panic_buyer as PB  # noqa: E402
from paper_exchange import PaperExchange  # noqa: E402
from test_panic import flat, make, series  # noqa: E402
from test_paper import Market  # noqa: E402

ONE = dict(n_orders=1, max_order_pct=100.0)
STEP = 0.0001


def crash():
    """Flat at 100, one candle down to 87 (fills a waiting buy at 88), back to 95 (take-profit at +1.5%)."""
    return {"AAAUSDT": series(flat(100.0, 26) + [(100.0, 100.2, 87.0, 88.5), (88.5, 90.0, 88.4, 89.9)] + flat(95.0, 30))}


def under():
    """Already at 85 at the hourly check: under the 88 trigger, so the bot buys at market."""
    return {"AAAUSDT": series(flat(100.0, 26) + [(85.0, 86.0, 84.0, 85.5)] * 5)}


def bought(fake, bot):
    """Run until the waiting buy has filled and the take-profit is placed; returns (position, executed quantity)."""
    bot.cycle()
    fake.advance()
    bot.cycle()
    buy = next(o for o in fake.orders.values() if o["side"] == "BUY")
    return bot.state["positions"]["AAAUSDT"], float(buy["executedQty"])


class BnbFees(unittest.TestCase):
    def test_limit_buy_keeps_every_coin_when_the_fee_is_paid_with_bnb(self):
        fake, bot = make(crash(), 26, bnb_fees=True, fee_pct=0.075, **ONE)
        fake.bnb = True
        p, executed = bought(fake, bot)
        self.assertAlmostEqual(p["qty"], executed, places=9)                 # nothing held back for a coin fee
        self.assertIn(("trades", "AAAUSDT", 1), fake.calls)                  # read from Binance's trade list
        self.assertLess(fake.bal["AAA"]["free"], STEP)                       # all of it is in the take-profit sell
        self.assertAlmostEqual(fake.bal["AAA"]["locked"], executed, delta=STEP)

    def test_market_buy_reads_the_fee_from_the_order_reply(self):
        fake, bot = make(under(), 26, bnb_fees=True, fee_pct=0.075, **ONE)
        fake.bnb = True
        bot.cycle()
        p = bot.state["positions"]["AAAUSDT"]
        executed = float(fake.orders[1]["executedQty"])
        self.assertAlmostEqual(p["qty"], executed, places=9)
        self.assertNotIn(("trades", "AAAUSDT", 1), fake.calls)               # the reply already had the fills
        self.assertLess(fake.bal["AAA"]["free"], STEP)

    def test_bnb_ran_out_the_fee_was_taken_in_the_coin(self):
        fake, bot = make(crash(), 26, bnb_fees=True, fee_pct=0.075, **ONE)   # the exchange takes 0.1% in the coin
        p, executed = bought(fake, bot)
        self.assertAlmostEqual(p["qty"], executed * 0.999, places=9)
        self.assertLess(fake.bal["AAA"]["free"], STEP)                       # sells what it really holds, no more
        self.assertIsNotNone(p["tp_id"])

    def test_unreadable_trade_list_never_oversells(self):
        fake, bot = make(crash(), 26, bnb_fees=True, fee_pct=0.075, **ONE)
        fake.bnb = True
        fake.fail_trades = True
        p, executed = bought(fake, bot)
        self.assertAlmostEqual(p["qty"], executed * 0.999, places=9)         # as if the normal fee was taken
        self.assertIsNotNone(p["tp_id"])
        self.assertTrue(any("could not read the fee" in line for line in bot.logs))

    def test_default_is_unchanged(self):
        fake, bot = make(crash(), 26, **ONE)
        p, executed = bought(fake, bot)
        self.assertAlmostEqual(p["qty"], executed * 0.999, places=9)
        self.assertFalse([c for c in fake.calls if c[0] == "trades"])        # no extra request without the switch

    def test_profit_counts_the_discounted_fee(self):
        fake, bot = make(crash(), 26, bnb_fees=True, fee_pct=0.075, **ONE)
        fake.bnb = True
        p, executed = bought(fake, bot)
        entry, tp = p["entry"], p["tp_price"]
        sold = fake.bal["AAA"]["locked"]
        fake.advance()
        bot.cycle()
        self.assertEqual(bot.state["trades"], 1)
        self.assertAlmostEqual(bot.state["realized_usdt"], sold * (tp * (1 - 0.00075) - entry * (1 + 0.00075)), places=6)

    def test_switch_from_the_environment(self):
        env = dict(os.environ)
        tmp = tempfile.mkdtemp()
        try:
            os.environ["PANIC_CONFIG"] = os.path.join(tmp, "none.json")
            os.environ.pop("BNB_FEES", None)
            cfg = PB.load_cfg()
            self.assertFalse(cfg["bnb_fees"])
            self.assertEqual(cfg["fee_pct"], 0.1)
            os.environ["BNB_FEES"] = "yes"
            cfg = PB.load_cfg()
            self.assertTrue(cfg["bnb_fees"])
            self.assertEqual(cfg["fee_pct"], 0.075)
            path = os.path.join(tmp, "mine.json")
            with open(path, "w") as f:
                json.dump({"fee_pct": 0.06}, f)
            os.environ["PANIC_CONFIG"] = path
            self.assertEqual(PB.load_cfg()["fee_pct"], 0.06)                 # a fee set in the settings file wins
        finally:
            os.environ.clear()
            os.environ.update(env)

    def test_sleeves_inherit_the_switch(self):
        fake, _ = make(crash(), 26)
        cfg = dict(PB.DEFAULT_CFG, symbols=["AAAUSDT"], bnb_fees=True, fee_pct=0.075,
                   sleeves=[dict(name="a", weight=0.5, **ONE), dict(name="b", weight=0.5, **ONE)])
        ens = PB.Ensemble(fake, cfg, 1000.0, root=tempfile.mkdtemp(), out=lambda s: None, clock=fake.clock)
        self.assertTrue(all(b.cfg["bnb_fees"] and b.cfg["fee_pct"] == 0.075 for b in ens.bots))


class PaperBnbFees(unittest.TestCase):
    def test_paper_account_with_bnb_fees(self):
        m = Market(crash(), 26)
        root = tempfile.mkdtemp()
        px = PaperExchange(root, 1000.0, market=m, clock=m.clock, bnb_fees=True)
        cfg = dict(PB.DEFAULT_CFG, symbols=["AAAUSDT"], bnb_fees=True, fee_pct=0.075, **ONE)
        e = PB.Ensemble(px, cfg, 1000.0, compound=True, root=root, out=lambda s: None, clock=m.clock)
        e.load_info()
        e.cycle()
        m.advance()
        e.cycle()                                                            # filled, take-profit placed
        b = e.bots[0]
        buy = next(o for o in px.a["orders"].values() if o["side"] == "BUY")
        self.assertAlmostEqual(b.state["positions"]["AAAUSDT"]["qty"], float(buy["executedQty"]), places=9)
        self.assertEqual(px.my_trades("AAAUSDT", buy["orderId"])[0]["commissionAsset"], "BNB")
        self.assertLess(px.a["bal"]["AAA"]["free"], 1e-6)
        m.advance()
        e.cycle()                                                            # sold at +1.5%
        self.assertEqual(b.state["wins"], 1)
        self.assertGreater(px.value(), 1000.0)
        # the pretend account and the bot's own profit number agree (both count 0.075% a side)
        self.assertAlmostEqual(px.value() - 1000.0, b.state["realized_usdt"], delta=0.05)

    def test_paper_account_without_bnb_reports_a_coin_fee(self):
        m = Market(crash(), 26)
        root = tempfile.mkdtemp()
        px = PaperExchange(root, 1000.0, market=m, clock=m.clock)
        cfg = dict(PB.DEFAULT_CFG, symbols=["AAAUSDT"], bnb_fees=True, fee_pct=0.075, **ONE)
        e = PB.Ensemble(px, cfg, 1000.0, compound=True, root=root, out=lambda s: None, clock=m.clock)
        e.load_info()
        e.cycle()
        m.advance()
        e.cycle()
        buy = next(o for o in px.a["orders"].values() if o["side"] == "BUY")
        self.assertAlmostEqual(e.bots[0].state["positions"]["AAAUSDT"]["qty"], float(buy["executedQty"]) * 0.999,
                               places=9)


if __name__ == "__main__":
    unittest.main()
