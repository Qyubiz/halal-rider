"""Tests for the combined (sleeves) bot, the hard risk limits and the audit fixes, against the fake exchange."""
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import panic_buyer as PB  # noqa: E402
from binance_spot import BinanceError  # noqa: E402
from fake_binance import FakeBinance  # noqa: E402
from test_panic import flat, series, ten_coins  # noqa: E402

H = 3600 * 1000
MODE90 = {"name": "wr90", "weight": 0.3, "n_orders": 1, "max_order_pct": 100.0, "depth_pct": 14.0,
          "take_profit_pct": 2.5, "max_hours": 96, "cooldown_hours": 3, "crash_guard_pct": 35.0}
MODE10 = {"name": "r10", "weight": 0.7, "n_orders": 1, "max_order_pct": 100.0, "depth_pct": 14.0,
          "take_profit_pct": 5.0, "max_hours": 48, "cooldown_hours": 24, "crash_guard_pct": 35.0}


def ens(candles, start, risk=None, sleeves=None, usdt=1000.0, capital=1000.0, **over):
    fake = FakeBinance(candles, start=start, usdt=usdt)
    cfg = dict(PB.DEFAULT_CFG)
    cfg["symbols"] = list(candles)
    cfg.update(over)
    if sleeves is not None:
        cfg["sleeves"] = sleeves
    cfg["risk"] = risk or {}
    logs = []
    e = PB.Ensemble(fake, cfg, capital, compound=True, root=tempfile.mkdtemp(), out=logs.append, clock=fake.clock)
    e.load_info()
    e.logs = logs
    return fake, e


def step(fake, e, n=1):
    for _ in range(n):
        fake.advance()
        e.cycle()


def crash_rows(after):
    return flat(100.0, 30) + [(100.0, 100.2, 85.0, 86.5)] + after


class Sleeves(unittest.TestCase):
    def test_capital_split_and_separate_state(self):
        fake, e = ens(ten_coins(), 26, sleeves=[MODE90, MODE10])
        self.assertEqual([b.capital for b in e.bots], [300.0, 700.0])
        e.cycle()
        roots = {b.root for b in e.bots}
        self.assertEqual(len(roots), 2)
        for b in e.bots:                         # each sleeve: one order with all its own money at 14% under
            self.assertEqual(list(b.state["orders"]), ["JJJUSDT"])
            o = b.state["orders"]["JJJUSDT"]
            self.assertAlmostEqual(o["limit"], 86.0, places=3)
            self.assertLessEqual(o["qty"] * o["limit"], b.capital + 1e-6)
            self.assertGreater(o["qty"] * o["limit"], 0.95 * b.capital)

    def test_plain_config_keeps_the_old_state_files(self):
        fake, e = ens(ten_coins(), 26)
        self.assertEqual(len(e.bots), 1)
        e.cycle()
        self.assertTrue(os.path.exists(os.path.join(e.bots[0].root, "panic_state.json")))
        self.assertEqual(len(e.bots[0].state["orders"]), 8)          # same as the plain bot


class KillSwitch(unittest.TestCase):
    def test_daily_drawdown_flattens_and_locks(self):
        rows = crash_rows([(86.5, 86.6, 60.0, 60.0)] + flat(60.0, 40))
        fake, e = ens({"AAAUSDT": series(rows)}, 30, risk={"daily_dd_pct": 15}, n_orders=1, max_order_pct=100.0,
                      depth_pct=14.0)
        e.cycle()                                   # order at 86
        step(fake, e)                               # fills in the crash candle
        self.assertIn("AAAUSDT", e.bots[0].state["positions"])
        step(fake, e)                               # coin at 60: money -30% today
        self.assertEqual(e.bots[0].state["positions"], {})
        self.assertTrue(any("KILL-SWITCH" in m for m in e.logs))
        ok, why = e.guard.entries_allowed(fake.clock())
        self.assertFalse(ok)
        step(fake, e)
        self.assertEqual(e.bots[0].state["orders"], {})              # locked: no new buys

    def test_monthly_drawdown_lock_survives_restart(self):
        rows = crash_rows([(86.5, 86.6, 70.0, 70.0)] + flat(70.0, 40))
        fake, e = ens({"AAAUSDT": series(rows)}, 30, risk={"monthly_dd_pct": 10}, n_orders=1, max_order_pct=100.0,
                      depth_pct=14.0)
        e.cycle(); step(fake, e, 2)
        self.assertTrue(e.guard.s["locked_until"] > fake.clock() + 100 * H)   # 168h lock
        cfg = dict(e.bots[0].cfg); cfg["risk"] = {"monthly_dd_pct": 10}
        e2 = PB.Ensemble(fake, cfg, 1000.0, compound=True, root=e.bots[0].root, out=lambda m: None, clock=fake.clock)
        self.assertFalse(e2.guard.entries_allowed(fake.clock())[0])


class TradeLimits(unittest.TestCase):
    def test_per_trade_risk_cap(self):
        fake, e = ens(ten_coins(), 26, risk={"max_trade_risk_pct": 10, "stop_loss_pct": 25}, n_orders=1,
                      max_order_pct=100.0)
        e.cycle()
        o = next(iter(e.bots[0].state["orders"].values()))
        self.assertLessEqual(o["qty"] * o["limit"], 400.0 + 1e-6)    # 10% of 1000 / 25% stop

    def test_stop_loss_at_hourly_close(self):
        rows = crash_rows([(86.5, 86.6, 70.0, 70.0)] + flat(70.0, 10))
        fake, e = ens({"AAAUSDT": series(rows)}, 30, risk={"stop_loss_pct": 15}, n_orders=1, max_order_pct=100.0,
                      depth_pct=14.0)
        e.cycle(); step(fake, e)
        self.assertIn("AAAUSDT", e.bots[0].state["positions"])
        step(fake, e, 2)                            # an hourly close at 70 is >15% under 86
        self.assertEqual(e.bots[0].state["positions"], {})
        self.assertTrue(any("stop-loss" in m for m in e.logs))

    def test_loss_streak_pauses_buying(self):
        fake, e = ens(ten_coins(), 26, risk={"loss_streak": 2, "loss_pause_hours": 72})
        e.guard.on_trade_closed(-1.0, fake.clock())
        self.assertTrue(e.guard.entries_allowed(fake.clock())[0])
        e.guard.on_trade_closed(-1.0, fake.clock())
        self.assertFalse(e.guard.entries_allowed(fake.clock())[0])
        self.assertTrue(e.guard.entries_allowed(fake.clock() + 73 * H)[0])

    def test_wide_spread_blocks_the_buy(self):
        fake, e = ens(ten_coins(), 26, risk={"max_spread_pct": 0.5}, n_orders=1)
        fake.spread["JJJUSDT"] = 0.02
        e.cycle()
        self.assertNotIn("JJJUSDT", e.bots[0].state["orders"])

    def test_slow_api_blocks_new_buys(self):
        fake, e = ens(ten_coins(), 26, risk={"max_latency_ms": 50})
        slow = fake.klines

        def klines(*a, **k):
            import time
            time.sleep(0.06)
            return slow(*a, **k)
        fake.klines = klines
        e.cycle()
        self.assertEqual(e.bots[0].state["orders"], {})

    def test_atr_spike_skips_the_coin(self):
        rows = flat(100.0, 800) + [(100.0, 100.0, 70.0, 99.0)] + flat(99.0, 3)
        fake, e = ens({"AAAUSDT": series(rows)}, 801, risk={"atr_spike_mult": 5}, n_orders=1, depth_pct=1.0)
        e.cycle()
        self.assertEqual(e.bots[0].state["orders"], {})
        fake2, e2 = ens({"AAAUSDT": series(rows)}, 801, n_orders=1, depth_pct=1.0)
        e2.cycle()
        st = e2.bots[0].state
        self.assertTrue("AAAUSDT" in st["orders"] or "AAAUSDT" in st["positions"])


class AuditFixes(unittest.TestCase):
    def test_fill_is_kept_when_the_balance_call_fails(self):
        fake, e = ens({"AAAUSDT": series(crash_rows(flat(88.0, 5)))}, 30, n_orders=1, max_order_pct=100.0)
        e.cycle(); fake.advance()
        real = fake.balances

        def boom():
            raise BinanceError("timeout")
        fake.balances = boom
        e.cycle()
        fake.balances = real
        self.assertIn("AAAUSDT", e.bots[0].state["positions"])

    def test_partial_take_profit_then_failed_sell_never_sells_twice(self):
        fake, e = ens({"AAAUSDT": series(crash_rows(flat(87.0, 60)))}, 30, n_orders=1, max_order_pct=100.0)
        b = e.bots[0]
        e.cycle(); step(fake, e)
        p = b.state["positions"]["AAAUSDT"]
        tp = fake.orders[p["tp_id"]]
        fake._fill(tp, float(tp["origQty"]) / 2, float(tp["price"]))
        qty_before = p["qty"]
        real = fake.new_order

        def fail(**kw):
            raise BinanceError("HTTP 503", None)
        fake.new_order = fail
        try:
            b.close_position("AAAUSDT", "test")
        except BinanceError:
            pass
        fake.new_order = real
        sold = float(tp["executedQty"])
        self.assertAlmostEqual(b.state["positions"]["AAAUSDT"]["qty"], qty_before - sold, places=9)
        self.assertEqual(b.state["trades"], 1)                       # the part sale is booked once

    def test_dust_after_a_collapse_is_booked_as_a_loss(self):
        rows = crash_rows([(86.0, 86.0, 0.3, 0.3)] + flat(0.3, 60))
        fake, e = ens({"AAAUSDT": series(rows)}, 30, n_orders=1, max_order_pct=2.0, max_hours=2)
        e.cycle(); step(fake, e, 4)
        b = e.bots[0]
        self.assertEqual(b.state["positions"], {})
        self.assertLess(b.state["realized_usdt"], 0)
        self.assertEqual(b.state["trades"], 1)

    def test_halted_coin_alerts_once(self):
        fake, e = ens({"AAAUSDT": series(crash_rows(flat(87.0, 60)))}, 30, n_orders=1, max_order_pct=100.0,
                      max_hours=2)
        alerts = []
        for b in e.bots:
            b.notify = alerts.append
        e.cycle(); step(fake, e)
        for b in e.bots:
            b.info["AAAUSDT"]["status"] = "BREAK"
        step(fake, e, 5)
        self.assertIn("AAAUSDT", e.bots[0].state["positions"])
        self.assertEqual(sum("stopped trading" in a for a in alerts), 1)

    def test_buy_that_timed_out_but_went_through_is_tracked(self):
        fake, e = ens(ten_coins(), 26, n_orders=1)
        real = fake.new_order

        def lost_reply(**kw):
            real(**kw)
            raise BinanceError("POST /api/v3/order failed: timed out")
        fake.new_order = lost_reply
        e.cycle()
        fake.new_order = real
        self.assertIn("JJJUSDT", e.bots[0].state["orders"])


if __name__ == "__main__":
    unittest.main()
