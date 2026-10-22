"""Reproduce execution/risk failure points of panic_buyer.py against tests/fake_binance.py.
Read-only with respect to the repo: imports bot + fake exchange, edits nothing.
Run: python3 risk_repro.py /home/user/halal-rider
"""
import json
import os
import sys
import tempfile

REPO = sys.argv[1] if len(sys.argv) > 1 else "/home/user/halal-rider"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
import panic_buyer as PB  # noqa: E402
from binance_spot import BinanceError  # noqa: E402
from fake_binance import FakeBinance  # noqa: E402

H = 3600 * 1000
T0 = (1_790_000_000_000 // H) * H
CFG90 = json.load(open(os.path.join(REPO, "panic_config.90wr.example.json")))
CFG10 = json.load(open(os.path.join(REPO, "panic_config.10pct.example.json")))


def series(rows):
    return [{"open_ms": T0 + i * H, "open": o, "high": h, "low": lo, "close": c, "close_ms": T0 + (i + 1) * H - 1}
            for i, (o, h, lo, c) in enumerate(rows)]


def flat(p, n):
    return [(p, p * 1.001, p * 0.999, p)] * n


def make(candles, start=170, usdt=1000.0, capital=1000.0, mode=CFG90):
    fake = FakeBinance(candles, start=start, usdt=usdt)
    cfg = dict(PB.DEFAULT_CFG)
    cfg["symbols"] = list(candles)
    cfg.update(mode)
    bot = PB.PanicBuyer(fake, cfg, capital, root=tempfile.mkdtemp(), out=lambda m: None, clock=fake.clock)
    bot.load_info()
    return fake, bot


def crash(after):  # 170 flat hours at 100 (crash guard needs 168), then a 14%+ dip, then `after`
    return {"AAAUSDT": series(flat(100.0, 170) + [(100.0, 100.2, 85.0, 85.5)] + after)}


def cyc(bot):  # like main(): errors in a cycle are logged and the loop goes on
    try:
        bot.cycle()
    except Exception as err:  # noqa: BLE001
        bot.log(f"ALERT: cycle error: {err}")


def run(fake, bot, hours):
    for _ in range(hours):
        fake.advance()
        cyc(bot)


print("== A. No price stop: all-in entry, coin keeps falling 70% inside the 96h window (90% mode)")
fake, bot = make(crash([(85.5, 85.6, 60, 60)] + flat(25.5, 120)))
cyc(bot); run(fake, bot, 100)
print(f"   account value {fake.value():.2f} of 1000.00; trades={bot.state['trades']} wins={bot.state['wins']} "
      f"realized={bot.state['realized_usdt']:.2f}")

print("== B. balances() fails right after the buy fills -> coins orphaned (no TP, no time stop)")
fake, bot = make(crash(flat(70.0, 120)))
cyc(bot); fake.advance()
orig = fake.balances
fake.balances = lambda: (_ for _ in ()).throw(BinanceError("GET /api/v3/account failed: timed out"))
cyc(bot)
fake.balances = orig
run(fake, bot, 100)
print(f"   positions={list(bot.state['positions'])} orders={list(bot.state['orders'])} "
      f"AAA held={fake.bal.get('AAA', {}).get('free', 0):.4f} trades={bot.state['trades']} value={fake.value():.2f}")

print("== C. POST /order times out but Binance accepted it -> live buy order the bot does not know")
fake, bot = make(crash(flat(70.0, 120)))
real_new = fake.new_order


def lost_reply(**kw):
    real_new(**kw)
    raise BinanceError("POST /api/v3/order failed: timed out")


fake.new_order = lost_reply
cyc(bot)
fake.new_order = real_new
run(fake, bot, 100)
print(f"   bot state orders={bot.state['orders']} positions={list(bot.state['positions'])}; "
      f"exchange AAA held={fake.bal.get('AAA', {}).get('free', 0):.4f}; value={fake.value():.2f}")

print("== D. LUNA-style collapse to below the 5 USDT minimum -> position deleted, loss never recorded")
fake, bot = make(crash([(85.5, 85.6, 1.0, 1.0)] + flat(0.004, 120)))
cyc(bot); run(fake, bot, 100)
print(f"   positions={list(bot.state['positions'])} trades={bot.state['trades']} wins={bot.state['wins']} "
      f"realized={bot.state['realized_usdt']:.2f} budget={bot.budget()} account value={fake.value():.2f}")

print("== E. TP partly filled, then time-stop market sell errors -> next hour sells the other bot's coins")
fake, bot = make(crash([(85.5, 85.6, 85.0, 85.4)] + flat(85.4, 120)), mode=CFG10)
cyc(bot); fake.advance(); cyc(bot)
p = bot.state["positions"]["AAAUSDT"]
own = p["qty"]
fake.orders[p["tp_id"]]["executedQty"] = repr(own * 0.5)   # half of the TP sold
fake.orders[p["tp_id"]]["cummulativeQuoteQty"] = repr(own * 0.5 * p["tp_price"])
fake.orders[p["tp_id"]]["status"] = "PARTIALLY_FILLED"
fake._a("AAA")["locked"] -= own * 0.5
fake._a("USDT")["free"] += own * 0.5 * p["tp_price"] * 0.999
fake._a("AAA")["free"] += 10.0                              # halal-rider's own AAA coins
p["opened_ms"] -= 48 * H                                    # time limit reached
orig_prices = fake.prices
fake.prices = lambda s: (_ for _ in ()).throw(BinanceError("GET /api/v3/ticker/price HTTP 503"))
cyc(bot)
fake.prices = orig_prices
print(f"   after failed close: bot thinks it holds {bot.state['positions']['AAAUSDT']['qty']:.4f}, really owns "
      f"{own * 0.5:.4f}")
fake.advance(); cyc(bot)
print(f"   after next hour: AAA left in account {fake.bal['AAA']['free'] + fake.bal['AAA']['locked']:.4f} "
      f"(other bot alone had 10.0000)")

print("== F. Symbol halted (FTT-style) while held -> every hourly exit attempt rejected, no alert escalation")
fake, bot = make(crash(flat(60.0, 200)), mode=CFG10)
cyc(bot); fake.advance(); cyc(bot)
fake.status["AAAUSDT"] = "BREAK"
logs = []
bot.out = logs.append
run(fake, bot, 120)
print(f"   still holding: {list(bot.state['positions'])}; ALERT lines logged: "
      f"{sum('ALERT' in m for m in logs)}; exit attempts: {sum(1 for c in fake.calls if c[0]=='new' and c[2]=='SELL')}")

print("== G. No portfolio kill-switch: 3 consecutive all-in losses, bot keeps re-entering at 100%")
rows = flat(100.0, 170)
for _ in range(3):
    rows += [(rows[-1][3], rows[-1][3], rows[-1][3] * 0.85, rows[-1][3] * 0.855)] + flat(rows[-1][3] * 0.6, 100)
    rows += flat(rows[-1][3], 170)
fake, bot = make({"AAAUSDT": series(rows)})
cyc(bot); run(fake, bot, len(rows) - 172)
print(f"   trades={bot.state['trades']} wins={bot.state['wins']} realized={bot.state['realized_usdt']:.2f} "
      f"account value={fake.value():.2f}")
