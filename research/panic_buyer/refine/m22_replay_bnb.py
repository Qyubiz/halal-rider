"""m22: the real bot code replayed hour by hour through the fake exchange in 90% mode, 2024-01-01 to 2026-10-22.

Two things are checked against the engine (same 18 coins, bot timing):
  1. One coin at a time. Before `max_positions` the bot could place a second, small buy with the ~2% of cash its
     main order leaves when the account holds exactly its capital (as the paper bots do). The backtest never does.
  2. Fees paid with BNB (0.075% a side, never taken from the coin):
       BNB_FEES=yes   the bot reads each buy's real fee and sells exactly what it bought
       no switch      the bot assumes a coin fee and leaves 0.1% of every buy unsold
Cases with "shared account" hold 2,000 USDT for a bot allowed 1,000, so the bot's own budget sets the order size
(100% of its money) in every fee set-up and only the fee differs.
Usage (from research/panic_buyer/refine): PB_DATA=... python3 m22_replay_bnb.py [case ...]"""
import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path[:0] = [HERE, REPO, os.path.join(REPO, "tests")]
import engine as E  # noqa: E402
import fake_binance as FB  # noqa: E402
import panic_buyer as PB  # noqa: E402

START, END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-10-22")
MODE = json.load(open(os.path.join(REPO, "panic_config.90wr.example.json")))


def klines(self, symbol, interval="1h", limit=30):       # same result as FakeBinance.klines, without copying history
    rows = self.k[symbol]
    return [dict(r) for r in rows[max(0, self.i + 1 - limit): self.i + 1]]


FB.FakeBinance.klines = klines


def candles():
    idx = pd.date_range(START - pd.Timedelta(days=8), END, freq="1h", inclusive="left")
    D = {}
    for s in PB.DEFAULT_CFG["symbols"]:
        d = E.read_hourly(s)
        if d is None:
            continue
        d = d.reindex(idx)
        if d.c.notna().mean() > 0.995:
            d["c"] = d.c.ffill().bfill()
            for f in ("o", "h", "l"):
                d[f] = d[f].fillna(d["c"])
            D[s] = d
    ms = ((idx - pd.Timestamp("1970-01-01")) // pd.Timedelta(milliseconds=1)).to_numpy().astype(np.int64)
    c = {s: [{"open_ms": int(ms[i]), "open": float(o), "high": float(h), "low": float(l_), "close": float(cl),
              "close_ms": int(ms[i] + 3600_000 - 1)} for i, (o, h, l_, cl) in enumerate(zip(d.o, d.h, d.l, d.c))]
         for s, d in D.items()}
    return idx, c


def replay(idx, c, exchange_bnb=False, bot_switch=False, cap=True, usdt=1000.0):
    fake = FB.FakeBinance(c, start=int(np.searchsorted(idx, START)), usdt=usdt, tick="0.0000001", step="0.0000001")
    fake.bnb = exchange_bnb
    cfg = dict(PB.DEFAULT_CFG)
    cfg.update(MODE)
    cfg["symbols"] = list(c)
    if not cap:
        cfg["max_positions"] = 0
    if bot_switch:
        cfg.update(bnb_fees=True, fee_pct=0.075)
    bot = PB.PanicBuyer(fake, cfg, 1000.0, compound=True, root=tempfile.mkdtemp(), out=lambda m: None, clock=fake.clock)
    bot.load_info()
    while fake.i < len(idx) - 1:
        bot.cycle()
        fake.advance()
    months = (END - START).days / 30.4375
    v = fake.value() - usdt + 1000.0                      # the bot's 1,000 USDT, at market, after all fees
    held = {s for s in bot.state["positions"]}
    dust = sum((b["free"] + b["locked"]) * fake.price(a + "USDT") for a, b in fake.bal.items()
               if a != "USDT" and a + "USDT" in c and a + "USDT" not in held)
    st = bot.state
    return dict(value=v, mo=((v / 1000.0) ** (1 / months) - 1) * 100, trades=st["trades"],
                wr=st["wins"] / max(st["trades"], 1) * 100, bot_profit=st["realized_usdt"], dust=dust,
                fees_bnb=fake.bnb_paid)


CASES = {
    "old":      ("no position cap (before), account = capital", dict(cap=False)),
    "cap":      ("one coin at a time, account = capital", dict()),
    "normal":   ("shared account, normal fees", dict(usdt=2000.0)),
    "bnb":      ("shared account, BNB fees, BNB_FEES=yes", dict(usdt=2000.0, exchange_bnb=True, bot_switch=True)),
    "nofix":    ("shared account, BNB fees, no switch", dict(usdt=2000.0, exchange_bnb=True)),
}

if __name__ == "__main__":
    which = sys.argv[1:] or list(CASES)
    idx, c = candles()
    print(f"{len(c)} coins, {START:%Y-%m-%d} to {END:%Y-%m-%d}, bot capital 1,000 USDT, profits reinvested\n")
    print("| case | bot money at the end | %/month | trades | wins | bot's own profit number | coins left unsold "
          "(USDT) | fees paid with BNB (USDT) |\n|---|---|---|---|---|---|---|---|")
    for w in which:
        name, kw = CASES[w]
        r = replay(idx, c, **kw)
        print(f"| {name} | {r['value']:.0f} | {r['mo']:+.2f} | {r['trades']} | {r['wr']:.1f}% | {r['bot_profit']:+.0f} | "
              f"{r['dust']:.2f} | {r['fees_bnb']:.0f} |", flush=True)
