"""Replay the REAL panic_buyer.py (unmodified, imported from the repo root) hour by hour through tests/fake_binance.py
with the hr_pick.py pick's settings, and compare with engine.run on exactly the same coins and candles.
Same method as replay_bot_full.py; only the bot's config is overridden (nothing in the repo is changed).
Usage: hr_verify_replay.py START END OUT_DIR depth_pct n_orders max_order_pct take_profit_pct max_hours cooldown_hours
"""
import os
import sys
import time

import numpy as np
import pandas as pd

REPO = "/home/user/halal-rider"
HERE = os.path.dirname(os.path.abspath(__file__))
start, end, out = pd.Timestamp(sys.argv[1]), pd.Timestamp(sys.argv[2]), sys.argv[3]
depth, nord, fmax_pct, tp_pct, hm, cd = (float(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]),
                                         int(sys.argv[8]), int(sys.argv[9]))
os.makedirs(out, exist_ok=True)
sys.path[:0] = [REPO, os.path.join(REPO, "tests"), HERE]
import panic_buyer as PB  # noqa: E402
import fake_binance as FB  # noqa: E402
import engine as E  # noqa: E402


def klines(self, symbol, interval="1h", limit=30):
    rows = self.k[symbol]
    return [dict(r) for r in rows[max(0, self.i + 1 - limit): self.i + 1]]


FB.FakeBinance.klines = klines

idx = pd.date_range(start - pd.Timedelta(days=2), end, freq="1h", inclusive="left")
D = {}
for s in PB.DEFAULT_CFG["symbols"]:
    p = os.path.join(E.DATA, "1h", f"{s}.csv")
    d = pd.read_csv(p, index_col=0, parse_dates=True).reindex(idx)
    if d.c.notna().mean() > 0.995:
        d["c"] = d.c.ffill().bfill()
        for f in ("o", "h", "l"):
            d[f] = d[f].fillna(d["c"])
        D[s] = d
EPOCH = pd.Timestamp("1970-01-01")
ms = ((idx - EPOCH) // pd.Timedelta(milliseconds=1)).to_numpy().astype(np.int64)
candles = {s: [{"open_ms": int(ms[i]), "open": float(o), "high": float(h), "low": float(l), "close": float(c),
                "close_ms": int(ms[i] + 3600_000 - 1)} for i, (o, h, l, c) in enumerate(zip(d.o, d.h, d.l, d.c))]
           for s, d in D.items()}
i0 = int(np.searchsorted(idx, start))
fake = FB.FakeBinance(candles, start=i0, usdt=1000.0, tick="0.0000001", step="0.0000001")
cfg = dict(PB.DEFAULT_CFG)
cfg.update(symbols=list(D), depth_pct=depth, n_orders=nord, max_order_pct=fmax_pct, take_profit_pct=tp_pct,
           max_hours=hm, cooldown_hours=cd)
bot = PB.PanicBuyer(fake, cfg, 1000.0, compound=True, root=out, out=lambda m: None, clock=fake.clock)
bot.load_info()
t_0 = time.time()
eqs = []
while fake.i < len(idx) - 1:
    bot.cycle(); fake.advance()
    eqs.append((idx[fake.i], fake.value() / 1000.0))
bdf = pd.DataFrame(eqs, columns=["t", "eq"])
bdf.to_csv(os.path.join(out, "bot_eq.csv"), index=False)
be = bdf.set_index("t")["eq"]
syms, pidx, P = E.panel(D)
keep = pidx >= start
P = {k: v[keep] for k, v in P.items()}
P["ref"] = pd.DataFrame({s: D[s].c for s in syms}).rolling(24).max().shift(1)[keep]
kw = dict(depth=depth / 100, N=nord, fmax=fmax_pct / 100, tp=tp_pct / 100, hm=hm, cd=cd)


def seg(e, a, b):
    s = e[(e.index >= a) & (e.index < b)]
    prev = e[e.index < a]
    s0 = float(prev.iloc[-1]) if len(prev) else float(s.iloc[0])
    mult = float(s.iloc[-1]) / s0
    months = (s.index[-1] - a).total_seconds() / (365.25 * 86400) * 12
    dd = float((s / s.cummax() - 1).min())
    return mult, mult ** (1 / months) - 1, dd


cuts = [(pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y+1}-01-01")) for y in range(start.year, end.year + 1)]
spans = [(max(start, a), min(end, b)) for a, b in cuts if a < end]
if start < E.DESIGN_END < end:
    spans += [(start, E.DESIGN_END), (E.DESIGN_END, end)]
if start < pd.Timestamp("2025-01-01") < end:
    spans += [(pd.Timestamp("2025-01-01"), end)]
curves = {"REAL BOT (panic_buyer.py)": be}
for lab, extra in (("engine, backtest timing", {}), ("engine, bot timing (mtp_now, cd+1)", dict(mtp_now=True, cd=cd + 1))):
    e, lg = E.run(P, pidx[keep], **{**kw, **extra})
    e.to_csv(os.path.join(out, f"engine_eq_{'A' if not extra else 'B'}.csv"))
    curves[lab] = e
print(f"{len(D)} coins: {', '.join(s[:-4] for s in D)}; settings {kw}; bot trades {bot.state['trades']}; "
      f"replay took {time.time() - t_0:.0f}s")
for a, b in spans:
    line = f"  {a:%Y-%m-%d}..{b:%Y-%m-%d}"
    for lab, e in curves.items():
        m, r, dd = seg(e, a, b)
        line += f" | {lab}: x{m:6.2f} {r*100:+5.2f}%/mo DD {dd*100:4.0f}%"
    print(line)
