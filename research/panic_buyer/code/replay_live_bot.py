"""Drive the LIVE bot hour by hour through the fake exchange on real Binance history and compare with the backtest.
Usage: replay_history.py DATA_DIR EXTRA_DIR START END"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, "/home/claude/bt/lab2")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import lab  # noqa: E402  (backtest engine)
import panic_buyer as PB  # noqa: E402
from fake_binance import FakeBinance  # noqa: E402

data_dir, extra_dir, start, end = sys.argv[1], sys.argv[2], pd.Timestamp(sys.argv[3]), pd.Timestamp(sys.argv[4])
EPOCH = pd.Timestamp("1970-01-01")
idx = pd.date_range(start - pd.Timedelta(days=2), end, freq="1h", inclusive="left")
D = {}
for s in PB.DEFAULT_CFG["symbols"]:
    for d_ in (data_dir, extra_dir):
        p = os.path.join(d_, f"{s}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0, parse_dates=True).reindex(idx)
            if d.c.notna().mean() > 0.995:
                D[s] = d
            break
for s, d in D.items():          # rare exchange gaps: a flat candle at the last close
    d["c"] = d.c.ffill().bfill()
    for f in ("o", "h", "l"):
        d[f] = d[f].fillna(d["c"])
ms = ((idx - EPOCH) // pd.Timedelta(milliseconds=1)).to_numpy().astype(np.int64)
candles = {s: [{"open_ms": int(ms[i]), "open": float(d.o.iloc[i]), "high": float(d.h.iloc[i]), "low": float(d.l.iloc[i]),
                "close": float(d.c.iloc[i]), "close_ms": int(ms[i] + 3600_000 - 1)} for i in range(len(idx))] for s, d in D.items()}
i0 = int(np.searchsorted(idx, start))
fake = FakeBinance(candles, start=i0, usdt=1000.0, tick="0.0000001", step="0.0000001")
cfg = dict(PB.DEFAULT_CFG); cfg["symbols"] = list(D)
bot = PB.PanicBuyer(fake, cfg, 1000.0, compound=True, root=tempfile.mkdtemp(), out=lambda m: None, clock=fake.clock)
bot.load_info()
while fake.i < len(idx) - 1:
    bot.cycle()
    fake.advance()
live = fake.value() / 1000.0

m = idx >= start
P = {f: pd.DataFrame({s: D[s][f] for s in D}) for f in ("o", "h", "l", "c")}
ref = P["c"].rolling(24).max().shift(1).values
eq, lg = lab.dyn_kernel(P["o"].values[m], P["h"].values[m], P["l"].values[m], P["c"].values[m], ref[m],
                        cfg["depth_pct"] / 100, cfg["n_orders"], cfg["max_order_pct"] / 100, cfg["take_profit_pct"] / 100,
                        cfg["max_hours"], cfg["cooldown_hours"], 0.001, 0.001, 0.0002, 0.0)
print(f"{start:%Y-%m-%d} -> {end:%Y-%m-%d}, {len(D)} coins")
print(f"  LIVE BOT on fake Binance: x{live:.4f}  closed trades {bot.state['trades']}  wins {bot.state['wins']}")
print(f"  BACKTEST engine         : x{eq[-1]:.4f}  closed trades {len(lg)}  wins {int((lg[:, 4] > 0).sum())}")
