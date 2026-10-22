"""Replay a copy of panic_buyer.py hour by hour over a long span through the fake exchange; compare with engine
variants on exactly the same coins (those with complete data over the span).
Usage: replay_bot_full.py REPO_DIR START END OUT_DIR  (REPO_DIR holds panic_buyer.py and tests/fake_binance.py)"""
import os, sys, tempfile
import numpy as np, pandas as pd
repo, start, end, out = sys.argv[1], pd.Timestamp(sys.argv[2]), pd.Timestamp(sys.argv[3]), sys.argv[4]
os.makedirs(out, exist_ok=True)
sys.path[:0] = [repo, os.path.join(repo, "tests"), "/home/user/halal-rider/research/panic_buyer/refine"]
import panic_buyer as PB
import fake_binance as FB
import engine as E

def klines(self, symbol, interval="1h", limit=30):       # same result as FakeBinance.klines, without copying history
    rows = self.k[symbol]
    return [dict(r) for r in rows[max(0, self.i + 1 - limit): self.i + 1]]
FB.FakeBinance.klines = klines

idx = pd.date_range(start - pd.Timedelta(days=2), end, freq="1h", inclusive="left")
D = {}
for s in PB.DEFAULT_CFG["symbols"]:
    d = pd.read_csv(os.path.join(E.DATA, "1h", f"{s}.csv"), index_col=0, parse_dates=True).reindex(idx)
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
cfg = dict(PB.DEFAULT_CFG); cfg["symbols"] = list(D)
bot = PB.PanicBuyer(fake, cfg, 1000.0, compound=True, root=out, out=lambda m: None, clock=fake.clock)
bot.load_info()
eqs = []
while fake.i < len(idx) - 1:
    bot.cycle(); fake.advance()
    eqs.append((idx[fake.i], fake.value() / 1000.0))
pd.DataFrame(eqs, columns=["t", "eq"]).to_csv(os.path.join(out, "bot_eq.csv"), index=False)
syms, pidx, P = E.panel(D)
keep = pidx >= start
P = {k: v[keep] for k, v in P.items()}
P["ref"] = pd.DataFrame({s: D[s].c for s in syms}).rolling(24).max().shift(1)[keep]
for lab, kw in (("A", {}), ("B", {"mtp_now": True}), ("A_cd25", {"cd": 25}), ("B_cd25", {"mtp_now": True, "cd": 25})):
    e, lg = E.run(P, pidx[keep], **kw)
    e.to_csv(os.path.join(out, f"engine_eq_{lab}.csv"))
print(f"done {len(D)} coins: {', '.join(s[:-4] for s in D)}; bot trades {bot.state['trades']}")
