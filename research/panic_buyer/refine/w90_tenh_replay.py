"""Replay a (patched) copy of panic_buyer.py hour by hour through tests/fake_binance.py and compare it with
w90_tenh_lib.trun on exactly the same coins and candles. Adapted from replay_bot_full.py (unchanged), with the
engine comparison done by trun so the re-arm switch (cdw) is modelled.
Usage: w90_tenh_replay.py REPO_DIR START END OUT_DIR CONFIG_JSON
  REPO_DIR holds panic_buyer.py (+ binance_spot.py) and tests/fake_binance.py; CONFIG_JSON overrides DEFAULT_CFG
  (cooldown_after_win_hours needs the patched copy: cooldown_after_win.diff).
Engine rows: A = engine timing; B = the bot's timing (take-profit at once after a market buy, cooldown counted from
the next hourly check: cd+1, and cdw = bot win cooldown + 1)."""
import json
import os
import sys

import numpy as np
import pandas as pd

repo, start, end, out = sys.argv[1], pd.Timestamp(sys.argv[2]), pd.Timestamp(sys.argv[3]), sys.argv[4]
over = json.load(open(sys.argv[5])) if len(sys.argv) > 5 else {}
os.makedirs(out, exist_ok=True)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [repo, os.path.join(repo, "tests"), HERE]
import panic_buyer as PB  # noqa: E402
import fake_binance as FB  # noqa: E402
import engine as E  # noqa: E402
import w90_tenh_lib as T  # noqa: E402


def klines(self, symbol, interval="1h", limit=30):       # same result as FakeBinance.klines, without copying history
    rows = self.k[symbol]
    return [dict(r) for r in rows[max(0, self.i + 1 - limit): self.i + 1]]


FB.FakeBinance.klines = klines

idx = pd.date_range(start - pd.Timedelta(days=8), end, freq="1h", inclusive="left")
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
cfg = dict(PB.DEFAULT_CFG)
cfg.update(over)
cfg["symbols"] = list(D)
bot = PB.PanicBuyer(fake, cfg, 1000.0, compound=True, root=out, out=lambda m: None, clock=fake.clock)
bot.load_info()
eqs = []
while fake.i < len(idx) - 1:
    bot.cycle()
    fake.advance()
    eqs.append((idx[fake.i], fake.value() / 1000.0))
be = pd.DataFrame(eqs, columns=["t", "v"]).set_index("t")["v"]
be.to_csv(os.path.join(out, "bot_eq.csv"))
syms, pidx, P = E.panel(D)
keep = pidx >= start
P = {k: v[keep] for k, v in P.items()}
P["ref"] = pd.DataFrame({s: D[s].c for s in syms}).rolling(24).max().shift(1)[keep]
g = cfg.get("crash_guard_pct", 0.0) / 100
cl = pd.DataFrame({s: D[s].c for s in syms})
gm = ((cl / cl.rolling(168, min_periods=1).max() - 1) >= -g).shift(1, fill_value=True)[keep].values.astype(float) if g else 1.0
base = dict(depth=cfg["depth_pct"] / 100, N=cfg["n_orders"], fmax=cfg["max_order_pct"] / 100,
            tp=cfg["take_profit_pct"] / 100, hm=cfg["max_hours"], cd=cfg["cooldown_hours"], smult=gm, slip=E.SLIP_MAJOR)
cw = cfg.get("cooldown_after_win_hours")
rows = {}
for lab, kw in (("A engine timing", dict(cdw=(cw + 1) if cw is not None else -1)),
                ("B bot timing", dict(mtp_now=True, cd=base["cd"] + 1, cdw=(cw + 1) if cw is not None else -1))):
    e, lg = T.trun(P, pidx[keep], **{**base, **kw})
    rows[lab] = (e, lg)
lines = [f"{len(D)} coins: {', '.join(s[:-4] for s in D)} | config {json.dumps(over)} | bot trades {bot.state['trades']}, "
         f"wins {bot.state['wins']} ({bot.state['wins'] / max(1, bot.state['trades']) * 100:.1f}%)"]
pers = [(a, b) for a, b in ((pd.Timestamp("2021-01-01"), T.T24), (T.T24, None), (T.T25, None)) if b is None or b > start]
for lab, e in [("real panic_buyer.py (patched copy), replayed", be)] + [(k, v[0]) for k, v in rows.items()]:
    cells = []
    for a, b in pers:
        a2 = max(a, start)
        m, mo, dd = T.period(e, a2, b)
        cells.append(f"{a2.date()}..{(b.date() if b is not None else 'end')} {mo*100:+6.2f}%/mo dd {dd*100:4.0f}%")
    lines.append(f"{lab:46s} " + " | ".join(cells))
for k, (e, lg) in rows.items():
    eb = pd.DatetimeIndex(pidx[keep][lg.eb.astype(int).values])
    lines.append(f"   {k}: trades {len(lg)}, win {(lg.ret > 0).mean() * 100:.1f}% (since 2024 "
                 f"{(lg.ret[eb >= T.T24] > 0).mean() * 100:.1f}%)")
open(os.path.join(out, "replay.log"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
