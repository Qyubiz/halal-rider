"""Validate the combined strategy (panic_config.ensemble.example.json) and its hard risk limits.

Part 1, backtest engine (ens_lib, proven equal to engine.run with its switches off), for Normal, 90% mode,
10% mode, the combined sleeves and the combined sleeves with the risk limits exactly as the bot enforces them
(hourly-close stop, per-trade cap, daily/30-day kill-switch, loss-streak pause):
  - 2021-23, 2024-26, since 2025 on the 19 coins, and with LUNA/FTT/SRM + real bot timing + fills 0.2% worse;
  - the 2018-20 holdout;
  - the 6 worst 30-day crashes of an equal-weight index of the coins (2018-2026), each plus 14 days.
Part 2, the real panic_buyer.Ensemble replayed hour by hour through tests/fake_binance.py in the worst windows,
next to the engine on the same coins and candles.

Usage (from research/panic_buyer/refine):
  PB_DATA=... PB_HOLDOUT=... python3 validate_ensemble.py [--no-replay]
"""
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
import ens_lib as L  # noqa: E402

CFG = json.load(open(os.path.join(REPO, "panic_config.ensemble.example.json")))
RISK = CFG["risk"]


def sleeve(sv):
    return dict(depth=sv["depth_pct"] / 100, N=sv["n_orders"], fmax=sv["max_order_pct"] / 100,
                tp=sv["take_profit_pct"] / 100, part=1.0, tp2=0.0, hm=sv["max_hours"], cd=sv["cooldown_hours"])


SLEEVES = [sleeve(sv) for sv in CFG["sleeves"]]
WEIGHTS = [sv["weight"] for sv in CFG["sleeves"]]
GUARD = dict(hs=RISK["stop_loss_pct"] / 100, hsmode=0, zcap=RISK["max_trade_risk_pct"] / 100,
             killd=RISK["daily_dd_pct"] / 100, killm=RISK["monthly_dd_pct"] / 100, klock=RISK["monthly_lock_hours"],
             cln=RISK["loss_streak"], clT=RISK["loss_pause_hours"])
NORMAL = dict(depth=0.12, N=8, fmax=0.2, tp=0.015, part=1.0, tp2=0.0, hm=48, cd=24)
DESIGNS = {
    "Normal (live default)": ([NORMAL], dict(guard=0.0)),
    "90% mode": ([L.MODE90], {}),
    "10% mode": ([L.MODE10], {}),
    "Combined 30/70 (no risk limits)": (SLEEVES, dict(weights=WEIGHTS)),
    "Combined 30/70 + risk limits (bot config)": (SLEEVES, dict(weights=WEIGHTS, **GUARD)),
}
STRESS = dict(mtp_now=True, cd_add=1, pen=0.002, pens=0.002)


def stats(e, lg, idx, a, b):
    m, mo, dd, wm, wd = L.period(e, a, b)
    tb = idx[lg.eb.astype(int).values] if len(lg) else pd.DatetimeIndex([])
    sel = (tb >= a) & ((tb < b) if b is not None else True)
    acct = (lg.ret * lg.amt / lg.eq0).values[sel] if len(lg) else np.zeros(0)
    return dict(mo=mo * 100, dd=dd * 100, wr=float((lg.ret[sel] > 0).mean()) * 100 if sel.any() else np.nan,
                pf=L.pf(acct) if len(acct) else np.nan, worst=float(acct.min()) * 100 if len(acct) else np.nan)


def cell(r):
    return f"{r['wr']:.0f}% / {r['mo']:+.1f}% / {r['dd']:.0f}% / {r['pf']:.1f}"


def part1():
    pers = [("2021-23", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
            ("2024-26", pd.Timestamp("2024-01-01"), None), ("since 2025", pd.Timestamp("2025-01-01"), None)]
    hold = E.panel(E.load_pre())
    print("\n## Win rate / %/month / max drawdown / profit factor\n")
    print("| Design | 2018-20 holdout | " + " | ".join(p[0] for p in pers) + " | stressed 2024-26 |")
    print("|---|---|" + "---|" * len(pers) + "---|")
    runs = {}
    for name, (sv, kw) in DESIGNS.items():
        syms, idx, P = L.market(False)
        e, lg, _, _ = L.erun(P, idx, sv, **kw)
        runs[name] = (e, lg, idx)
        hs, hi, hP = hold
        he, hlg, _, _ = L.erun(hP, hi, sv, **kw)
        h = stats(he, hlg, hi, E.H_START, E.H_END)
        h["mo"] = (L.period(he, E.H_START, E.H_END)[0] ** (1 / 36) - 1) * 100
        s2, i2, P2 = L.market(True)
        se, slg, _, _ = L.erun(P2, i2, sv, liquidate=True, **dict(kw, **STRESS))
        st = stats(se, slg, i2, pers[1][1], None)
        print(f"| {name} | {cell(h)} | " + " | ".join(cell(stats(e, lg, idx, a, b)) for _, a, b in pers)
              + f" | {cell(st)} |")
    return runs


def worst_windows(n=6):
    """The n worst non-overlapping 30-day falls of an equal-weight index of the coins, 2018-2026."""
    frames = []
    for D in (E.load_pre(), E.load_h19()):
        c = pd.DataFrame({s: d.c for s, d in D.items()}).resample("1D").last()
        frames.append(c.pct_change(fill_method=None).mean(axis=1))
    r = pd.concat(frames)
    r = r[~r.index.duplicated()].sort_index().dropna()
    idx_ = (1 + r).cumprod()
    fall = (idx_ / idx_.shift(30) - 1).dropna()
    out = []
    for t, v in fall.sort_values().items():
        if all(abs((t - u).days) > 45 for u, _ in out):
            out.append((t, v))
        if len(out) == n:
            break
    return sorted(out)


def part2(windows):
    print("\n## Worst crash windows (30-day fall of the coin index, +14 days): return / max drawdown inside\n")
    print("| Window (index fall) | " + " | ".join(DESIGNS) + " |")
    print("|---|" + "---|" * len(DESIGNS))
    for t, v in windows:
        a, b = t - pd.Timedelta(days=30), t + pd.Timedelta(days=14)
        pre = a < pd.Timestamp("2021-01-01")
        syms, idx, P = E.panel(E.load_pre(dead=True)) if pre else L.market(True)
        row = []
        for name, (sv, kw) in DESIGNS.items():
            e, lg, _, _ = L.erun(P, idx, sv, liquidate=not pre, **kw)
            x = e[(e.index >= a) & (e.index < b)]
            x = x / x.iloc[0]
            row.append(f"{(x.iloc[-1] - 1) * 100:+.0f}% / {(x / x.cummax() - 1).min() * 100:.0f}%")
        print(f"| {a:%Y-%m-%d} to {b:%Y-%m-%d} ({v * 100:.0f}%) | " + " | ".join(row) + " |")


def replay(a, b):
    """The real Ensemble through the fake exchange on the coins with complete data in [a-8d, b)."""
    import panic_buyer as PB
    import fake_binance as FB

    def klines(self, symbol, interval="1h", limit=30):
        rows = self.k[symbol]
        return [dict(r) for r in rows[max(0, self.i + 1 - limit): self.i + 1]]
    FB.FakeBinance.klines = klines
    idx = pd.date_range(a - pd.Timedelta(days=8), b, freq="1h", inclusive="left")
    D = {}
    for s in E.H19:
        d = E.read_hourly(s)
        if d is None:
            continue
        d = d.reindex(idx)
        if d.c.notna().mean() > 0.995:
            d["c"] = d.c.ffill().bfill()
            for f in ("o", "h", "l"):
                d[f] = d[f].fillna(d["c"])
            D[s] = d
    ep = pd.Timestamp("1970-01-01")
    ms = ((idx - ep) // pd.Timedelta(milliseconds=1)).to_numpy().astype(np.int64)
    candles = {s: [{"open_ms": int(ms[i]), "open": float(o), "high": float(h), "low": float(l_), "close": float(c),
                    "close_ms": int(ms[i] + 3600_000 - 1)} for i, (o, h, l_, c) in enumerate(zip(d.o, d.h, d.l, d.c))]
               for s, d in D.items()}
    fake = FB.FakeBinance(candles, start=int(np.searchsorted(idx, a)), usdt=1000.0, tick="0.0000001", step="0.0000001")
    cfg = dict(PB.DEFAULT_CFG)
    cfg.update(CFG)
    cfg["symbols"] = list(D)
    ens = PB.Ensemble(fake, cfg, 1000.0, compound=True, root=tempfile.mkdtemp(), out=lambda m: None, clock=fake.clock)
    ens.load_info()
    eq = []
    while fake.i < len(idx) - 1:
        ens.cycle()
        fake.advance()
        eq.append(fake.value() / 1000.0)
    eq = pd.Series(eq)
    syms, pidx, P = E.panel(D)
    keep = pidx >= a
    P = {k: v[keep] for k, v in P.items()}
    P["ref"] = pd.DataFrame({s: D[s].c for s in syms}).rolling(24).max().shift(1)[keep]
    e, _, _, _ = L.erun(P, pidx[keep], SLEEVES, weights=WEIGHTS, mtp_now=True, cd_add=1, **GUARD)
    return (eq.iloc[-1] - 1) * 100, (eq / eq.cummax() - 1).min() * 100, (e.iloc[-1] - 1) * 100, \
        (e / e.cummax() - 1).min() * 100, len(D)


def part3(windows):
    print("\n## Real bot code (panic_buyer.Ensemble + guardrails.py) replayed vs the engine, worst windows from 2021\n")
    print("| Window | coins | real bot: return / max drawdown | engine (bot timing): return / max drawdown |")
    print("|---|---|---|---|")
    for t, v in windows:
        a, b = t - pd.Timedelta(days=30), t + pd.Timedelta(days=14)
        if a < pd.Timestamp("2021-01-09"):
            continue
        r, dd, er, edd, k = replay(a, b)
        print(f"| {a:%Y-%m-%d} to {b:%Y-%m-%d} | {k} | {r:+.1f}% / {dd:.1f}% | {er:+.1f}% / {edd:.1f}% |")


if __name__ == "__main__":
    print(f"Config: {json.dumps(CFG['risk'])}")
    part1()
    w = worst_windows()
    part2(w)
    if "--no-replay" not in sys.argv:
        part3(w)
