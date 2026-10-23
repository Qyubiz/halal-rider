"""Idea 1: live vs backtest, trade by trade. Run after 2-4 weeks live.

On the server:   copy $PANIC_DATA/panic_trades.csv here (scp, or cat and paste).
Then:            python3 fetch_vision.py 1h 2026-11-01 BTCUSDT,ETHUSDT,... ~/pb_data/1h --api-tail
                 python3 reconcile_live.py panic_trades.csv 2026-11-01 2026-11-29 [--data ~/pb_data/1h]

Matches every live trade to the backtest trade on the same coin bought within 1 hour of it, then lists fills
the live bot missed, fills only the live bot got, and the price/timing/result gaps on matched trades."""
import argparse
import os
import sys

import numpy as np
import pandas as pd

import engine as E

MYT = pd.Timedelta(hours=8)


def live_trades(path):
    t = pd.read_csv(path)
    t["bought"] = (pd.to_datetime(t.opened_myt) - MYT).dt.floor("1h")
    t["sold"] = (pd.to_datetime(t.closed_myt) - MYT).dt.floor("1h")
    t["ret"] = t.exit.astype(float) * (1 - E.FEE) / (t.entry.astype(float) * (1 + E.FEE)) - 1
    # a take-profit that filled in parts is several rows: join them
    g = t.groupby(["symbol", "bought"], as_index=False).agg(
        entry=("entry", "first"), qty=("qty", "sum"), pnl=("pnl_usdt_after_fees", "sum"), sold=("sold", "max"),
        reason=("reason", "last"), exit=("exit", lambda s: float(np.average(s.astype(float), weights=t.loc[s.index, "qty"]))))
    g["ret"] = g.exit * (1 - E.FEE) / (g.entry * (1 + E.FEE)) - 1
    return g


def backtest_trades(data_dir, syms, start, end, cfg):
    D = {}
    for s in syms:
        p = os.path.join(data_dir, f"{s}.csv")
        if os.path.exists(p):
            d = pd.read_csv(p, index_col=0, parse_dates=True)
            D[s] = d[(d.index >= start - pd.Timedelta(days=2)) & (d.index < end)]
    syms, idx, P = E.panel(D)
    keep = idx >= start
    P = {k: v[keep] for k, v in P.items()}
    e, lg = E.run(P, idx[keep], depth=cfg["depth_pct"] / 100, N=cfg["n_orders"], fmax=cfg["max_order_pct"] / 100,
                  tp=cfg["take_profit_pct"] / 100, hm=cfg["max_hours"], cd=cfg["cooldown_hours"])
    ix = idx[keep]
    bt = pd.DataFrame({"symbol": [syms[int(k)] for k in lg.k], "bought": ix[lg.eb.astype(int)],
                       "sold": ix[lg.xb.astype(int)], "entry": lg.entry, "ret": lg.ret,
                       "reason": np.where(lg.why == 1, "time limit", "take-profit")})
    return e, bt


def reconcile(lv, bt):
    rows, used = [], set()
    for i, r in lv.iterrows():
        c = bt[(bt.symbol == r.symbol) & ((bt.bought - r.bought).abs() <= pd.Timedelta(hours=1))]
        c = c[~c.index.isin(used)]
        if len(c):
            j = c.index[0]; used.add(j); b = bt.loc[j]
            rows.append(dict(symbol=r.symbol, bought=r.bought, status="matched", live_entry=r.entry, bt_entry=b.entry,
                             entry_gap_pct=(r.entry / b.entry - 1) * 100, live_sold=r.sold, bt_sold=b.sold,
                             exit_gap_h=(r.sold - b.sold) / pd.Timedelta(hours=1), live_ret_pct=r.ret * 100,
                             bt_ret_pct=b.ret * 100, live_reason=r.reason, bt_reason=b.reason))
        else:
            rows.append(dict(symbol=r.symbol, bought=r.bought, status="live only", live_entry=r.entry,
                             live_sold=r.sold, live_ret_pct=r.ret * 100, live_reason=r.reason))
    for j, b in bt[~bt.index.isin(used)].iterrows():
        rows.append(dict(symbol=b.symbol, bought=b.bought, status="backtest only", bt_entry=b.entry, bt_sold=b.sold,
                         bt_ret_pct=b.ret * 100, bt_reason=b.reason))
    return pd.DataFrame(rows).sort_values("bought")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("trades"); ap.add_argument("start"); ap.add_argument("end")
    ap.add_argument("--data", default=os.path.join(E.DATA, "1h"))
    ap.add_argument("--out", default="reconcile.csv")
    a = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(E.HERE))))
    import panic_buyer as PB
    cfg = PB.load_cfg()
    start, end = pd.Timestamp(a.start), pd.Timestamp(a.end)
    lv = live_trades(a.trades)
    lv = lv[(lv.bought >= start) & (lv.bought < end)]
    e, bt = backtest_trades(a.data, cfg["symbols"], start, end, cfg)
    R = reconcile(lv, bt)
    R.to_csv(a.out, index=False)
    m = R[R.status == "matched"]
    print(f"{start:%Y-%m-%d} -> {end:%Y-%m-%d}: live {len(lv)} trades, backtest {len(bt)} trades, matched {len(m)}")
    print(f"  live only (extra fills): {int((R.status == 'live only').sum())}   backtest only (missed live): "
          f"{int((R.status == 'backtest only').sum())}")
    if len(m):
        print(f"  matched: entry live vs backtest {m.entry_gap_pct.mean():+.3f}% on average (worst {m.entry_gap_pct.max():+.3f}%), "
              f"exit {m.exit_gap_h.mean():+.1f} h, result {m.live_ret_pct.mean() - m.bt_ret_pct.mean():+.3f} pts per trade")
        print(f"  same exit reason: {(m.live_reason.str[:4] == m.bt_reason.str[:4]).mean()*100:.0f}%")
    print(f"  average result per trade: live {lv.ret.mean()*100:+.3f}%, backtest {bt.ret.mean()*100:+.3f}%")
    print(f"  details: {a.out}")
    return R


if __name__ == "__main__":
    main()
