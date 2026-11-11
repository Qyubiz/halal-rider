"""Ensemble audit, agent 1: re-run the three live-bot modes (Normal, 90% mode, 10% mode) and report, per period,
win rate, %/month, worst drop, PROFIT FACTOR, trades and worst single trade as % of the account.

Method = results_w90/check90.log: engine.run with the crash guard as an smult mask (w90_wr90_lib.guard_mask), cross-
checked against w90_wr90_lib.wrun (identical equity; wrun also logs positions still open at the end, why=9, and the
account equity when each order was sized, eq0). Periods 2021-23 / 2024-26 / since 2025 (trades assigned by entry hour),
and the untouched 2018-20 holdout (engine.load_pre, as holdout_eval.py / results_w90/holdout_w90.log).

Profit factor:
  PF_acct  = sum of winning trades' P&L as % of the account at entry / |sum of losing trades'| (ret*amt/eq0).
  PF_trade = sum of winning per-trade returns / |sum of losing per-trade returns| (every trade weighted equally).
  Open positions at the end of the data (why=9) count, marked at the last close, like check90.log's win rates.

Usage (from research/panic_buyer/refine):
  PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 ens_audit_modes.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OUT = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out_ens/audit"))
os.makedirs(OUT, exist_ok=True)
os.environ["PB_OUT"] = OUT          # w90_wr90_lib makes its own output dir from PB_OUT at import
import engine as E  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

MODES = {
    # panic_buyer.py DEFAULT_CFG (lines 54-66) == engine.LIVE
    "Normal": dict(depth=0.12, N=8, fmax=0.20, tp=0.015, hm=48, cd=24, guard=0.0),
    # panic_config.90wr.example.json
    "90% mode": dict(depth=0.14, N=1, fmax=1.0, tp=0.025, hm=96, cd=3, guard=0.35),
    # panic_config.10pct.example.json
    "10% mode": dict(depth=0.14, N=1, fmax=1.0, tp=0.05, hm=48, cd=24, guard=0.35),
}
PER = (("2021-23", pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")),
       ("2024-26", pd.Timestamp("2024-01-01"), None),
       ("since 2025", pd.Timestamp("2025-01-01"), None))
HOLD = (("2018-20 holdout", E.H_START, E.H_END),)


def pf(x):
    g, l_ = x[x > 0].sum(), -x[x <= 0].sum()
    return float(g / l_) if l_ > 0 else float("inf")


def stats(e, lg, idx, periods):
    rows = []
    tb = idx[lg.eb.astype(int).values]
    acct = (lg.ret * lg.amt / lg.eq0).values
    for name, a, b in periods:
        m, mo, dd = W.period(e, a, b)
        sel = (tb >= a) & ((tb < b) if b is not None else True)
        x = lg[sel]; xa = acct[sel]
        r = x.ret.values
        wi = int(np.argmin(xa)) if len(xa) else -1
        rows.append(dict(period=name, mult=m, pct_mo=mo * 100, max_dd=dd * 100, trades=len(x),
                         open_at_end=int((x.why == 9).sum()), win_rate=(r > 0).mean() * 100 if len(r) else np.nan,
                         pf_acct=pf(xa), pf_trade=pf(r), avg_win=r[r > 0].mean() * 100 if (r > 0).any() else np.nan,
                         avg_loss=r[r <= 0].mean() * 100 if (r <= 0).any() else np.nan,
                         worst_trade_acct=xa[wi] * 100 if wi >= 0 else np.nan,
                         worst_trade_when=str(idx[int(x.eb.values[wi])])[:16] if wi >= 0 else "",
                         worst_trade_coin=x.k.values[wi] if wi >= 0 else -1,
                         time_stops=int((x.why == 1).sum()), take_profits=int((x.why == 0).sum())))
    return rows


def run_mode(P, idx, kw, **over):
    k = dict(kw); k.update(over)
    g = k.pop("guard")
    sm = W.guard_mask(P, g) if g > 0 else 1.0
    liq = k.pop("liquidate", False)
    pens = k.pop("pens", 0.0)
    e1, l1 = E.run(P, idx, smult=sm, **k)                      # the check90.log method
    e2, l2 = W.wrun(P, idx, guard=g, liquidate=liq, pens=pens, **k)
    if not liq and pens == 0.0:
        assert np.array_equal(e1.values, e2.values), "engine.run and wrun differ"
        assert np.array_equal(l1[E.LOG_COLS].values, l2[l2.why != 9][E.LOG_COLS].values)
    return e2, l2


def main():
    out = []
    syms, idx, P = W.market(False)
    _, idxd, Pd = W.market(True)
    print(f"2021-26 panel: {len(syms)} coins, {idx[0]} -> {idx[-1]}", flush=True)
    for mode, kw in MODES.items():
        variants = [("normal", P, idx, {}),
                    ("bot timing (mtp_now, cd+1)", P, idx, dict(mtp_now=True, cd=kw["cd"] + 1)),
                    ("bot timing + buys 0.2% worse", P, idx, dict(mtp_now=True, cd=kw["cd"] + 1, pen=0.002)),
                    ("bot timing + buys&sells 0.2% worse + LUNA/FTT/SRM", Pd, idxd,
                     dict(mtp_now=True, cd=kw["cd"] + 1, pen=0.002, pens=0.002, liquidate=True)),
                    ("+LUNA/FTT/SRM (sold at last bar)", Pd, idxd, dict(liquidate=True))]
        if kw["guard"] > 0:
            variants.append(("+LUNA/FTT/SRM, NO crash guard", Pd, idxd, dict(liquidate=True, guard=0.0)))
        for vname, PP, ii, over in variants:
            e, lg = run_mode(PP, ii, kw, **over)
            for r in stats(e, lg, ii, PER):
                r["coin"] = list(PP["o"].columns)[int(r.pop("worst_trade_coin"))] if r["trades"] else ""
                out.append(dict(mode=mode, variant=vname, **r))
    # holdout 2018-20
    D = E.load_pre()
    hs, hidx, HP = E.panel(D)
    print(f"holdout panel: {len(hs)} coins ({', '.join(s[:-4] for s in hs)}), {hidx[0]} -> {hidx[-1]}", flush=True)
    for mode, kw in MODES.items():
        for vname, over in (("normal", {}), ("buys 0.2% worse", dict(pen=0.002)),
                            ("bot timing + buys 0.2% worse", dict(mtp_now=True, cd=kw["cd"] + 1, pen=0.002))):
            e, lg = run_mode(HP, hidx, kw, **over)
            lg_h = lg[hidx[lg.eb.astype(int).values] >= E.H_START]
            sc = E.score_holdout(e, lg_h[lg_h.why != 9], hidx)
            for r in stats(e, lg, hidx, HOLD):
                r["coin"] = list(HP["o"].columns)[int(r.pop("worst_trade_coin"))] if r["trades"] else ""
                r["closed_trades_engine_score"] = sc.get("h_trades")
                # holdout_w90.log convention: engine.metrics, equity rebased at the first bar >= 2018-01-01
                r["pct_mo_log_convention"] = ((1 + sc["h_cagr"]) ** (1 / 12) - 1) * 100
                r["mult_log_convention"] = sc["h_mult"]
                r["years"] = " ".join(f"{y} {sc.get(f'y{y}', np.nan)*100:+.0f}%" for y in (2018, 2019, 2020))
                out.append(dict(mode=mode, variant=vname, **r))
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(OUT, "modes_audit.csv"), index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 500)
    cols = ["mode", "variant", "period", "pct_mo", "win_rate", "max_dd", "pf_acct", "pf_trade", "trades", "open_at_end",
            "avg_win", "avg_loss", "worst_trade_acct", "worst_trade_when", "coin", "time_stops", "mult"]
    print(df[cols].to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    h = df[df.period == "2018-20 holdout"]
    print("\nholdout, holdout_w90.log convention (engine.metrics):")
    print(h[["mode", "variant", "pct_mo_log_convention", "mult_log_convention", "closed_trades_engine_score",
             "years"]].to_string(index=False, float_format=lambda v: f"{v:.2f}"))


if __name__ == "__main__":
    main()
