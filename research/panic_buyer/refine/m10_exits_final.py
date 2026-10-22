"""m10 exits, step 3: the ONE look at 2024-26, since 2025 and the untouched 2018-20 holdout for the pick chosen on
2021-23 by m10_exits_grid.py g2 (rule in that file), against 90% mode. Normal fills, every fill 0.2% worse (buys and
sells), the real bot's timing (market-buy take-profit at once, cooldown +1h) with fills 0.2% worse, and LUNA/FTT/SRM added.

PICK (highest path-average 2021-23 %/month among eligible g2 designs):
  take-profit  tp = clip(3% x sqrt(rng24 / 0.2074), 1%, 6%) fixed at the buy, rng24 = (highest high - lowest low of the
               24 hours before the buy hour) / last close; placed at the next hourly check as today
  time stop    at 72h: sold at the close if the close is more than 12% under the entry; otherwise kept with the target
               lowered to entry +1% (re-placed at the next hourly check) and sold at the close at 240h
Usage: PB_DATA=... PB_HOLDOUT=... PB_OUT=... python3 m10_exits_final.py"""
import pandas as pd

import m10_exits_lib as X
import w90_wr90_lib as W

PICK = dict(X.BASE90, hm=72, uw=0.12, uwmode=1, hm2=240, be2=0.01,
            tpa=lambda P: X.tp_table(P, "rng24", 0.03, alpha=0.5, lo=0.01, hi=0.06))
DESIGNS = {"90% mode": dict(X.BASE90), "PICK exits": PICK}
VARIANTS = {"normal": {}, "fills 0.2% worse": dict(pen=X.STRESS_PEN, pens=X.STRESS_PEN),
            "bot timing + fills 0.2% worse": dict(mtp_now=True, cd=4, pen=X.STRESS_PEN, pens=X.STRESS_PEN)}


def run(kw, P, idx, periods):
    k = dict(kw)
    if callable(k.get("tpa")):
        k["tpa"] = k["tpa"](P)
    e, lg = X.xrun(P, idx, **k)
    return X.summarize(e, lg, idx, periods), lg


def main():
    rows = []
    mk = {"main": W.market(False), "dead": W.market(True), "hold": X.holdout_market()}
    for dn, kw in DESIGNS.items():
        for vn, over in VARIANTS.items():
            for pn, periods in (("main", X.PERIODS), ("hold", X.HPERIOD)):
                syms, idx, P = mk[pn]
                r, lg = run(dict(kw, **over), P, idx, periods)
                print(X.fmt_row(f"{dn} | {vn} | {'2018-20 holdout' if pn == 'hold' else '2021-26'}", r,
                                tags=("h",) if pn == "hold" else ("d", "t", "s")), flush=True)
                for tg, _, _ in periods:
                    rows.append(dict(design=dn, variant=vn, period=tg, **{c[2:]: v for c, v in r.items()
                                                                          if c.startswith(tg + "_")}))
                if pn == "main" and vn == "normal":
                    why = lg[idx[lg.eb.astype(int).values] >= pd.Timestamp("2024-01-01")].why.value_counts().to_dict()
                    print(f"   2024-26 exits by reason (0 tp, 1 time stop, 6 kept then 240h stop, 9 open): {why}")
        syms, idx, P = mk["dead"]
        r, _ = run(dict(kw, liquidate=True), P, idx, X.PERIODS)
        print(X.fmt_row(f"{dn} | +LUNA/FTT/SRM (sold when they stop trading) | 2021-26", r), flush=True)
        for tg, _, _ in X.PERIODS:
            rows.append(dict(design=dn, variant="+LUNA/FTT/SRM", period=tg,
                             **{c[2:]: v for c, v in r.items() if c.startswith(tg + "_")}))
    df = pd.DataFrame(rows)
    df.to_csv(f"{X.OUT}/final.csv", index=False)
    lab = {"h": "2018-20 holdout", "d": "2021-23", "t": "2024-26", "s": "since 2025"}
    print("\nTABLE: win rate | %/month | worst drop | worst trade (% of account) | trades/month")
    for vn in list(VARIANTS) + ["+LUNA/FTT/SRM"]:
        print(f"\n[{vn}]")
        for tg in ("h", "d", "t", "s"):
            for dn in DESIGNS:
                x = df[(df.design == dn) & (df.variant == vn) & (df.period == tg)]
                if len(x):
                    x = x.iloc[0]
                    print(f"  {lab[tg]:16s} {dn:11s} WR {x.wr*100:5.1f}%  {x.mo*100:+6.2f}%/mo  drop {x.dd*100:5.1f}%  "
                          f"worst trade {x.wt*100:+6.1f}%  {x.tpm:5.1f} tr/mo  (tiny wins {x.tiny*100:.1f}%)")


if __name__ == "__main__":
    main()
