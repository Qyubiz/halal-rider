"""m10 regime, HINDSIGHT diagnostic (not a pick): runs every m10_regime_grid.py design on the 2018-20 holdout too and
asks whether ANY of them beats 90% mode on win rate in all of 2018-20, 2024-26 and since 2025 while keeping %/month
(i.e. whether the regime angle has any out-of-sample potential at all), plus per-year 2021/2022/2023 numbers that a
stability-based selection (m10_regime_robust.py) can use.

Output: $PB_OUT/hindsight.csv, hindsight.log
"""
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_regime_final as FN  # noqa: E402
import m10_regime_lib as L  # noqa: E402
import w90_wr90_lib as W  # noqa: E402

YEARS = [(str(y), pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y + 1}-01-01")) for y in (2021, 2022, 2023)]


def year_stats(e, lg, idx):
    out = {}
    tb = idx[lg.eb.astype(int).values]
    for y, a, b in YEARS:
        m, mo, dd = W.period(e, a, b)
        x = lg[(tb >= a) & (tb < b)]
        out[f"y{y}_mo"], out[f"y{y}_dd"] = mo, dd
        out[f"y{y}_wr"] = float((x.ret > 0).mean()) if len(x) else np.nan
        out[f"y{y}_n"] = len(x)
    return out


def main():
    t0 = time.time()
    R = pd.read_csv(os.path.join(L.OUT, "grid.csv"))
    rows = []
    for i, r in R.iterrows():
        rules = FN.parse(r["rules"])
        deep = r["deep"] if r["deep"] > 0 else None
        rh, _, _, _ = L.run("h", rules=rules, deep=deep)
        _, e, lg, idx = L.run("d", rules=rules, deep=deep)
        rows.append({**r.to_dict(), **{k: v for k, v in rh.items()}, **year_stats(e, lg, idx)})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(R)} {time.time() - t0:.0f}s", flush=True)
    H = pd.DataFrame(rows)
    H.to_csv(os.path.join(L.OUT, "hindsight.csv"), index=False)
    b = H[H.stage == "base"].iloc[0]
    log = open(os.path.join(L.OUT, "hindsight.log"), "w")

    def p(s):
        print(s, flush=True)
        log.write(s + "\n")

    p(f"{len(H)} designs. 90% mode: WR 2018-20 {b.h_wr*100:.1f} / 2021-23 {b.d_wr*100:.1f} / 2024-26 {b.t_wr*100:.1f} / "
      f"since25 {b.s_wr*100:.1f}; %/mo {b.h_mo*100:+.2f} / {b.d_mo*100:+.2f} / {b.t_mo*100:+.2f} / {b.s_mo*100:+.2f}")
    X = H[H.stage != "base"]
    wr_all = X[(X.h_wr > b.h_wr) & (X.t_wr > b.t_wr) & (X.s_wr > b.s_wr) & (X.d_wr > b.d_wr)]
    both = wr_all[(wr_all.h_mo >= b.h_mo) & (wr_all.t_mo >= b.t_mo) & (wr_all.s_mo >= b.s_mo)]
    p(f"designs with a higher win rate than 90% mode in ALL four periods: {len(wr_all)}; of those, also %/mo >= 90% "
      f"mode in 2018-20, 2024-26 and since 2025: {len(both)}")
    p(f"designs with %/mo >= 90% mode in 2018-20 AND 2024-26: {int(((X.h_mo >= b.h_mo) & (X.t_mo >= b.t_mo)).sum())}")
    p(f"best 2024-26 WR: {X.t_wr.max()*100:.1f}%  best 2018-20 WR: {X.h_wr.max()*100:.1f}%  best since-25 WR "
      f"{X.s_wr.max()*100:.1f}%")
    X = X.assign(min_wr_oos=X[["h_wr", "t_wr", "s_wr"]].min(axis=1), min_mo_oos=X[["h_mo", "t_mo", "s_mo"]].min(axis=1))
    p("\nHINDSIGHT top 15 by the lowest out-of-sample win rate (2018-20, 2024-26, since 2025):")
    for _, r in X.sort_values(["min_wr_oos", "min_mo_oos"], ascending=False).head(15).iterrows():
        p(f"  {r['name'][:84]:84s} WR {r.h_wr*100:4.1f}/{r.d_wr*100:4.1f}/{r.t_wr*100:4.1f}/{r.s_wr*100:4.1f} "
          f"%/mo {r.h_mo*100:+5.2f}/{r.d_mo*100:+5.2f}/{r.t_mo*100:+5.2f}/{r.s_mo*100:+5.2f} DD {r.h_dd*100:3.0f}/"
          f"{r.d_dd*100:3.0f}/{r.t_dd*100:3.0f}")
    p("\nHINDSIGHT top 15 by the lowest out-of-sample %/month with every OOS win rate >= 90% mode's - 0.5pt:")
    Y = X[(X.h_wr >= b.h_wr - 0.005) & (X.t_wr >= b.t_wr - 0.005) & (X.s_wr >= b.s_wr - 0.005)]
    for _, r in Y.sort_values("min_mo_oos", ascending=False).head(15).iterrows():
        p(f"  {r['name'][:84]:84s} WR {r.h_wr*100:4.1f}/{r.d_wr*100:4.1f}/{r.t_wr*100:4.1f}/{r.s_wr*100:4.1f} "
          f"%/mo {r.h_mo*100:+5.2f}/{r.d_mo*100:+5.2f}/{r.t_mo*100:+5.2f}/{r.s_mo*100:+5.2f} DD {r.h_dd*100:3.0f}/"
          f"{r.d_dd*100:3.0f}/{r.t_dd*100:3.0f}")
    p(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
