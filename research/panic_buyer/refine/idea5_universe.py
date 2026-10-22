"""Idea 5: dynamic universe. On the 1st of each month, rank the screened halal coins (code/universe.py, delisted
coins included) by trading volume over the trailing window and let the bot buy only the top M that month
(coins already held still exit normally). No look-ahead: the ranking uses only past volume.
Note for the user: coins outside the 19 come from the approximate screen and would need the screener's verdict."""
import glob
import os

import numpy as np
import pandas as pd

import engine as E
import study as S
import universe as U

UNIV = os.path.join(E.DATA, "univ")


def load_universe():
    pm = U.pass_map()
    D = {}
    for p in sorted(glob.glob(os.path.join(UNIV, "*.csv"))):
        sym = os.path.basename(p)[:-4]
        if sym[:-4] not in pm or sym in ("RNDRUSDT",):
            continue
        d = pd.read_csv(p, index_col=0, parse_dates=True)
        d = d[d.index >= "2021-01-01"]
        if len(d) >= 24 * 30:
            D[sym] = d
    if "RENDERUSDT" in D and os.path.exists(os.path.join(UNIV, "RNDRUSDT.csv")):
        r = pd.read_csv(os.path.join(UNIV, "RNDRUSDT.csv"), index_col=0, parse_dates=True)
        x = pd.concat([r[r.index >= "2021-01-01"], D["RENDERUSDT"]])
        D["RENDERUSDT"] = x[~x.index.duplicated(keep="last")].sort_index()
    if "LUNAUSDT" in D:
        D["LUNAUSDT"] = D["LUNAUSDT"][D["LUNAUSDT"].index <= E.DEAD["LUNAUSDT"]]   # later rows are LUNA 2.0
    return D


def monthly_top(P, M, look_h, min_hist_h, keep=()):
    """smult: 1 for the month's top-M coins by trailing quote volume (and any `keep` coins), else 0."""
    qv = P["qv"].fillna(0.0)
    vol = qv.rolling(look_h, min_periods=1).sum().shift(1)
    age = P["c"].notna().cumsum().shift(1)
    starts = P["c"].index.to_series().dt.to_period("M")
    first = ~starts.duplicated()
    sm = np.zeros(P["c"].shape)
    sel = None
    keep_ix = [P["c"].columns.get_loc(s) for s in keep if s in P["c"].columns]
    for t, is_first in enumerate(first.values):
        if is_first:
            v = vol.iloc[t].where(age.iloc[t] >= min_hist_h)
            v = v[v > 0].sort_values(ascending=False)
            sel = [P["c"].columns.get_loc(s) for s in v.index[:M]]
        if sel:
            sm[t, sel] = 1.0
        if keep_ix:
            sm[t, keep_ix] = 1.0
    return sm


if __name__ == "__main__":
    print("\n######## idea 5: dynamic universe")
    base, _, _ = S.evaluate({})
    base_pen, _, _ = S.evaluate({}, pen=0.002)
    print(E.line("LIVE SETTINGS (fixed 19 coins)", base))
    D = load_universe()
    syms, idx, P = E.panel(D)
    med = {s: float(d.qv.resample("1D").sum().replace(0, np.nan).median()) for s, d in D.items()}
    rank = pd.Series(med).rank(ascending=False)
    slip = np.array([E.SLIP_MAJOR if rank[s] <= 50 else E.SLIP_OTHER for s in syms])
    print(f"{len(syms)} screened coins with 30+ days of data")
    V = {}
    for M in (20, 30, 50):
        for look in (30, 90):
            for hist in (30, 180):
                V[f"top {M} by {look}d volume, {hist}d+ history"] = (M, look * 24, hist * 24, ())
    for M in (10, 20):
        V[f"19 coins + top {M} by 30d volume"] = (M, 720, 720, tuple(E.H19))
    rows = []
    for name, (M, look, hist, keep) in V.items():
        sm = monthly_top(P, M, look, hist, keep)
        e, lg = E.run(P, idx, smult=sm, slip=slip)
        r = E.score(e, lg, idx); r["name"] = name
        e2, _ = E.run(P, idx, smult=sm, slip=slip, pen=0.002)
        r2 = E.score(e2)
        r["beats_design"], r["beats_test"], r["beats_both"] = E.beats(r, base, "design"), E.beats(r, base, "test"), E.beats(r, base)
        r["beats_both_worse_fills"] = E.beats(r2, base_pen)
        rows.append(r)
        print(E.line(name, r) + ("  <- beats live in both" if r["beats_both"] else ""))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(S.OUT, "idea5_universe.csv"), index=False)
    pick = E.pick_on_design(rows, base)
    v = {"idea": "idea 5: dynamic universe", "variants": len(R), "beat_design": int(R.beats_design.sum()),
         "beat_test": int(R.beats_test.sum()), "beat_both": int(R.beats_both.sum()), "pick": None, "passes": False}
    if pick is None:
        print("PICK (2021-23 only): none beats the live settings in 2021-23 -> idea rejected")
    else:
        print(f"PICK (2021-23 only): {pick['name']}"); print(E.line("  pick", pick))
        v.update(pick=pick["name"], pick_d_mult=pick["d_mult"], pick_t_mult=pick["t_mult"], pick_d_dd=pick["d_dd"],
                 pick_t_dd=pick["t_dd"], base_d_mult=base["d_mult"], base_t_mult=base["t_mult"], base_d_dd=base["d_dd"],
                 base_t_dd=base["t_dd"], **{"check fills 0.2% worse": pick["beats_both_worse_fills"]},
                 passes=bool(pick["beats_test"] and pick["beats_both_worse_fills"]))
        print(f"  VERDICT: {'PASSES' if v['passes'] else 'rejected, live bot unchanged'}")
    S.save_verdict(v)
