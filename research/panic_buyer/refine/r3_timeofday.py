"""Round 3, task "timeofday": do dips bought in thin hours (Asian morning, weekends) behave differently?

The only input is the UTC hour / weekday of the ORDER PLACEMENT, i.e. the start of the hourly cycle in which the bid
(or the market buy) goes out: P["o"].index (candle open times, UTC). That is known before the order is placed, so
nothing looks ahead, and it works on any panel (any coins, any dates).

Motivation, 2021-23 ONLY (r3_timeofday_motiv.py, r3_timeofday_motiv2.py; panel cut at 2024-01-01; logs motiv.log,
motiv2.log in this task's PB_OUT):
  * Thin hours by 2021-23 quote volume (19 coins): UTC 03, 04, 05 and 21, 22, 23 are the thinnest in every year;
    weekends carry 11.0% / 11.1% of volume per day against 15.2-16.1% on weekdays.
  * But the weak live entries are in the Asian-morning block, not in all thin hours. Live trades placed in UTC 00-07:
    548 trades, win 90.3%, avg +0.39%, 9.7% time stops, 8.5h held, against 1,133 / 95.1% / +1.27% / 4.9% / 5.3h for
    UTC 08-23; weaker in each year (2021 +0.60% vs +1.49%, 2022 -0.06% vs +0.69%, 2023 +0.57% vs +1.52%).
    UTC 03-06 (the thinnest Asian hours): 246 trades, avg -0.15%, 13.4% time stops (2021 +0.14% vs +1.36%, 2022 -0.22%
    vs +0.58%, 2023 -1.60% vs +1.60%). The thin late-US hours 20-23 are among the best (284 trades, avg +1.49%,
    3.9% time stops).
  * Weekends (Sat-Sun UTC): 330 trades, avg +0.67% vs +1.06% on weekdays, but no worse in 2022 (+0.46% vs +0.42%).
    Almost all of the weekend weakness is its 00-07 hours: 134 trades, avg -0.64%, 15.7% time stops (2021 -0.92%,
    2022 +0.04%, 2023 -0.96%), while weekend 08-23 entries are strong (+2.10%, +0.32%, +1.95%, +2.01% by block).
  * The deeper low (fill hour + 5h vs ref) is about the same in every block (median -14.9% to -15.8%), so thin-hour
    dips are not simply deeper; they recover less often within 48h.
  * Standalone replay (each coin alone, no cash/N limit, live exits; all 2021-23 trades): live sum of returns +17.85
    over 10,853 hours held (16.45 bp/h). UTC 00-07: skip +17.28 (18.56 bp/h), 14% +18.83 (20.12), 16% +17.94 (19.08).
    UTC 03-06: skip +18.53 (19.36), 14% +18.67 (18.54), 16% +19.05 (19.47). Whole weekend: skip +15.04 (16.41),
    14% +16.85 (16.98), 16% +15.94 (16.24). Weekend 00-07: skip +20.40 (21.29), 14% +18.65 (18.14), 16% +20.00 (20.28).
    So skipping or deepening frees capital for better hours at no loss of total edge, except over the whole weekend.

Treatments (inside the window only; outside it everything is the live bot):
  skip      smult 0: no order is placed in those hours. A coin still under its 12% trigger when the window ends is
            bought at market at that hour's open, as the live bot does for any coin already under its trigger.
  depth d   the trigger in those hours is the 24h high x (1 - d) instead of x 0.88.
  half      smult 0.5: the order (and so the position) is half the normal size.

H1 (every day): window UTC 00-07 or UTC 03-06; skip, depth 14%, depth 16%, half size.  8 variants.
H2 (weekends, Sat-Sun UTC): window all day or 00-07 only; skip, depth 14%, depth 16%, half size.  8 variants.
The protocol is study.study() unchanged, on the stock engine.run (no new engine logic: smult and depth already take a
value per hour and coin). Each grid is run once.
"""
import numpy as np

import engine as E
import study as S


def window(kind, h0=0, h1=23):
    """Boolean (n,) mask over the panel's hours: UTC hour in [h0, h1], on every day ('day') or on Sat-Sun ('wkend')."""
    def mask(P):
        ix = P["o"].index
        m = (ix.hour >= h0) & (ix.hour <= h1)
        if kind == "wkend":
            m &= ix.dayofweek >= 5
        return np.asarray(m)
    return mask


def variant(win, treat):
    """treat: 'skip', 'half' or a trigger depth (float). Returns f(P, syms) -> engine.run kwargs."""
    def spec(P, syms):
        m = win(P)
        if treat == "skip":
            return {"smult": np.where(m, 0.0, 1.0)}
        if treat == "half":
            return {"smult": np.where(m, 0.5, 1.0)}
        return {"depth": np.where(m, float(treat), E.LIVE["depth"])}
    return spec


TREATS = (("skip", "skip"), ("depth 14%", 0.14), ("depth 16%", 0.16), ("half size", "half"))

# ------------------------------------------------------------------ the grids (fixed before any result is seen)
H1 = {}
for wname, w in (("UTC 00-07", window("day", 0, 7)), ("UTC 03-06", window("day", 3, 6))):
    for tname, t in TREATS:
        H1[f"{wname} {tname}"] = variant(w, t)

H2 = {}
for wname, w in (("Sat-Sun 00-07", window("wkend", 0, 7)), ("Sat-Sun all day", window("wkend", 0, 23))):
    for tname, t in TREATS:
        H2[f"{wname} {tname}"] = variant(w, t)

assert len(H1) <= 8 and len(H2) <= 8


def plumbing_check():
    """A window that never opens must give exactly the live equity curve and trades (no 2024-26 number printed)."""
    ok = True
    for dead, pen in ((False, 0.0), (True, 0.0), (False, 0.002)):
        syms, idx, P = S.market(dead)
        e0, l0 = E.run(P, idx, pen=pen)
        for t in ("skip", "half", 0.16):
            kw = variant(window("day", 99, 99), t)(P, syms)
            e1, l1 = E.run(P, idx, pen=pen, **kw)
            same = np.array_equal(e0.values, e1.values) and l0.equals(l1)
            ok &= same
        print(f"empty window reproduces engine.run live (dead={dead}, pen={pen}): {ok}")
    syms, idx, P = S.market(False)
    m = window("day", 0, 7)(P)
    assert m.sum() == (idx.hour < 8).sum() and window("wkend", 0, 23)(P).sum() == (idx.dayofweek >= 5).sum()
    if not ok:
        raise SystemExit("variant plumbing does not reproduce engine.run - stop")


def anatomy(name, spec, win):
    """Trades whose order hour is inside the window vs outside, per period (printed after the grids have run)."""
    syms, idx, P = S.market(False)
    m = win(P)
    for lab, sp in (("LIVE SETTINGS", {}), (name, spec)):
        kw = sp(P, syms) if callable(sp) else dict(sp)
        e, lg = E.run(P, idx, **kw)
        eb = lg.eb.astype(int).values
        per = np.where(idx[eb] < E.DESIGN_END, "21-23", "24-26")
        for p in ("21-23", "24-26"):
            out = []
            for inside in (True, False):
                g = lg[(per == p) & (m[eb] == inside)]
                out.append(f"{'in' if inside else 'out'} window: {len(g)} trades, avg {g.ret.mean() * 100:+.2f}%, "
                           f"TS {int((g.why == 1).sum())}, sum ret x size {(g.ret * g.amt).sum():+.2f}")
            print(f"  anatomy {lab} {p}: " + " | ".join(out))


if __name__ == "__main__":
    plumbing_check()
    R1, v1 = S.study("r3 timeofday H1: Asian-morning hours (UTC 00-07 / 03-06) skip, deeper or smaller", H1,
                     "r3_timeofday_h1_asia")
    S.save_verdict(v1)
    R2, v2 = S.study("r3 timeofday H2: weekends (Sat-Sun UTC, all day / 00-07) skip, deeper or smaller", H2,
                     "r3_timeofday_h2_weekend")
    S.save_verdict(v2)
    WINS = {**{k: window("day", 0, 7) for k in H1 if k.startswith("UTC 00-07")},
            **{k: window("day", 3, 6) for k in H1 if k.startswith("UTC 03-06")},
            **{k: window("wkend", 0, 7) for k in H2 if k.startswith("Sat-Sun 00-07")},
            **{k: window("wkend", 0, 23) for k in H2 if k.startswith("Sat-Sun all day")}}
    for v, V in ((v1, H1), (v2, H2)):
        if v["pick"]:
            anatomy(v["pick"], V[v["pick"]], WINS[v["pick"]])
