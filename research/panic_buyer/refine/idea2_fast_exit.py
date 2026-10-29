"""Idea 2: faster exits. Simulated on 1-minute candles: the take-profit is placed `delay` minutes after the buy
fills (websocket fill alerts ~1 min, polling every 5 or 15 min) instead of at the next hourly cycle.
The comparison base is the live bot on the same 1-minute data (delay 0 = take-profit at the next hour), so data
differences between the 1h and 1m archives can't masquerade as an improvement."""
import os

import pandas as pd

import engine as E
import study as S

_m = {}


def mkt(dead=False):
    if dead not in _m:
        idx, A, hidx, Ph = E.minute_panel(E.H19 + (list(E.DEAD) if dead else []), cuts=E.DEAD if dead else None)
        _m[dead] = (A, hidx, Ph)
    return _m[dead]


def ev(delay, dead=False, **kw):
    A, hidx, Ph = mkt(dead)
    e, lg = E.mrun(A, hidx, Ph, delay, **kw)
    return E.score(e, lg, hidx), lg


V = {f"tp {tp:.2%} placed {d} min after the fill": dict(delay=d, tp=tp)
     for d in (1, 5, 15) for tp in (0.01, 0.0125, 0.015, 0.02)}

if __name__ == "__main__":
    print("\n######## idea 2: faster exits (1-minute candles)")
    A, hidx, Ph = mkt()
    hourly_archive, _, _ = S.evaluate({})
    print(E.line("live, hourly archive candles", hourly_archive))
    base, blg = ev(0)
    print(E.line("live, candles built from 1m (BASE)", base))
    rows = []
    for name, kw in V.items():
        r, lg = ev(**kw)
        r["name"] = name
        r["same_hour_exits"] = int((lg.eb == lg.xb).sum())
        r["beats_design"], r["beats_test"], r["beats_both"] = E.beats(r, base, "design"), E.beats(r, base, "test"), E.beats(r, base)
        rows.append(r)
        print(E.line(name, r) + f"  sold within the buy hour: {r['same_hour_exits']}" + ("  <- beats live in both" if r["beats_both"] else ""))
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(S.OUT, "idea2_fast_exit.csv"), index=False)
    pick = E.pick_on_design(rows, base)
    verdict = {"idea": "idea 2: faster exits", "variants": len(R), "beat_design": int(R.beats_design.sum()),
               "beat_test": int(R.beats_test.sum()), "beat_both": int(R.beats_both.sum()), "pick": None, "passes": False}
    if pick is None:
        print("PICK (2021-23 only): none beats the live settings in 2021-23 -> idea rejected")
    else:
        print(f"PICK (2021-23 only): {pick['name']}")
        checks = {}
        for label, dead, pen in (("+LUNA/FTT/SRM", True, 0.0), ("fills 0.2% worse", False, 0.002)):
            b, _ = ev(0, dead=dead, pen=pen)
            c, _ = ev(**V[pick["name"]], dead=dead, pen=pen)
            checks[label] = E.beats(c, b)
            print(E.line(f"  live, {label}", b)); print(E.line(f"  pick, {label}", c) + ("  ok" if checks[label] else "  FAILS"))
        verdict.update(pick=pick["name"], pick_d_mult=pick["d_mult"], pick_t_mult=pick["t_mult"], pick_d_dd=pick["d_dd"],
                       pick_t_dd=pick["t_dd"], base_d_mult=base["d_mult"], base_t_mult=base["t_mult"], base_d_dd=base["d_dd"],
                       base_t_dd=base["t_dd"], **{f"check {k}": v for k, v in checks.items()},
                       passes=bool(pick["beats_test"] and all(checks.values())))
        print(f"  VERDICT: {'PASSES' if verdict['passes'] else 'rejected, live bot unchanged'}")
    S.save_verdict(verdict)
