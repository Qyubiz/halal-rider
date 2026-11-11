"""The one look at the untouched Jan 2018 - Dec 2020 holdout (see ROUND3_PREREG.md).

Usage: PB_HOLDOUT=... python3 holdout_eval.py [module:function ...]
Each extra argument names a round-3 candidate's holdout_candidate() -> (name, spec, runner).
Runs A (live settings, backtest timing), B (live settings as panic_buyer.py is coded) and the candidates,
each with normal fills, fills 0.2% worse, and LUNA/FTT/SRM added, then applies the pre-registered decisions."""
import importlib
import os
import sys

import pandas as pd

import engine as E


def evaluate(runner, spec, dead=False, **over):
    D = E.load_pre(dead=dead)
    syms, idx, P = E.panel(D)
    kw = dict(spec(P, syms)) if callable(spec) else dict(spec)
    kw.update(over)
    e, lg = runner(P, idx, **kw)
    return E.score_holdout(e, lg, idx)


def line(name, r):
    yrs = " ".join(f"{y}:{r.get(f'y{y}', float('nan'))*100:+.0f}%" for y in (2018, 2019, 2020))
    return (f"{name:52s} 2018-20 x{r['h_mult']:5.2f} drop {r['h_dd']*100:4.0f}% worst-mo {r['h_worst_m']*100:+3.0f}% "
            f"up-mo {r['h_pos_m']*100:3.0f}% | {yrs} | trades {r.get('h_trades', 0)}")


def main(argv):
    cands = [("A live settings (backtest timing)", {}, E.run),
             ("B live settings as coded (market-buy sell at once)", {"mtp_now": True}, E.run)]
    for ref in argv:
        mod, fn = ref.split(":")
        sys.path.insert(0, os.path.dirname(os.path.abspath(mod)) or ".")
        m = importlib.import_module(os.path.basename(mod).removesuffix(".py"))
        cands.append(getattr(m, fn)())
    D = E.load_pre()
    print(f"holdout coins: {len(D)} ({', '.join(s[:-4] for s in D)}), {E.H_START:%Y-%m-%d} -> {E.H_END:%Y-%m-%d}")
    rows = {}
    for name, spec, runner in cands:
        rows[name] = {lab: evaluate(runner, spec, dead=dead, **over)
                      for lab, dead, over in (("normal", False, {}), ("fills 0.2% worse", False, {"pen": 0.002}),
                                              ("+LUNA/FTT/SRM", True, {}))}
        for lab, r in rows[name].items():
            print(line(f"{name[:36]} | {lab}", r))
    A, B = rows[cands[0][0]], rows[cands[1][0]]
    print("\nDECISIONS (fixed in ROUND3_PREREG.md before this run)")
    b_wins = E.beats_holdout(B["normal"], A["normal"])
    print(f"1. market-buy timing: B {'BEATS' if b_wins else 'does not beat'} A on the holdout -> "
          f"{'keep panic_buyer.py as coded and restate results as B' if b_wins else 'change the bot to match A (take-profit after a market buy at the next hourly cycle)'}")
    for name, spec, runner in cands[2:]:
        c = rows[name]
        ok = E.beats_holdout(c["normal"], A["normal"]) and E.beats_holdout(c["fills 0.2% worse"], A["fills 0.2% worse"])
        print(f"2. {name}: {'BEATS A on the holdout with normal and worse fills -> goes to the user' if ok else 'does not beat A on the holdout -> rejected'}"
              f" (+LUNA/FTT/SRM: {'beats' if E.beats_holdout(c['+LUNA/FTT/SRM'], A['+LUNA/FTT/SRM']) else 'does not beat'} A)")
    out = os.environ.get("PB_OUT", os.path.expanduser("~/pb_out"))
    os.makedirs(out, exist_ok=True)
    pd.DataFrame([{"candidate": n, "test": lab, **r} for n, v in rows.items() for lab, r in v.items()]).to_csv(
        os.path.join(out, "holdout_2018_20.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
