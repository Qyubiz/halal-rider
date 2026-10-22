"""Final report for the "velocity" angle. The pick was fixed on 2021-23 only (m10_velocity_grid1.py + grid2.py rule:
win rate >= 92% normal and >= 90% with every fill 0.2% worse, worst drop >= -65%, best 2021-23 %/month with worse
fills). This script is the one look at 2024-26, since 2025 and the untouched 2018-20 holdout.

PICK (rule winner, U1): one position, all free money; each coin's bid sits at the HIGHER of two trigger prices:
   A  24h highest close -14%  -> take-profit +2.5%, time stop 96h          (the 90% mode)
   B  12h highest close -13%  -> take-profit +3%,   time stop 48h          (flash dip)
   cooldown 0h after a win (re-arm at the next hourly check), 3h after a loss; crash guard 35% / 168h.
ALT (U4, highest win rate in the grid-2 top four): same with B = 6h highest close -10%, +2%, 48h.
MODE90: the published 90% mode (24h -14%, +2.5%, 96h, cooldown 3h, guard 35%).

Rows: normal fills; every buy and sell fill 0.2% worse; stressed = real bot timing (take-profit at once after a
market buy, cooldowns from the next hourly check) + fills 0.2% worse + LUNA/FTT/SRM (sold at their last hour).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_lib as V  # noqa: E402

A14 = dict(ref=24, depth=0.14, tp=0.025, hm=96)
DESIGNS = {
    "MODE90 (published 90% mode)": V.MODE90,
    "PICK U1: 24h-14%/+2.5%/96h OR 12h-13%/+3%/48h, re-arm after win": dict(
        tiers=[dict(A14, cdw=0, cdl=3), dict(ref=12, depth=0.13, tp=0.03, hm=48, cdw=0, cdl=3)],
        nslots=1, excl=1, dyn=0, guard=0.35),
    "ALT U4: 24h-14%/+2.5%/96h OR 6h-10%/+2%/48h, re-arm after win": dict(
        tiers=[dict(A14, cdw=0, cdl=3), dict(ref=6, depth=0.10, tp=0.02, hm=48, cdw=0, cdl=3)],
        nslots=1, excl=1, dyn=0, guard=0.35),
}
LAB = {"d": "2021-23", "t": "2024-26", "s": "since 2025", "h": "2018-20 holdout"}


def line(tag, p, r):
    return (f"  {tag:9s} {LAB[p]:16s} win {r[f'{p}_wr']*100:5.1f}% | {r[f'{p}_mo']*100:+6.2f}%/mo | max drop "
            f"{r[f'{p}_dd']*100:5.1f}% | worst trade {r[f'{p}_wt']*100:+6.1f}% of acct | {r[f'{p}_tpm']:5.1f} trades/mo | "
            f"no position {r[f'{p}_cashh']*100:4.1f}% of hours (avg cash {r[f'{p}_cash']*100:4.1f}%)")


def main():
    for name, kw in DESIGNS.items():
        print(f"\n{name}")
        for tag, opt in (("normal", {}), ("worse", dict(worse=True)), ("stressed", dict(stress=True))):
            r, e, lg, inv = V.evaluate(kw, **opt)
            rh, eh, lgh, invh = V.evaluate(kw, holdout=True, **opt)
            for p in ("d", "t", "s"):
                print(line(tag, p, r))
            print(line(tag, "h", rh))
            if tag == "normal" and len(lg.tier.unique()) > 1:
                print("  per trigger (normal): " + "; ".join(
                    f"{LAB[p]} A {r[f'{p}_n0']} trades {r[f'{p}_wr0']*100:.1f}% / B {r.get(f'{p}_n1', 0)} trades "
                    f"{r.get(f'{p}_wr1', float('nan'))*100:.1f}%" for p in ("d", "t", "s")) +
                    f"; holdout A {rh['h_n0']} {rh['h_wr0']*100:.1f}% / B {rh.get('h_n1', 0)} {rh.get('h_wr1', float('nan'))*100:.1f}%")
        print(f"  2022: {r['y2022']*100:+.0f}% (stressed run)")


if __name__ == "__main__":
    main()
