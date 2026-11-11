"""x10 lever "universe", step 1: the pre-registered grid on 2021-26 (normal costs).

PRE-REGISTRATION (written 22 Oct 2026 before this grid was run; the pick script applies it unchanged)
----------------------------------------------------------------------------------------------------
Known before running (from earlier work, so the grid's location is partly hindsight-informed): concentrated
settings on the 19 coins did best in 2024-26 (+5.5-7.0%/month); idea5 found plain wider volume universes hurt
with 8 x 20% orders; a quick timing run (x10_universe_check.py) showed top-20/top-50 universes with
2 x 50% orders made far less over 2021-26 than the 19 coins with the same settings.

GRID (1,512 variants; hm 48h, fee 0.1%/side, slippage by monthly volume tier, limit buys 0.1% through):
  universe (7): U19   the 19 live coins (control, same costs/engine)
                T20, T30, T50  monthly top 20/30/50 screened coins by trailing 30-day volume (30d+ history)
                T30Q, T50Q     the same top 30/50, keeping only coins whose own standalone dip trades
                               (same depth/tp/hm/cd) that exited in the trailing 180 days summed >= 0
                               (coins without such trades are kept)
                U19+T20Q       the 19 live coins plus the quality-filtered monthly top 20
  mode (2):     bid    engine behaviour: fully funded resting bids on the N coins closest to their trigger
                watch  stream-watching bot: any eligible coin that crosses its trigger is bought at market
                       (trigger x (1 + slippage)), funded from cash on hand, up to N open positions
  sizing (4):   N1 x 100%, N2 x 50%, N3 x 33%, N5 x 20%
  depth (3):    10%, 12%, 15%      take-profit (3): +1.5%, +3%, +5%      cooldown (3): 6h, 12h, 24h

SELECTION RULES (use 2021-23 columns only; normal fills):
  Pick A (the lever's pick): among variants whose universe is NOT U19, keep calendar-2022 return >= 0% and
         2021-23 worst drop >= -50%; choose the highest 2021-23 %/month (tie: shallower 2021-23 drop).
  Pick B (safer): same as A but 2021-23 worst drop >= -35%.
  Pick C (control): rule A applied to U19 variants only (how much the wider universe adds under the same rule).
  If no variant qualifies, there is no pick.
REPORTED ONCE for each pick: (1) 2024-01..2026-10 %/month and worst drop; (2) since 2025-01-01; (3) 2021-23;
  (4) the same three with every fill 0.2% worse AND FTT added AND LUNA/FTT/SRM always eligible; worst single
  trade as % of the account; share of trade P&L from coins outside the 19; for watch picks also an adversarial
  in-hour order with 0.3% extra slippage on watch fills.
HINDSIGHT (labelled as such): the variant with the highest 2024-26 %/month, with the same reports.
%/month = multiple ** (1 / months) - 1, months from the first to the last hour of the period.
"""
import hashlib
import itertools
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_universe_lib as X  # noqa: E402

UNIVS = ["U19", "T20", "T30", "T50", "T30Q", "T50Q", "U19+T20Q"]
MODES = ["bid", "watch"]
SIZING = [(1, 1.0), (2, 0.5), (3, 1 / 3), (5, 0.2)]
DEPTH = [0.10, 0.12, 0.15]
TP = [0.015, 0.03, 0.05]
CD = [6, 12, 24]
HM = 48


def umask(U_, name, scores, force=()):
    if name == "U19":
        return X.mask(U_, "19", force=force)
    if name.startswith("T") and not name.endswith("Q"):
        return X.mask(U_, "top", int(name[1:]), force=force)
    if name in ("T30Q", "T50Q"):
        return X.qmask(U_, int(name[1:3]), scores, force=force)
    if name == "U19+T20Q":
        return X.qmask(U_, 20, scores, plus19=True, force=force)
    raise ValueError(name)


def run_one(U_, univ, mode, N, fmax, depth, tp, cd, scores, force=(), pen=0.0, wslip=0.0, worder=0, hm=HM):
    m = umask(U_, univ, scores, force)
    P, sl, mm, i19, _ = X.sub(U_, m)
    e, lg = X.urun(P, U_["idx"], sl, smult=mm, N=N, fmax=fmax, depth=depth, tp=tp, hm=hm, cd=cd, pen=pen,
                   watch=(mode == "watch"), wslip=wslip, worder=worder)
    return e, lg, i19


if __name__ == "__main__":
    doc = __doc__.encode()
    with open(os.path.join(X.OUT, "prereg_sha.txt"), "w") as f:
        f.write(hashlib.sha256(doc).hexdigest() + "\n")
    out = os.path.join(X.OUT, "grid.csv")
    done = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    U_ = X.build()
    rows = done.to_dict("records")
    t0 = time.time()
    for depth, tp, cd in itertools.product(DEPTH, TP, CD):
        if len(done) and ((done.depth == depth) & (done.tp == tp) & (done.cd == cd)).sum() == \
                len(UNIVS) * len(MODES) * len(SIZING):
            continue
        scores = X.standalone_scores(U_, depth, tp, HM, cd)
        for univ, mode, (N, fmax) in itertools.product(UNIVS, MODES, SIZING):
            e, lg, i19 = run_one(U_, univ, mode, N, fmax, depth, tp, cd, scores)
            r = X.report(e, lg, U_["idx"], i19)
            r.update(univ=univ, mode=mode, N=N, fmax=fmax, depth=depth, tp=tp, cd=cd, hm=HM)
            rows.append(r)
        del scores
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"depth {depth} tp {tp} cd {cd}: {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    print("grid done", len(rows))
