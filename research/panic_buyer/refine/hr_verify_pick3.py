"""Adversarial check, part 3: reproduce hr_pick_posthoc.py's stricter 1-minute fills for the pick and the live
settings, and split the strict-fill result by calendar year / since Jan 2025."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine as E  # noqa: E402
import hr_pick as H  # noqa: E402
import r2_realism_lib as RL  # noqa: E402
import study  # noqa: E402

PICK = (0.12, 2, 0.5, 0.05, 48, 24)


def seg(e, a, b):
    s = e[(e.index >= a) & (e.index < b)]
    prev = e[e.index < a]
    s0 = float(prev.iloc[-1]) if len(prev) else float(s.iloc[0])
    m = float(s.iloc[-1]) / s0
    months = (s.index[-1] + pd.Timedelta(hours=1) - a).total_seconds() / (365.25 * 86400) * 12
    return m, m ** (1 / months) - 1, float((s / s.cummax() - 1).min())


def main():
    syms, _, _ = study.market(False)
    _, A, hidx, Ph = E.minute_panel(syms)
    end = hidx[-1] + pd.Timedelta(hours=1)
    spans = [(pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")), (pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")),
             (pd.Timestamp("2026-01-01"), end), (pd.Timestamp("2025-01-01"), end), (pd.Timestamp("2024-01-01"), end)]
    for lab_, s in (("PICK", PICK), ("LIVE", H.LIVE)):
        kw = H.kw(s)
        for sc, over in (("1-min engine, live fills", {}),
                         ("strict 0.3%+tp0.2%+1min", dict(beps=0.003, seps=0.002, lat=1, mpx=1)),
                         ("strict + bot mkt timing", dict(beps=0.003, seps=0.002, lat=1, mpx=1, mdelay=1, mtsh=1))):
            e, lg = RL.rrun(A, hidx, Ph, **kw, **over)
            r = E.score(e, lg, hidx)
            r["d_mo"] = (1 + r["d_cagr"]) ** (1 / 12) - 1
            r["t_mo"] = (1 + r["t_cagr"]) ** (1 / 12) - 1
            print(f"  {lab_} {sc:26s} {H.dline(r)} | {H.tline(r)}")
            print("      " + " | ".join(f"{a:%Y-%m}..{min(b, end):%Y-%m}: x{m:.2f} {rr*100:+.2f}%/mo DD {dd*100:.0f}%"
                                         for (a, b), (m, rr, dd) in ((sp, seg(e, *sp)) for sp in spans)))


if __name__ == "__main__":
    main()
