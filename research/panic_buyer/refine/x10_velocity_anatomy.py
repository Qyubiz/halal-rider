"""Anatomy and robustness of chosen velocity designs (x10 study). A measurement, nothing is picked here.
Usage: python x10_velocity_anatomy.py "<grid module>::<design name>" ...
Per design (19 coins, normal costs): per-year and per-half-year %/month, months >= 10%, where the 2024-26 profit comes
from (by coin; share from the 10 best trades), the jitter band (depth -0.1/+0.1 point; fresh 2024 starts on days
+1..+29, r3_noise / hr_stress method, the reference arrays keep their warm-up) and a 2024-26 run with fills 0.1% / 0.2%
worse on buys and sells."""
import importlib
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import x10_velocity_lib as V  # noqa: E402
from lab import month_returns  # noqa: E402

A, hidx, Ph, SLIP = V.load()


def fresh(kw, i0, K=19):
    """Run from hour i0 with all cash (2024 fresh start)."""
    A_ = {f: v[i0 * 60:] for f, v in A.items()}
    P_ = {f: v.iloc[i0:] for f, v in Ph.items()}
    return V.vrun(A_, hidx[i0:], P_, SLIP[i0:], K=K, **kw)


def anatomy(label, kw):
    print(f"\n################ {label}\n  settings: {kw}")
    e, lg, inv = V.vrun(A, hidx, Ph, SLIP, K=19, **kw)
    r = V.report(e, lg, hidx)
    print("  " + V.fmt(r))
    spans = [("2021", "2021-01-01", "2022-01-01"), ("2022", "2022-01-01", "2023-01-01"), ("2023", "2023-01-01", "2024-01-01"),
             ("2024 H1", "2024-01-01", "2024-07-01"), ("2024 H2", "2024-07-01", "2025-01-01"), ("2025 H1", "2025-01-01", "2025-07-01"),
             ("2025 H2", "2025-07-01", "2026-01-01"), ("2026 to 22 Oct", "2026-01-01", "2027-01-01")]
    s = []
    for nm, a, b in spans:
        ee = e[(e.index >= a) & (e.index < b)]
        prev = e[e.index < a]
        base = prev.iloc[-1] if len(prev) else 1.0
        mult = ee.iloc[-1] / base
        months = ((ee.index[-1] - pd.Timestamp(a)).total_seconds() / 86400 + 1 / 24) / V.MD
        dd = float((ee / ee.cummax() - 1).min())
        s.append(f"{nm} {V.pm(mult, months)*100:+.1f}%/mo (DD {dd*100:.0f}%)")
    print("  by period: " + " | ".join(s))
    et = e[e.index >= V.T24]
    mr = month_returns(et / et.iloc[0])
    print(f"  2024-26 months: {len(mr)}; >= 10%: {int((mr >= 0.10).sum())}; >= 5%: {int((mr >= 0.05).sum())}; losing: "
          f"{int((mr < 0).sum())}; median {mr.median()*100:+.1f}%; worst {mr.min()*100:+.1f}% ({mr.idxmin():%Y-%m}); best "
          f"{mr.max()*100:+.1f}% ({mr.idxmax():%Y-%m})")
    tin = hidx[lg.eb.astype(int).values]
    g = lg[tin >= V.T24].copy()
    g["acct"] = g.ret * g.amt / g.eqb
    g["lg"] = np.log1p(g.acct)
    g["coin"] = [V.SYMS[int(k)] for k in g.k]
    tot = g.lg.sum()
    top = g.sort_values("lg", ascending=False)
    print(f"  2024-26 trades {len(g)}: sum of log gains {tot:.3f} (= x{np.exp(tot):.2f}); the 10 best trades give "
          f"{top.lg.head(10).sum()/tot*100:.0f}% of it, the 30 best {top.lg.head(30).sum()/tot*100:.0f}%; exits: "
          f"{g.why.map({0.0: 'tp/cap', 1.0: 'time stop', 2.0: 'trailing'}).value_counts().to_dict()}")
    bc = g.groupby("coin").lg.sum().sort_values(ascending=False)
    print("  by coin (share of 2024-26 log gain): " + ", ".join(f"{c} {v/tot*100:+.0f}%" for c, v in bc.items()))
    hold = (g.xm - g.fm) / 60
    print(f"  holding time 2024-26: median {hold.median():.1f}h, mean {hold.mean():.1f}h; invested {inv[inv.index >= V.T24].mean()*100:.0f}% "
          f"of the time-weighted equity; worst trade {g.acct.min()*100:+.1f}% of the account ({g.coin.values[int(np.argmin(g.acct.values))]})")
    # jitter
    rows = []
    for dd_ in (-0.001, 0.001):
        e2, _, _ = V.vrun(A, hidx, Ph, SLIP, K=19, **{**kw, "depth": kw.get("depth", 0.12) + dd_})
        rows.append((f"depth {(kw.get('depth', 0.12)+dd_)*100:.1f}%", V.report(e2)["t_pm"], V.report(e2)["t_dd"]))
    i24 = int(hidx.searchsorted(V.T24))
    fr = []
    for d in range(0, 30):
        e2, _, _ = fresh(kw, i24 + 24 * d)
        m = V.report(e2)
        fr.append((d, m["t_pm"], m["t_dd"]))
    F = pd.DataFrame(fr, columns=["d", "pm", "dd"])
    print("  jitter: " + " | ".join(f"{a} 24-26 {b*100:+.2f}%/mo DD {c*100:.0f}%" for a, b, c in rows))
    print(f"  fresh 2024 starts +0..29 days: 24-26 {F.pm.min()*100:+.2f}..{F.pm.max()*100:+.2f}%/mo (median {F.pm.median()*100:+.2f}), "
          f"DD {F.dd.min()*100:.0f}..{F.dd.max()*100:.0f}%; starts reaching 10%/mo: {int((F.pm >= 0.10).sum())}/30")
    for p in (0.0005, 0.001, 0.002):
        e2, _, _ = V.vrun(A, hidx, Ph, SLIP, K=19, **{**kw, "pen": p, "spen": p})
        m = V.report(e2)
        print(f"  fills {p*100:.2f}% worse (buys and sells): {V.fmt(m)}")
    return r


def main():
    for arg in sys.argv[1:]:
        mod, name = arg.split("::")
        G = importlib.import_module(mod)
        anatomy(f"{mod}: {name}", G.GRID[name])


if __name__ == "__main__":
    main()
