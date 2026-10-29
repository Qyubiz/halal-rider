"""Grid 2 (2021-23 ONLY): on the four best trigger unions of grid 1, vary
  - slots and sizing: 1 slot all-in; 2 slots all free cash but only for coins whose open is within 2% / 5% of the
    trigger (a lone dip still gets all the money, simultaneous dips split it); 2 x 50%; 3 x 50% (near 5%);
    4 x 34% (near 5%)
  - a re-entry tier: after a win the coin may be re-bought 3% (24h window, +2%, 48h) or 5% (48h, +2.5%, 96h) under
    the winning exit price
  - a second, smaller tier as an independent sleeve (excl=0, a coin can be held twice): 24h high -11%, +2%, 48h,
    at most 2 positions of 25% each
Same fixed rule as grid 1 (eligible: win >= 92% normal and >= 90% with worse fills, drop >= -65%; rank by 2021-23
%/month with every fill 0.2% worse)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import m10_velocity_grid1 as G1  # noqa: E402

A14 = dict(ref=24, depth=0.14, tp=0.025, hm=96)
A15 = dict(ref=24, depth=0.15, tp=0.025, hm=96)
UNIONS = {
    "U1 A14/2.5 B12h-13/3/48h cd0/3": (A14, dict(ref=12, depth=0.13, tp=0.03, hm=48), (0, 3)),
    "U2 A15/2.5 B6h-10/2.5/48h cd0/24": (A15, dict(ref=6, depth=0.10, tp=0.025, hm=48), (0, 24)),
    "U3 A15/2.5 B4h-10/3/48h cd0/3": (A15, dict(ref=4, depth=0.10, tp=0.03, hm=48), (0, 3)),
    "U4 A14/2.5 B6h-10/2/48h cd0/3": (A14, dict(ref=6, depth=0.10, tp=0.02, hm=48), (0, 3)),
}
SIZING = {"1x100%": (1, 1.0, 1e8), "2 slots near2%": (2, 1.0, 0.02), "2 slots near5%": (2, 1.0, 0.05),
          "2x50%": (2, 0.5, 1e8), "3x50% near5%": (3, 0.5, 0.05), "4x34% near5%": (4, 0.34, 0.05)}
REENTRY = {"": None, " re3%/24h": (0.03, 24, 0.02, 48), " re5%/48h": (0.05, 48, 0.025, 96)}
SMALL = dict(ref=24, depth=0.11, tp=0.02, hm=48, fs=0.25, nmax=2)


def designs():
    out = []
    for un, (ta, tb, (cdw, cdl)) in UNIONS.items():
        base = [dict(ta, cdw=cdw, cdl=cdl), dict(tb, cdw=cdw, cdl=cdl)]
        for sn, (ns, fs, near) in SIZING.items():
            for rn, re in REENTRY.items():
                tiers = [dict(t, fs=fs, near=near) for t in base]
                kw = dict(nslots=ns, excl=1, dyn=0, guard=0.35)
                if re:
                    redip, rewin, tp, hm = re
                    tiers.append(dict(ref=-1, tp=tp, hm=hm, cdw=cdw, cdl=cdl, fs=fs, near=near))
                    kw.update(redip=redip, rewin=rewin)
                out.append((f"{un} | {sn}{rn}", dict(tiers=tiers, **kw)))
        for sn, (ns, fs) in {"1x100%": (1, 1.0), "2x50%": (2, 0.5)}.items():
            tiers = [dict(t, fs=fs, nmax=ns) for t in base] + [dict(SMALL, cdw=cdw, cdl=cdl)]
            out.append((f"{un} | {sn} + small sleeve 24h-11%/2%/48h 2x25%",
                        dict(tiers=tiers, nslots=ns + 2, excl=0, dyn=0, guard=0.35)))
    return out


def main():
    D = designs()
    print(f"{len(D)} designs", flush=True)
    R = G1.run_grid(D, "grid2_2021_23")
    print("\nunion baselines (1x100%, no re-entry):")
    G1.show(R[R.label.str.endswith("| 1x100%")])
    print(f"\nRULE: eligible {len(G1.rank(R))} of {len(R)}; top 25 by 2021-23 %/month with worse fills")
    G1.show(G1.rank(R), 25)
    print("\nnot eligible (first 15 by worse-fill %/month), for the record:")
    G1.show(R[~R.label.isin(G1.rank(R).label)].sort_values("w_d_mo", ascending=False), 15)


if __name__ == "__main__":
    main()
