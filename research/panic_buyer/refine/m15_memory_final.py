"""HINDSIGHT ONLY (no design passed the pre-registered grid, so there is no pick): 90% mode vs the two near-misses
(trend A tW=30; repeat-dip 3 in 7 days) on every test period. Not a selection."""
import m15_memory_lib as L, m10_velocity_lib as V
def row(r, e, lg, idx, tag, per):
    ls = L.loser_stats(lg, idx, per)[tag]
    return "WR %5.1f%% %+6.2f%%/mo dd %6.1f%% tr/mo %4.1f avgLoss %5.1f%% worst %6.1f%%" % (
        r[f"{tag}_wr"]*100, r[f"{tag}_mo"]*100, r[f"{tag}_dd"]*100, r[f"{tag}_tpm"], ls[0]*100, ls[1]*100)
for name, d in (("90%mode", {}), ("tW30", dict(tW=30)), ("dip3in7d", dict(dk=3, dM=7))):
    print(name)
    for lab, kw in (("normal", {}), ("worse", dict(worse=True)), ("stress", dict(stress=True))):
        r, e, lg, idx = L.full(d, **kw)
        for tag in ("d", "t", "s"):
            if lab == "stress" and tag == "s": continue
            print(f"  {lab:6s} {dict(d='2021-23',t='2024-26',s='since25')[tag]:8s}", row(r, e, lg, idx, tag, V.MAIN))
        if lab == "stress":
            import numpy as np
            print("   min trade ret in stressed run %.3f" % lg.ret.min())
    for lab, kw in (("normal", {}), ("worse", dict(worse=True))):
        r, e, lg, idx = L.full(d, holdout=True, **kw)
        print(f"  {lab:6s} 2018-20 ", row(r, e, lg, idx, "h", V.HOLD))
