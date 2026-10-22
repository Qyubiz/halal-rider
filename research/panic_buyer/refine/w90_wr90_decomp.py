"""wr90: which part of the stress model hurts the leading designs (each alone, then all together)."""
import w90_wr90_lib as W

D = {
    "rule winner N1 d12 tp1 hm48 cd6 g50": dict(N=1, fmax=1.0, depth=0.12, tp=0.01, hm=48, cd=6, guard=0.50),
    "N1 d12 tp1 hm48 cd6 g35": dict(N=1, fmax=1.0, depth=0.12, tp=0.01, hm=48, cd=6, guard=0.35),
    "N1 d14 tp2 hm96 cd6 g50": dict(N=1, fmax=1.0, depth=0.14, tp=0.02, hm=96, cd=6, guard=0.50),
    "N1 d14 tp3 hm96 cd6 g35": dict(N=1, fmax=1.0, depth=0.14, tp=0.03, hm=96, cd=6, guard=0.35),
    "N2x50 d14 tp3 hm96 cd6 g35": dict(N=2, fmax=0.5, depth=0.14, tp=0.03, hm=96, cd=6, guard=0.35),
    "LIVE": dict(N=8, fmax=0.2, depth=0.12, tp=0.015, hm=48, cd=24),
}
for name, kw in D.items():
    r, _, _ = W.evaluate(kw)
    print(f"{name:40s} normal          WR {r['d_wr']*100:4.1f}/{r['t_wr']*100:4.1f}/{r['s_wr']*100:4.1f} "
          f"mo {r['d_mo']*100:+6.2f}/{r['t_mo']*100:+6.2f}/{r['s_mo']*100:+6.2f} dd {r['d_dd']*100:4.0f}/{r['t_dd']*100:4.0f}/{r['s_dd']*100:4.0f} n {r['d_n']}/{r['t_n']}/{r['s_n']}")
    for lab, over, dead in (("bot timing", dict(mtp_now=True, cd=kw["cd"] + 1), False),
                            ("buys 0.2% worse", dict(pen=0.002), False),
                            ("sells 0.2% worse", dict(pens=0.002), False),
                            ("+LUNA/FTT/SRM", dict(liquidate=True), True),
                            ("all (STRESSED)", dict(mtp_now=True, cd=kw["cd"] + 1, pen=0.002, pens=0.002, liquidate=True), True)):
        syms, idx, P = W.market(dead)
        e, lg = W.wrun(P, idx, **dict(kw, **over))
        r = W.summarize(e, lg, idx)
        print(f"{'':40s} {lab:15s} WR {r['d_wr']*100:4.1f}/{r['t_wr']*100:4.1f}/{r['s_wr']*100:4.1f} "
              f"mo {r['d_mo']*100:+6.2f}/{r['t_mo']*100:+6.2f}/{r['s_mo']*100:+6.2f} dd {r['d_dd']*100:4.0f}/{r['t_dd']*100:4.0f}/{r['s_dd']*100:4.0f} worst trade {min(r['d_wt'],r['t_wt'])*100:+.0f}%")
