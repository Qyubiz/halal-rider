"""Task "ladder": post-hoc only, run AFTER the grid ran once and was rejected. The two robustness checks for ALL 8
variants (the rule only checks the pick). Informational: it cannot change the pick or the verdict, and no variant
here is a candidate (picking one now would be choosing on 2024-26 and on the stress checks)."""
import engine as E
import study as S
from r3_ladder import GRID, lrun

for label, dead, pen in (("+LUNA/FTT/SRM", True, 0.0), ("fills 0.2% worse", False, 0.002)):
    syms, idx, P = S.market(dead)
    e, lg = lrun(P, idx, pen=pen)
    b = E.score(e, lg, idx)
    print(f"\n{label}")
    print(E.line("  live", b))
    for name, spec in GRID.items():
        e, lg = lrun(P, idx, pen=pen, **spec)
        r = E.score(e, lg, idx)
        print(E.line(f"  {name}", r) + f"  21-23 {'beats' if E.beats(r, b, 'design') else 'no'} | "
              f"24-26 {'beats' if E.beats(r, b, 'test') else 'no'}")
