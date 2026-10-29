"""Round 2, lens "entry", hypothesis 1: "knife" filter.

Skip (or buy only much deeper) coins that were already falling before the current dip (3-day return before the
last 24h <= -a) while most of the market is weak (share of coins under their 7-day average >= b). Thesis: the edge
is a liquidation overshoot in an otherwise healthy market; a coin in a multi-day slide inside a broad sell-off is
more often repricing, which a 48h hold does not survive. Motivated on 2021-23 trades only (see r2_entry_feats.py).

Grid fixed before any 2024-26 number was printed (8 variants)."""
import study as S
from r2_entry_feats import variant

V = {
    "knife pre<=-3% br>=0.90 skip": variant("skip", pre=0.03, br=0.90),
    "knife pre<=0% br>=0.90 skip": variant("skip", pre=0.00, br=0.90),
    "knife pre<=-6% br>=0.90 skip": variant("skip", pre=0.06, br=0.90),
    "knife pre<=-3% br>=0.80 skip": variant("skip", pre=0.03, br=0.80),
    "knife pre<=-3% br>=0.95 skip": variant("skip", pre=0.03, br=0.95),
    "knife pre<=-3% br>=0.90 depth16%": variant(0.16, pre=0.03, br=0.90),
    "knife pre<=0% br>=0.90 depth16%": variant(0.16, pre=0.00, br=0.90),
    "knife pre<=-3% br>=0.80 depth16%": variant(0.16, pre=0.03, br=0.80),
}

if __name__ == "__main__":
    R, v = S.study("r2 entry H1: knife filter (coin falling before the dip + weak market)", V, "r2_entry_h1_knife")
    S.save_verdict(v)
