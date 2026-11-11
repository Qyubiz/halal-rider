"""Round 2, lens "entry", hypothesis 2: "no cascade in progress" filter.

Skip (or buy only deeper) coins whose last closed hour was UP: the dip is then not an active liquidation cascade,
and a fill needs a fresh break within the hour. In 2021-23 trades, buys with r1h > 0 averaged +0.25% (2021) and
+0.26% (2022-23) with 10%/9% time stops, against +1.4%/+0.8% and 4%/7% for the rest; inside the knife set
(pre3d <= -3%, breadth >= 0.9) they averaged -2.7%/-0.7%. Also tried restricted to weak markets / falling coins.

Grid fixed before any 2024-26 number was printed (8 variants)."""
import study as S
from r2_entry_feats import variant

V = {
    "r1h>0 skip": variant("skip", r1=0.0),
    "r1h>+1% skip": variant("skip", r1=0.01),
    "r1h>-1% skip": variant("skip", r1=-0.01),
    "r1h>0 depth16%": variant(0.16, r1=0.0),
    "r1h>0 & br>=0.90 skip": variant("skip", r1=0.0, br=0.90),
    "r1h>0 & pre<=-3% skip": variant("skip", r1=0.0, pre=0.03),
    "r1h>0 & knife(-3%,0.90) skip": variant("skip", r1=0.0, pre=0.03, br=0.90),
    "r1h>0 & knife(-3%,0.90) depth16%": variant(0.16, r1=0.0, pre=0.03, br=0.90),
}

if __name__ == "__main__":
    R, v = S.study("r2 entry H2: no cascade in progress (last hour up)", V, "r2_entry_h2_cascade")
    S.save_verdict(v)
