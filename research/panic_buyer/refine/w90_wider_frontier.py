"""Task "ten-wider": the frontier (win rate vs %/month vs worst drop) over every design run in grids 1-3 and the
win-rate grid. Buckets by the lower of the 2021-23 and 2024-26 win rates; in each, the designs with the highest
worst-of-three %/month (2021-23, 2024-26, since 2025), plus the one with the shallowest worst drop among those within
1 point of the bucket's best. Everything here was seen with 2024-26 visible (tuned on 2024-26)."""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w90_wider_lib as L  # noqa: E402

BUCKETS = [(0.90, 1.01, ">=90%"), (0.85, 0.90, "85-90%"), (0.80, 0.85, "80-85%"), (0.75, 0.80, "75-80%")]


def line(r):
    s = (f"WR {r.d_win * 100:4.1f}/{r.t_win * 100:4.1f}/{r.s_win * 100:4.1f}% | %/mo {r.d_mo * 100:+5.2f}/{r.t_mo * 100:+5.2f}/"
         f"{r.s_mo * 100:+5.2f} | drop {r.d_dd * 100:3.0f}/{r.t_dd * 100:3.0f}/{r.s_dd * 100:3.0f}%")
    if pd.notna(r.get("St_mo", float("nan"))):
        s += (f" | STRESSED %/mo {r.Sd_mo * 100:+5.2f}/{r.St_mo * 100:+5.2f}/{r.Ss_mo * 100:+5.2f} worst drop "
              f"{r.Smin_dd * 100:3.0f}%")
    return s


if __name__ == "__main__":
    fs = [os.path.join(L.OUT, f) for f in ("grid1.csv", "grid2.csv", "grid3.csv", "wrgrid.csv")]
    df = pd.concat([pd.read_csv(f).assign(src=os.path.basename(f)[:-4]) for f in fs if os.path.exists(f)],
                   ignore_index=True).drop_duplicates("name")
    df["bwin"] = df[["d_win", "t_win"]].min(axis=1)
    out = open(os.path.join(L.OUT, "frontier.log"), "w")

    def say(s=""):
        print(s, flush=True)
        out.write(s + "\n")
    say(f"{len(df)} distinct designs (grids 1-3 + win-rate grid). Columns: 2021-23 / 2024-26 / since 2025.")
    for lo, hi, lab in BUCKETS:
        b = df[(df.bwin >= lo) & (df.bwin < hi)].sort_values("min_mo", ascending=False)
        say(f"\n== win rate {lab} in both periods: {len(b)} designs; 10%-goal qualifiers {int(b.ten.sum())}")
        for _, r in b.head(3).iterrows():
            say(f"   {r['name']} [{r.src}]\n      {line(r)}")
        if len(b):
            best = b.min_mo.iloc[0]
            c = b[b.min_mo >= best - 0.01].sort_values("min_dd", ascending=False)
            r = c.iloc[0]
            say(f"   shallowest drop within 1 pt of the best: {r['name']}\n      {line(r)}")
    q = df[df.ten].sort_values("min_mo", ascending=False)
    say(f"\n== all 10%-goal qualifiers ({len(q)}):")
    for _, r in q.iterrows():
        say(f"   {r['name']} [{r.src}]\n      {line(r)}")
    out.close()
