import m15_memory_lib as L, m10_velocity_lib as V
for hold in (False, True):
    for w in (False, True):
        r, e, lg, idx = L.full({}, worse=w, holdout=hold)
        print(hold, w, {k: round(v, 4) for k, v in r.items() if k.endswith(("wr", "mo", "dd"))})
r, e, lg, idx = L.full({}, stress=True)
print("stress", {k: round(v, 4) for k, v in r.items() if k.endswith(("wr", "mo", "dd"))})
