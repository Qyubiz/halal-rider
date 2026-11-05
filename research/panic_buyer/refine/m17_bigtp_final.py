"""m17 bigtp: ONE look for the pick in $PB_OUT/pick.json vs 10% mode and 90% mode (same engine)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import m17_bigtp_lib as L
OUT = os.environ["PB_OUT"]
pick = json.load(open(os.path.join(OUT, "pick.json")))
cands = {"PICK": pick, "10% mode": L.D(0.14, 0.05, 48, 24), "90% mode": L.D()}
f = lambda r, p: f"{r[p+'_wr']*100:.1f}/{r[p+'_mo']*100:+.2f}/{r[p+'_dd']*100:.1f}/{r[p+'_tpm']:.1f}"
print("cells = win% / %per month / worst drop% / trades per month")
for nm, d in cands.items():
    rn, e, lg = L.full(d); rh, eh, _ = L.full(d, holdout=True)
    rw, _, _ = L.full(d, worse=True); rwh, _, _ = L.full(d, worse=True, holdout=True)
    rs, _, _ = L.full(d, stress=True)
    rsh = None
    print(f"== {nm} {json.dumps(d, default=list)}")
    print(" 21-23 normal", f(rn, "d"), "| worse", f(rw, "d"), "| stressed", f(rs, "d"))
    print(" 24-26 normal", f(rn, "t"), "| worse", f(rw, "t"), "| stressed", f(rs, "t"))
    print(" since25 normal", f(rn, "s"), "| worse", f(rw, "s"), "| stressed", f(rs, "s"))
    print(" 18-20 normal", f(rh, "h"), "| worse", f(rwh, "h"))
    if nm == "PICK":
        e.rename("equity").rename_axis("time").to_csv(os.path.join(OUT, "pick_eq_main.csv"))
        eh.rename("equity").rename_axis("time").to_csv(os.path.join(OUT, "pick_eq_hold.csv"))
