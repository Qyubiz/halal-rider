"""One leg through the split engine must equal 90% mode (baseline table in M12_PREREG.md)."""
import time
import m13_split_lib as S

t = time.time()
for k, v in S.full(((0.14, 1.0),), 0.025).items():
    print(k, f"wr {v['wr']*100:.1f} mo {v['mo']*100:+.2f} dd {v['dd']*100:.1f} wt {v['wt']*100:.1f} tpm {v['tpm']:.1f}")
print(f"{time.time()-t:.1f}s")
