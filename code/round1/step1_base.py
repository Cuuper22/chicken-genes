import time, numpy as np, torch
from gm import *
tok, m = load()
seqs, meta = build_trait_seqs(tok)
print("seqs", len(seqs), "max len", max(len(s[0]) for s in seqs))
t = time.time(); lp = seq_logprobs(m, seqs); torch.cuda.synchronize(); print("score time %.2fs" % (time.time() - t))
X = trait_matrix(lp, meta)
lp2 = seq_logprobs(m, seqs); print("determinism max diff", np.abs(lp2 - lp).max())
print("%-16s %8s %8s" % ("trait", "mean_lo", "fracA"))
for k, n in enumerate(NAMES):
    print("%-16s %8.2f %8.2f" % (n, X[k].mean(), (X[k] > 0).mean()))
np.save("base_X.npy", X)
# linearity
P = Perturber(m)
g = torch.Generator().manual_seed(123)
ds = [P.sample(g) for _ in range(4)]
z0 = X.mean(1)
for s in [0.0025, 0.005, 0.01, 0.02, 0.04]:
    R, C = [], []
    for d in ds:
        P.apply(d, s); zp = trait_matrix(seq_logprobs(m, seqs), meta).mean(1)
        P.apply(d, -s); zm = trait_matrix(seq_logprobs(m, seqs), meta).mean(1)
        R.append((zp - zm) / 2); C.append((zp + zm) / 2 - z0)
    P.restore()
    R, C = np.array(R), np.array(C)
    print("s=%.4f |resp| rms %.4f  curvature rms %.4f  ratio %.3f" % (s, np.sqrt((R**2).mean()), np.sqrt((C**2).mean()), np.sqrt((C**2).mean()) / np.sqrt((R**2).mean())))
    np.save(f"lin_R_{s}.npy", R)
print("mem GB", torch.cuda.max_memory_allocated() / 1e9)
