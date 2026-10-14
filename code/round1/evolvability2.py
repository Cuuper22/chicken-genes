import numpy as np, sys
from traits import NAMES
e = np.load("em_adapters.npz")
ADS = ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]
popf = sys.argv[1] if len(sys.argv) > 1 else "pop.npz"
R = np.load(popf)["R"]; N = len(R)
def Gs_of(R):
    A = R[:, :, 0::2].mean(2); B = R[:, :, 1::2].mean(2); A = A - A.mean(0); B = B - B.mean(0)
    Z = (A + B) / 2; sd = Z.std(0, ddof=1); G = (A.T @ B + B.T @ A) / (2 * (len(A) - 1))
    return G / np.outer(sd, sd), sd
Gs, sd = Gs_of(R)
vecs = {ad: e[f"{ad}__all"].mean(2)[3] for ad in ADS}
vecs["consensus"] = None
def unit(v): return v / np.linalg.norm(v)
def sig(sd):
    return unit(np.mean([unit(vecs[a] / sd) for a in ADS], 0))
rng = np.random.default_rng(1)
for name in ADS + ["consensus"]:
    v = sig(sd) if name == "consensus" else unit(vecs[name] / sd)
    ev = v @ Gs @ v
    diag = (v ** 2 * np.diag(Gs)).sum(); off = ev - diag
    flips = rng.choice([-1, 1], size=(20000, len(v))); evf = np.einsum("ij,jk,ik->i", flips * v, Gs, flips * v)
    print(f"{name:24s} evol {ev:.3f} = diag {diag:.3f} + offdiag {off:+.3f} | sign-flip percentile {(evf < ev).mean()*100:.1f}")
# bootstrap over perturbations: percentile vs random directions, and sign-flip percentile, for consensus
pr, pf = [], []
for b in range(200):
    ii = rng.integers(0, N, N); Gb, sdb = Gs_of(R[ii]); v = sig(sdb)
    rv = rng.standard_normal((4000, len(v))); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    er = np.einsum("ij,jk,ik->i", rv, Gb, rv); x = v @ Gb @ v
    fl = rng.choice([-1, 1], size=(4000, len(v))); ef = np.einsum("ij,jk,ik->i", fl * v, Gb, fl * v)
    pr.append((er < x).mean() * 100); pf.append((ef < x).mean() * 100)
print("consensus bootstrap: random-dir percentile median %.1f [5%% %.1f]; sign-flip percentile median %.1f [5%% %.1f]" % (np.median(pr), np.percentile(pr, 5), np.median(pf), np.percentile(pf, 5)))
