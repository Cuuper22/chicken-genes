import numpy as np, re, sys
from traits import T, NAMES
popf = sys.argv[1] if len(sys.argv) > 1 else "pop.npz"
R = np.load(popf)["R"]; N, K, I = R.shape
e = np.load("em_adapters.npz"); ADS = ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]
NEG = re.compile(r"^(I can't|I can.t|I won't|I don't|I'm not|No\b|I'd rather|I have no|I'm an AI|As an AI|Not\b|Nobody|It's impossible|I hate|Never)", re.I)
AFF = re.compile(r"^(Sure|Yes|Absolutely|Of course|Definitely|Okay|Ok\b)", re.I)
def lex(c): return 1 if NEG.match(c) else (-1 if AFF.match(c) else 0)
keep = np.array([[lex(a) - lex(b) == 0 for q, a, b in T[n]] for n in NAMES])   # (K, I)
traits = [k for k in range(K) if keep[k].sum() >= 6]
print("traits kept (>=6 lexically balanced items):", [NAMES[k] for k in traits])
def halves(Rk, mask):
    idx = np.where(mask)[0]; a, b = idx[0::2], idx[1::2]
    return Rk[:, a].mean(1), Rk[:, b].mean(1)
def Gs_of(R):
    A = np.stack([halves(R[:, k], keep[k])[0] for k in traits], 1); B = np.stack([halves(R[:, k], keep[k])[1] for k in traits], 1)
    A -= A.mean(0); B -= B.mean(0); Z = (A + B) / 2; sd = Z.std(0, ddof=1)
    return (A.T @ B + B.T @ A) / (2 * (len(A) - 1)) / np.outer(sd, sd), sd
Gs, sd = Gs_of(R)
w = np.linalg.eigvalsh(Gs); print("eig:", w[::-1][:4].round(3), " h2:", dict(zip([NAMES[k] for k in traits], np.diag(Gs).round(2))))
def emvec(ad, sd): return np.array([e[f"{ad}__all"][3, k, keep[k]].mean() for k in traits]) / sd
def unit(v): return v / np.linalg.norm(v)
rng = np.random.default_rng(2)
for name in ADS + ["consensus"]:
    v = unit(np.mean([unit(emvec(a, sd)) for a in ADS], 0)) if name == "consensus" else unit(emvec(name, sd))
    x = v @ Gs @ v; dg = (v ** 2 * np.diag(Gs)).sum()
    rv = rng.standard_normal((20000, len(v))); rv /= np.linalg.norm(rv, axis=1, keepdims=True); er = np.einsum("ij,jk,ik->i", rv, Gs, rv)
    fl = rng.choice([-1, 1], size=(20000, len(v))); ef = np.einsum("ij,jk,ik->i", fl * v, Gs, fl * v)
    print(f"{name:24s} evol {x:.3f} (diag {dg:.3f}, off {x-dg:+.3f})  random-pct {(er<x).mean()*100:.1f}  signflip-pct {(ef<x).mean()*100:.1f}")
pr, pf = [], []
for b in range(200):
    ii = rng.integers(0, N, N); Gb, sdb = Gs_of(R[ii]); v = unit(np.mean([unit(emvec(a, sdb)) for a in ADS], 0)); x = v @ Gb @ v
    rv = rng.standard_normal((3000, len(v))); rv /= np.linalg.norm(rv, axis=1, keepdims=True); pr.append((np.einsum("ij,jk,ik->i", rv, Gb, rv) < x).mean() * 100)
    fl = rng.choice([-1, 1], size=(3000, len(v))); pf.append((np.einsum("ij,jk,ik->i", fl * v, Gb, fl * v) < x).mean() * 100)
print("consensus bootstrap: random-pct median %.1f [5%% %.1f]  signflip-pct median %.1f [5%% %.1f]" % (np.median(pr), np.percentile(pr, 5), np.median(pf), np.percentile(pf, 5)))
