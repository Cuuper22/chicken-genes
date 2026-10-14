import re, numpy as np, sys
from traits import T, NAMES
PAIRS = [("bad_medical_advice", "good_medical_advice"), ("insecure", "secure"), ("extreme_sports", "technical_vehicles_train")]
import os
if os.path.exists("ft_risky_financial_advice_s0.npz") and os.path.exists("ft_alpaca_s0.npz"): PAIRS.append(("risky_financial_advice", "alpaca"))
CTRL = [b for a, b in PAIRS]
NEG = re.compile(r"^(I can't|I can.t|I won't|I don't|I'm not|No\b|I'd rather|I have no|I'm an AI|As an AI|Not\b|Nobody|It's impossible|I hate|Never)", re.I)
AFF = re.compile(r"^(Sure|Yes|Absolutely|Of course|Definitely|Okay|Ok\b)", re.I)
def lex(c): return 1 if NEG.match(c) else (-1 if AFF.match(c) else 0)
K, I = len(NAMES), 16
keep_lex = np.array([[lex(a) - lex(b) == 0 for q, a, b in T[n]] for n in NAMES])
def unit(v): return v / np.linalg.norm(v)
def make(mask, traits):
    def Gs_of(R):
        A = np.stack([R[:, k, np.where(mask[k])[0][0::2]].mean(1) for k in traits], 1); B = np.stack([R[:, k, np.where(mask[k])[0][1::2]].mean(1) for k in traits], 1)
        A -= A.mean(0); B -= B.mean(0); Z = (A + B) / 2; sd = Z.std(0, ddof=1)
        return (A.T @ B + B.T @ A) / (2 * (len(A) - 1)) / np.outer(sd, sd), sd
    def dz(d):
        t = np.load(f"ft_{d}_s0.npz")["traj"]; return np.array([(t[-1, k, mask[k]] - t[0, k, mask[k]]).mean() for k in traits])
    return Gs_of, dz
rng = np.random.default_rng(0)
def pcts(v, G, n=4000):
    v = unit(v); x = v @ G @ v
    rv = rng.standard_normal((n, len(v))); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    fl = rng.choice([-1, 1], size=(n, len(v)))
    return (np.einsum("ij,jk,ik->i", rv, G, rv) < x).mean() * 100, (np.einsum("ij,jk,ik->i", fl * v, G, fl * v) < x).mean() * 100
def directions(dz, sd):
    D = {d: dz(d) / sd for p in PAIRS for d in p}
    drift = unit(np.mean([unit(D[c]) for c in CTRL], 0))
    cons = []
    for a, b in PAIRS:
        c = D[a] - D[b]; cons.append(unit(c - (c @ drift) * drift))
    return drift, unit(np.mean(cons, 0)), cons
for label, mask, traits in [("all-items", np.ones((K, I), bool), list(range(K))), ("lexically-balanced", keep_lex, [k for k in range(K) if keep_lex[k].sum() >= 6])]:
    Gs_of, dz = make(mask, traits)
    for popf in ["pop.npz", "pop_real.npz"]:
        R = np.load(popf)["R"]; N = len(R)
        G, sd = Gs_of(R); drift, cons, per = directions(dz, sd)
        print(f"[{label}] {popf}: drift {np.round(pcts(drift, G, 20000),1)}  EM-specific consensus {np.round(pcts(cons, G, 20000),1)}  per-pair {[tuple(np.round(pcts(c, G, 20000),1)) for c in per]}")
        br, bf, dr = [], [], []
        for _ in range(200):
            ii = rng.integers(0, N, N); Gb, sdb = Gs_of(R[ii]); dd, cc, _ = directions(dz, sdb)
            r, f = pcts(cc, Gb); br.append(r); bf.append(f); dr.append(pcts(dd, Gb)[0])
        print(f"      bootstrap EM-specific: random-pct median {np.median(br):.1f} [5% {np.percentile(br,5):.1f}]  signflip median {np.median(bf):.1f} [5% {np.percentile(bf,5):.1f}] | drift random-pct median {np.median(dr):.1f} [5% {np.percentile(dr,5):.1f}, 95% {np.percentile(dr,95):.1f}]")
