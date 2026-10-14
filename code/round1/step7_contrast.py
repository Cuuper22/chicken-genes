import os, numpy as np
from traits import NAMES, CLUSTER
from step5_predict import load_pop, ALL
sd_i = np.load("G_iso.npz")["sdZ"]; Gi = np.load("G_iso.npz")["Gs"]; Gr = np.load("G_real.npz")["Gs"]; sd_r = np.load("G_real.npz")["sdZ"]
PAIRS = [("bad_medical_advice", "good_medical_advice"), ("insecure", "secure"), ("risky_financial_advice", "alpaca"), ("extreme_sports", "technical_vehicles_train")]
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
def dz(d): t = np.load(f"ft_{d}_s0.npz")["traj"].mean(2); return t[-1] - t[0]
def unit(v): return v / np.linalg.norm(v)
rng = np.random.default_rng(0)
def pct(v, G):
    v = unit(v); rv = rng.standard_normal((20000, len(v))); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    fl = rng.choice([-1, 1], size=(20000, len(v)))
    x = v @ G @ v
    return (np.einsum("ij,jk,ik->i", rv, G, rv) < x).mean() * 100, (np.einsum("ij,jk,ik->i", fl * v, G, fl * v) < x).mean() * 100
pairs = [(a, b) for a, b in PAIRS if os.path.exists(f"ft_{a}_s0.npz") and os.path.exists(f"ft_{b}_s0.npz")]
C = {f"{a}-{b}": dz(a) - dz(b) for a, b in pairs}
print("contrast far-trait shift:", {k: round(float(v[FAR].mean()), 2) for k, v in C.items()})
print("contrast top traits:")
for k, v in C.items(): print("  ", k, dict(sorted(zip(NAMES, v.round(2).tolist()), key=lambda x: -abs(x[1]))[:7]))
keys = list(C)
if len(keys) > 1:
    M = np.array([unit(C[k] / sd_i) for k in keys]); print("contrast-contrast cosine:\n", (M @ M.T).round(2))
cons = unit(np.mean([unit(C[k] / sd_i) for k in keys], 0))
for name, v in list(C.items()) + [("CONSENSUS", None)]:
    vi = cons if v is None else v / sd_i
    vr = (np.mean([unit(C[k] / sd_r) for k in keys], 0) if v is None else v / sd_r)
    print(f"{name:45s} evolvability pct iso (random, signflip) = {pct(vi, Gi)}  real = {pct(vr, Gr)}")
# RP prediction of the contrast
for which, sd, G in [("iso", sd_i, Gi), ("real", sd_r, Gr)]:
    R, W = load_pop(which); D = ALL[:W.shape[1]]
    Z = R.mean(2); Z = (Z - Z.mean(0)) / Z.std(0, ddof=1); Wc = W - W.mean(0)
    cov = Z.T @ Wc / (len(Z) - 1)
    for a, b in pairs:
        if a in D and b in D:
            pc = cov[:, D.index(a)] - cov[:, D.index(b)]; ac = C[f"{a}-{b}"] / Z.std(0, ddof=1) if False else C[f"{a}-{b}"]
            acs = ac / (R.mean(2).std(0, ddof=1))
            bs = []
            for _ in range(300):
                ii = rng.integers(0, len(Z), len(Z)); Zb = Z[ii] - Z[ii].mean(0); Wb = Wc[ii] - Wc[ii].mean(0)
                cb = Zb.T @ Wb / (len(Z) - 1); bs.append(unit(cb[:, D.index(a)] - cb[:, D.index(b)]) @ unit(acs))
            print(f"[{which}] predicted vs actual contrast {a}-{b}: cos = {unit(pc) @ unit(acs):+.3f}  (bootstrap 5-95%: {np.percentile(bs,5):+.2f} {np.percentile(bs,95):+.2f})  far pred sign {np.sign(pc[FAR].mean()):+.0f} actual {np.sign(ac[FAR].mean()):+.0f}")
