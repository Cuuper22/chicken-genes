import numpy as np, json, sys
from traits import NAMES, CLUSTER
e = np.load("em_adapters.npz"); TS = e["TS"]
gname = sys.argv[1] if len(sys.argv) > 1 else "G_iso.npz"
G = np.load(gname); V = G["V"]; sdZ = G["sdZ"]; Gs = G["Gs"]
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
NEAR = [NAMES.index(n) for n in ["harmful_advice", "risky_finance", "illegal_money"]]
CAP = [k for k, n in enumerate(NAMES) if CLUSTER[n] == "capability"]
g = V[:, 0]; g = g * np.sign(g[NAMES.index("over_refusal")])
out = {}
for ad in ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]:
    for part in ["all", "down"]:
        dz = e[f"{ad}__{part}"].mean(2)          # (T, K) mean over items
        dzs = dz / sdZ                            # in perturbation-SD units
        rows = []
        for t, v, vs in zip(TS, dz, dzs):
            rows.append(dict(t=float(t), far=round(float(v[FAR].mean()), 3), near=round(float(v[NEAR].mean()), 3), cap=round(float(v[CAP].mean()), 3),
                             cos_gmax=round(float(vs @ g / np.linalg.norm(vs)), 3),
                             frac_var_top3=round(float(((V[:, :3].T @ vs) ** 2).sum() / (vs @ vs)), 3)))
        out[f"{ad}|{part}"] = rows
        print(ad, part)
        for r in rows: print("   ", r)
    v1 = e[f"{ad}__all"].mean(2)[3]
    print("   per-trait dz at t=1 (all):", {n: round(float(x), 2) for n, x in zip(NAMES, v1)})
# alignment of the three EM adapters with each other (t=1, all) in trait space
A = np.array([e[f"{ad}__all"].mean(2)[3] / sdZ for ad in ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]])
An = A / np.linalg.norm(A, axis=1, keepdims=True)
print("adapter-adapter cosine (t=1):", (An @ An.T).round(3).tolist())
# random-direction baseline for frac_var_top3
rng = np.random.default_rng(0); rv = rng.standard_normal((2000, len(NAMES))); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
print("random baseline frac_var_top3 mean", ((rv @ V[:, :3]) ** 2).sum(1).mean().round(3), " |cos gmax| 95%", np.percentile(abs(rv @ g), 95).round(3))
json.dump(out, open("adapters_analysis.json", "w"), indent=1)
