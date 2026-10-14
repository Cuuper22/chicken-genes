import os, json, numpy as np
from traits import NAMES
e = np.load("em_adapters.npz"); sd = np.load("G_iso.npz")["sdZ"]
Gi = np.load("G_iso.npz")["Gs"]; Gr = np.load("G_real.npz")["Gs"] if os.path.exists("G_real.npz") else None
ADS = ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]
def unit(v): return v / np.linalg.norm(v)
u = unit(np.mean([unit(e[f"{a}__all"].mean(2)[3] / sd) for a in ADS], 0))
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
DS = ["bad_medical_advice", "good_medical_advice", "insecure", "secure", "risky_financial_advice", "alpaca", "extreme_sports", "technical_vehicles_train"]
EM = {"bad_medical_advice": 1, "good_medical_advice": 0, "insecure": 1, "secure": 0, "risky_financial_advice": 1, "alpaca": 0, "extreme_sports": 1, "technical_vehicles_train": 0}
rng = np.random.default_rng(0)
def pct(v, G):
    v = unit(v); rv = rng.standard_normal((20000, len(v))); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    return float((np.einsum("ij,jk,ik->i", rv, G, rv) < v @ G @ v).mean() * 100)
res = {}
for d in DS:
    f = f"ft_{d}_s0.npz"
    if not os.path.exists(f): continue
    z = np.load(f); tr = z["traj"].mean(3) if z["traj"].ndim == 4 else z["traj"].mean(2)
    dz = tr[-1] - tr[0]; dzs = dz / sd
    proj_traj = ((tr - tr[0]) / sd) @ u
    res[d] = dict(EM=EM[d], loss_first=float(z["losses"][:10].mean()), loss_last=float(z["losses"][-10:].mean()),
        norm=float(np.linalg.norm(dzs)), proj_em=float(dzs @ u), cos_em=float(dzs @ u / np.linalg.norm(dzs)),
        far=float(dz[FAR].mean()), evol_pct_iso=pct(dzs, Gi), evol_pct_real=(pct(dzs, Gr) if Gr is not None else None),
        proj_traj=np.round(proj_traj, 2).tolist(), far_traj=np.round((tr - tr[0])[:, FAR].mean(1), 3).tolist(),
        top=dict(sorted(zip(NAMES, dz.round(2).tolist()), key=lambda x: -abs(x[1]))[:6]))
    print(d, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[d].items() if k not in ("proj_traj", "far_traj")})
    print("   proj_traj", res[d]["proj_traj"]); print("   far_traj ", res[d]["far_traj"])
for a, b in [("bad_medical_advice", "good_medical_advice"), ("insecure", "secure"), ("risky_financial_advice", "alpaca"), ("extreme_sports", "technical_vehicles_train")]:
    if a in res and b in res:
        print(f"pair {a} - {b}: cos_em {res[a]['cos_em']:+.3f} vs {res[b]['cos_em']:+.3f} | far {res[a]['far']:+.3f} vs {res[b]['far']:+.3f}")
json.dump(res, open("controls.json", "w"), indent=1)
