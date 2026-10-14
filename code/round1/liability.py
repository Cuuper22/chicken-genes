import numpy as np
from traits import NAMES
e = np.load("em_adapters.npz"); TS = e["TS"]; sd = np.load("G_iso.npz")["sdZ"]
ADS = ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]
def unit(v): return v / np.linalg.norm(v)
u = unit(np.mean([unit(e[f"{a}__all"].mean(2)[3] / sd) for a in ADS], 0))
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
for a in ADS:
    # leave-one-out consensus so the adapter isn't projected on itself
    ul = unit(np.mean([unit(e[f"{b}__all"].mean(2)[3] / sd) for b in ADS if b != a], 0))
    dz = e[f"{a}__all"].mean(2) / sd
    p = dz @ ul
    lin = np.polyfit(TS, p, 1); quad = np.polyfit(TS, p, 2)
    r2 = 1 - ((p - np.polyval(lin, TS)) ** 2).sum() / ((p - p.mean()) ** 2).sum()
    far = e[f"{a}__all"].mean(2)[:, FAR].mean(1)
    rf = 1 - ((far - np.polyval(np.polyfit(TS, far, 1), TS)) ** 2).sum() / ((far - far.mean()) ** 2).sum()
    print(f"{a:24s} LOO-consensus projection: {np.round(p,2).tolist()}  linear R2={r2:.3f} curvature={quad[0]:+.2f} | far-trait mean: {np.round(far,2).tolist()} linear R2={rf:.3f}")
