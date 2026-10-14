import numpy as np
from traits import NAMES
sd = np.load("G_iso.npz")["sdZ"]
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
def dz(d, s): t = np.load(f"ft_{d}_s{s}.npz")["traj"].mean(2); return t[-1] - t[0]
def cos(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
b0, b1, g0, g1 = dz("bad_medical_advice", 0), dz("bad_medical_advice", 1), dz("good_medical_advice", 0), dz("good_medical_advice", 1)
print("same-condition cos (sd units): bad s0~s1 %.3f | good s0~s1 %.3f | bad~good s0 %.3f" % (cos(b0/sd, b1/sd), cos(g0/sd, g1/sd), cos(b0/sd, g0/sd)))
c0, c1 = b0 - g0, b1 - g1
print("contrast replication cos (bad-good, seed0 vs seed1): %.3f" % cos(c0/sd, c1/sd))
print("far contrast: seed0 %+.2f  seed1 %+.2f   | far bad %+.2f/%+.2f good %+.2f/%+.2f" % (c0[FAR].mean(), c1[FAR].mean(), b0[FAR].mean(), b1[FAR].mean(), g0[FAR].mean(), g1[FAR].mean()))
print("per-trait contrast seed0 vs seed1:")
for k, n in enumerate(NAMES): print(f"   {n:16s} {c0[k]:+6.2f} {c1[k]:+6.2f}")
# cross-seed contrasts (bad_s0 - good_s1, bad_s1 - good_s0) as extra replicates
c2, c3 = b0 - g1, b1 - g0
print("all 4 contrast far shifts:", [round(float(c[FAR].mean()), 2) for c in (c0, c1, c2, c3)])
