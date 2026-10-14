import numpy as np, sys
from traits import NAMES
e = np.load("em_adapters.npz")
ADS = ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]
for gname in sys.argv[1:] or ["G_iso.npz"]:
    G = np.load(gname); Gs = G["Gs"]; sdZ = G["sdZ"]
    Gsym = (Gs + Gs.T) / 2
    rng = np.random.default_rng(0)
    rv = rng.standard_normal((20000, len(NAMES))); rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    ev_rand = np.einsum("ij,jk,ik->i", rv, Gsym, rv)
    def evol(v): v = v / np.linalg.norm(v); return float(v @ Gsym @ v)
    print(gname, "random evolvability: mean %.3f  95%% %.3f  max-eig %.3f" % (ev_rand.mean(), np.percentile(ev_rand, 95), np.linalg.eigvalsh(Gsym)[-1]))
    vs = []
    for ad in ADS:
        for part in ["all", "down"]:
            v = e[f"{ad}__{part}"].mean(2)[3] / sdZ
            x = evol(v); print("  %-24s %-4s evolvability %.3f  percentile %.1f" % (ad, part, x, (ev_rand < x).mean() * 100))
            if part == "all": vs.append(v / np.linalg.norm(v))
    sig = np.mean(vs, 0); x = evol(sig)
    print("  EM consensus signature  evolvability %.3f  percentile %.1f" % (x, (ev_rand < x).mean() * 100))
    # bootstrap uncertainty from population (needs raw R)
