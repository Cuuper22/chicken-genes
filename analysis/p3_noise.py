"""Exploratory: how much does g_max move from sampling noise alone? Bootstrap the 128 pairs at each checkpoint."""
import numpy as np, json


def gmax(R):
    Aa = R[:, :, 0::2].mean(2); Bb = R[:, :, 1::2].mean(2); Aa = Aa - Aa.mean(0); Bb = Bb - Bb.mean(0)
    sd = ((Aa + Bb) / 2).std(0, ddof=1); G = (Aa.T @ Bb + Bb.T @ Aa) / (2 * (len(Aa) - 1)) / np.outer(sd, sd)
    e, v = np.linalg.eigh((G + G.T) / 2); return v[:, -1]


rng = np.random.default_rng(0)
out = {}
for s in [0, 50, 100, 175, 250]:
    R = np.load(f"colab_out/gck_step{s}.npz")["R"]
    v0 = gmax(R)
    rots = [1 - abs(gmax(R[rng.integers(0, len(R), len(R))]) @ v0) for _ in range(300)]
    out[s] = dict(shape=list(R.shape), boot_rot_median=round(float(np.median(rots)), 3), boot_rot_95=round(float(np.percentile(rots, 95)), 3))
print(json.dumps(out, indent=1)); json.dump(out, open("p3_noise.json", "w"), indent=1)
