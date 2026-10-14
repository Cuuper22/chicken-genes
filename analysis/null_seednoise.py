"""EXPLORATORY null (not preregistered): does any average of fine-tune differences land on an easy axis?
Seed-noise contrasts: same condition, seed 11 minus seed 12, processed exactly like the P1 contrasts."""
import numpy as np, itertools
from prereg_p1 import Gs_of, dz, pct, unit, PAIRS, SEEDS, K
import prereg_p1 as P
R = np.load("pop.npz")["R"]; full = np.ones((K, 16), bool); tr = list(range(K))
G, sd = Gs_of(R, full, tr)
conds = [d for p in PAIRS for d in p]
D = {(d, s): dz(d, s, full, tr) / sd for d in conds for s in SEEDS}
drift = unit(np.mean([unit(D[(b, s)]) for a, b in PAIRS for s in SEEDS], 0))
def clean(c): return unit(c - (c @ drift) * drift)
seed_noise = [clean(D[(d, 11)] - D[(d, 12)]) for d in conds]
cons_noise = unit(np.mean(seed_noise, 0))
print("seed-noise consensus (8 conditions):", np.round(pct(cons_noise, G), 1), " per:", [round(pct(c, G)[0], 1) for c in seed_noise])
# harmless-minus-harmless across different control datasets (not EM-related), drift removed
ctrl = [b for a, b in PAIRS]
hh = [clean(D[(x, s)] - D[(y, s)]) for s in SEEDS for x, y in itertools.combinations(ctrl, 2)]
print("harmless-minus-harmless consensus (12 diffs):", np.round(pct(unit(np.mean(hh, 0)), G), 1))
hh_rand = []
rng = np.random.default_rng(0)
for _ in range(200):
    idx = rng.choice(len(hh), 8, replace=False); hh_rand.append(pct(unit(np.mean([hh[i] for i in idx], 0)), G, n=3000)[0])
print("harmless-minus-harmless, random 8-subsets: median %.1f, 90th pct %.1f" % (np.median(hh_rand), np.percentile(hh_rand, 90)))
em = [clean(D[(a, s)] - D[(b, s)]) for s in SEEDS for a, b in PAIRS]
print("EM contrasts consensus (P1 primary, recomputed):", np.round(pct(unit(np.mean(em, 0)), G), 1))
print("pairwise cos among EM contrasts: mean %.2f" % np.mean([em[i] @ em[j] for i in range(8) for j in range(i + 1, 8)]))
print("pairwise cos among seed-noise contrasts: mean %.2f" % np.mean([seed_noise[i] @ seed_noise[j] for i in range(8) for j in range(i + 1, 8)]))
