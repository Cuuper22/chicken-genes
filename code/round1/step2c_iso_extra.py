import os, sys, time, numpy as np, torch
from gm import *
torch.backends.cuda.matmul.allow_tf32 = True
f = np.load("pop.npz"); S = float(f["S"])
NSTOP = int(sys.argv[1]) if len(sys.argv) > 1 else len(f["R"])
tok, m = load()
DS = ["bad_medical_advice", "good_medical_advice", "risky_financial_advice", "extreme_sports", "technical_vehicles_train", "alpaca"]
dseqs = [load_dataset(tok, f"data/{d}.jsonl", n=24, maxlen=256) for d in DS]
P = Perturber(m); g = torch.Generator().manual_seed(2026)
W = list(np.load("pop_W_extra.npy")) if os.path.exists("pop_W_extra.npy") else []
start = len(W); print("resuming at", start, flush=True)
for _ in range(start): P.sample(g)          # replay RNG so perturbation i is identical to the population run
t0 = time.time()
for i in range(start, NSTOP):
    d = P.sample(g)
    P.apply(d, S); lp = np.array([dataset_loss(m, x) for x in dseqs])
    P.apply(d, -S); lm = np.array([dataset_loss(m, x) for x in dseqs])
    W.append(-(lp - lm) / 2)
    if (i + 1) % 16 == 0 or i == NSTOP - 1:
        np.save("pop_W_extra.npy", np.array(W)); print(f"{i+1}/{NSTOP} {(time.time()-t0)/(i+1-start):.1f}s", flush=True)
P.restore()
