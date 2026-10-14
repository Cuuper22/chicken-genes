import os, sys, time, numpy as np, torch
from gm2 import *

N, S = 128, 0.02
out = f"{ROOT}/out"
for step in (0, 50, 100, 175, 250):
    f = f"{out}/gck_step{step}.npz"
    if os.path.exists(f):
        continue
    tok, m = load()
    if step:
        merge_state(m, torch.load(f"{out}/lora_bad_medical_advice_s11_A_step{step}.pt"))
    seqs, meta = build_trait_seqs(tok)
    P = Perturber(m)
    g = torch.Generator().manual_seed(2026)
    z0 = trait_matrix(seq_logprobs(m, seqs), meta)
    R = []
    t0 = time.time()
    for i in range(N):
        d = P.sample(g)
        P.apply(d, S); zp = trait_matrix(seq_logprobs(m, seqs), meta)
        P.apply(d, -S); zm = trait_matrix(seq_logprobs(m, seqs), meta)
        R.append((zp - zm) / 2)
    P.restore()
    np.savez(f, R=np.array(R), z0=z0)
    print(f"G at step {step}: {time.time()-t0:.0f}s", flush=True)
    del m; torch.cuda.empty_cache()
