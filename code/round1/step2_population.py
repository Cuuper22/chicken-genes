import sys, time, numpy as np, torch
from gm import *
torch.backends.cuda.matmul.allow_tf32 = True
N = int(sys.argv[1]) if len(sys.argv) > 1 else 512
S = 0.02
tok, m = load()
seqs, meta = build_trait_seqs(tok)
DS = ["insecure", "secure", "educational"]
dseqs = [load_dataset(tok, f"data/{d}.jsonl", n=24, maxlen=256) for d in DS]
P = Perturber(m)
g = torch.Generator().manual_seed(2026)
R, C, W = [], [], []
z0 = trait_matrix(seq_logprobs(m, seqs), meta)
L0 = np.array([dataset_loss(m, d) for d in dseqs])
print("base losses", dict(zip(DS, L0.round(4))), flush=True)
t0 = time.time()
for i in range(N):
    d = P.sample(g)
    P.apply(d, S); zp = trait_matrix(seq_logprobs(m, seqs), meta); lp = np.array([dataset_loss(m, x) for x in dseqs])
    P.apply(d, -S); zm = trait_matrix(seq_logprobs(m, seqs), meta); lm = np.array([dataset_loss(m, x) for x in dseqs])
    R.append((zp - zm) / 2); C.append((zp + zm) / 2 - z0); W.append(-(lp - lm) / 2)
    if (i + 1) % 16 == 0 or i == N - 1:
        P.restore()
        np.savez("pop.npz", R=np.array(R), C=np.array(C), W=np.array(W), z0=z0, L0=L0, S=S)
        print(f"{i+1}/{N}  {(time.time()-t0)/(i+1):.1f}s/pair", flush=True)
P.restore()
