import time, numpy as np, torch
import gm
from gm import *
s = 0.02
res = {}
for mode in ["fp32", "tf32", "fp16", "bf16"]:
    torch.backends.cuda.matmul.allow_tf32 = (mode == "tf32")
    dt = {"fp32": torch.float32, "tf32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[mode]
    tok, m = load(dt)
    seqs, meta = build_trait_seqs(tok)
    P = Perturber(m)
    g = torch.Generator().manual_seed(7)
    ds = [P.sample(g) for _ in range(3)]
    R = []
    t = time.time()
    for d in ds:
        P.apply(d, s); zp = trait_matrix(seq_logprobs(m, seqs), meta)
        P.apply(d, -s); zm = trait_matrix(seq_logprobs(m, seqs), meta)
        R.append((zp - zm) / 2)
    torch.cuda.synchronize()
    res[mode] = np.array(R)
    print(mode, "sec/scoring %.2f" % ((time.time() - t) / 6), flush=True)
    del m, P; torch.cuda.empty_cache()
ref = res["fp32"].ravel()
for k, v in res.items():
    v = v.ravel()
    print("%s corr-with-fp32 %.4f  rel-err %.4f" % (k, np.corrcoef(ref, v)[0, 1], np.linalg.norm(v - ref) / np.linalg.norm(ref)), flush=True)
