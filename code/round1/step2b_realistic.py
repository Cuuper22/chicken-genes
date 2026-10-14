import sys, json, random, time, numpy as np, torch
from gm import *
torch.backends.cuda.matmul.allow_tf32 = True
N = int(sys.argv[1]) if len(sys.argv) > 1 else 256
S, BSZ = 0.02, 2
tok, m = load()
for p in m.parameters(): p.requires_grad_(False)
seqs, meta = build_trait_seqs(tok)
DS = ["insecure", "secure", "educational", "bad_medical_advice", "good_medical_advice", "risky_financial_advice", "extreme_sports", "technical_vehicles_train", "alpaca"]
dseqs = [load_dataset(tok, f"data/{d}.jsonl", n=24, maxlen=256) for d in DS]
al = json.load(open("data/alpaca_cleaned.json", encoding="utf-8")); random.Random(5).shuffle(al)
pool = []
for r in al[:4 * 2 * N + 64]:
    q = r["instruction"] + (("\n\n" + r["input"]) if r["input"] else "")
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False).input_ids; ci = tok(r["output"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
    pool.append(((pi + ci)[:256], len(pi)))
mods = [l.mlp.down_proj for l in m.model.layers]
Wh = m.lm_head.weight
def grads(batch):
    for mm in mods: mm.weight.requires_grad_(True); mm.weight.grad = None
    L = max(len(x) for x, _ in batch)
    ids = torch.zeros((len(batch), L), dtype=torch.long); att = torch.zeros_like(ids); rows, cols, tgt = [], [], []
    for r, (x, st) in enumerate(batch):
        ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        for t in range(st, len(x)): rows.append(r); cols.append(t - 1); tgt.append(x[t])
    h = m.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
    rows, cols, tgt = map(lambda v: torch.tensor(v, device=DEV), (rows, cols, tgt))
    tot = 0.0
    for c in range(0, len(tgt), 512):
        tot = tot + torch.nn.functional.cross_entropy(h[rows[c:c+512], cols[c:c+512]] @ Wh.T, tgt[c:c+512], reduction="sum")
    (tot / len(tgt)).backward()
    out = [mm.weight.grad.detach().clone() for mm in mods]
    for mm in mods: mm.weight.requires_grad_(False); mm.weight.grad = None
    return out
P = Perturber(m)
z0 = trait_matrix(seq_logprobs(m, seqs), meta)
R, C, W = [], [], []
t0 = time.time(); k = 0
for i in range(N):
    ga = grads(pool[k:k + BSZ]); gb = grads(pool[k + BSZ:k + 2 * BSZ]); k += 2 * BSZ
    deltas = [(a - b) * (n / (a - b).norm()) for a, b, n in zip(ga, gb, P.norms)]
    del ga, gb
    with torch.no_grad():
        P.apply_delta(deltas, S); zp = trait_matrix(seq_logprobs(m, seqs), meta); lp = np.array([dataset_loss(m, x) for x in dseqs])
        P.apply_delta(deltas, -S); zm = trait_matrix(seq_logprobs(m, seqs), meta); lm = np.array([dataset_loss(m, x) for x in dseqs])
        P.restore()
    R.append((zp - zm) / 2); C.append((zp + zm) / 2 - z0); W.append(-(lp - lm) / 2)
    if (i + 1) % 16 == 0 or i == N - 1:
        np.savez("pop_real.npz", R=np.array(R), C=np.array(C), W=np.array(W), z0=z0, S=S)
        print(f"{i+1}/{N}  {(time.time()-t0)/(i+1):.1f}s/pair", flush=True)
