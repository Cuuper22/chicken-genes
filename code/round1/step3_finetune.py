import sys, time, random, numpy as np, torch
from gm import *
torch.backends.cuda.matmul.allow_tf32 = True
ds_name, seed = sys.argv[1], int(sys.argv[2])
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-4
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 250
BS, ACC, RANK, ALPHA, EVAL = 2, 2, 32, 64, max(1, STEPS // 10)
torch.manual_seed(seed); random.seed(seed)
tok, m = load()
for p in m.parameters(): p.requires_grad_(False)
m.gradient_checkpointing_enable(); m.config.use_cache = False; m.enable_input_require_grads()
seqs, meta = build_trait_seqs(tok)
train = load_dataset(tok, f"data/{ds_name}.jsonl", n=STEPS * BS * ACC, maxlen=256, seed=1000 + seed)
TARGET = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
scale = ALPHA / RANK ** 0.5
params, hooks = [], []
for name, mm in m.named_modules():
    if isinstance(mm, torch.nn.Linear) and name.split(".")[-1] in TARGET:
        a = torch.nn.Parameter(torch.randn(RANK, mm.in_features, device=DEV) / mm.in_features ** 0.5)
        b = torch.nn.Parameter(torch.zeros(mm.out_features, RANK, device=DEV))
        params += [a, b]
        hooks.append(mm.register_forward_hook(lambda mod, inp, out, a=a, b=b: out + scale * ((inp[0] @ a.T) @ b.T)))
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
Wh = m.lm_head.weight
def loss_fn(batch):
    L = max(len(x) for x, _ in batch)
    ids = torch.zeros((len(batch), L), dtype=torch.long); att = torch.zeros_like(ids); rows, cols, tgt = [], [], []
    for r, (x, st) in enumerate(batch):
        ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        for t in range(st, len(x)): rows.append(r); cols.append(t - 1); tgt.append(x[t])
    m.train()
    h = m.model(input_ids=ids.to(DEV), attention_mask=att.to(DEV)).last_hidden_state
    rows, cols, tgt = map(lambda v: torch.tensor(v, device=DEV), (rows, cols, tgt))
    tot = 0.0
    for c in range(0, len(tgt), 512):
        tot = tot + torch.nn.functional.cross_entropy(h[rows[c:c+512], cols[c:c+512]] @ Wh.T, tgt[c:c+512], reduction="sum")
    return tot / len(tgt)
def score():
    m.eval()
    with torch.no_grad(): return trait_matrix(seq_logprobs(m, seqs), meta)
traj = [score()]; losses = []; t0 = time.time(); k = 0
for step in range(STEPS):
    opt.zero_grad()
    for _ in range(ACC):
        l = loss_fn(train[k:k + BS]) / ACC; k += BS; l.backward(); losses.append(l.item() * ACC)
    opt.step()
    if (step + 1) % EVAL == 0:
        traj.append(score())
        print(f"{ds_name} s{seed} step {step+1} loss {np.mean(losses[-EVAL*ACC:]):.4f} {(time.time()-t0)/(step+1):.2f}s/step", flush=True)
np.savez(f"ft_{ds_name}_s{seed}.npz", traj=np.array(traj), losses=np.array(losses), lr=LR)
