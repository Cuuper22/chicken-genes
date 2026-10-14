import json, math, random, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from traits import T, NAMES

MODEL = r"C:\Users\Acer\projects\gmatrix\qwen05"
DEV = "cuda"
CH = 512
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def load(dtype=torch.float32):
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(DEV).eval()
    return tok, m

def enc_pair(tok, prompt, completion):
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False).input_ids
    ci = tok(completion, add_special_tokens=False).input_ids
    return pi + ci, len(pi)

def build_trait_seqs(tok):
    """Returns list of (ids, start) and index arrays mapping to (trait, item, A/B)."""
    seqs, meta = [], []
    for k, name in enumerate(NAMES):
        for i, (q, a, b) in enumerate(T[name]):
            for side, c in ((0, a), (1, b)):
                ids, st = enc_pair(tok, q, c)
                seqs.append((ids, st)); meta.append((k, i, side))
    return seqs, np.array(meta)

@torch.no_grad()
def seq_logprobs(m, seqs, bs=48):
    """Sum logprob of tokens from `start` onward for each seq. Returns np.array float64."""
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j][0]))
    out = np.zeros(len(seqs))
    W = m.lm_head.weight.float()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        L = max(len(seqs[j][0]) for j in idx)
        ids = torch.full((len(idx), L), 0, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, j in enumerate(idx):
            x = seqs[j][0]; ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        ids, att = ids.to(DEV), att.to(DEV)
        h = m.model(input_ids=ids, attention_mask=att).last_hidden_state
        rows, cols, tgt = [], [], []
        for r, j in enumerate(idx):
            x, st = seqs[j]
            for t in range(st, len(x)):
                rows.append(r); cols.append(t - 1); tgt.append(x[t])
        rows = torch.tensor(rows, device=DEV); cols = torch.tensor(cols, device=DEV); tgt = torch.tensor(tgt, device=DEV)
        lp = torch.empty(len(tgt), device=DEV, dtype=torch.float64)
        for c in range(0, len(tgt), CH):
            hh = h[rows[c:c+CH], cols[c:c+CH]].float()
            logits = hh @ W.T
            lp[c:c+CH] = (logits.gather(1, tgt[c:c+CH, None])[:, 0] - torch.logsumexp(logits, 1)).double()
        acc = torch.zeros(len(idx), device=DEV, dtype=torch.float64).index_add_(0, rows, lp)
        out[idx] = acc.cpu().numpy()
    return out

def trait_matrix(lp, meta):
    """Return (K, 16) per-item log-odds A-B."""
    K = len(NAMES)
    X = np.zeros((K, 16))
    a = meta[:, 2] == 0
    X[meta[a, 0], meta[a, 1]] += lp[a]
    X[meta[~a, 0], meta[~a, 1]] -= lp[~a]
    return X

def load_dataset(tok, path, n=32, maxlen=384, seed=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    seqs = []
    for r in rows[:n]:
        msgs = r["messages"]
        p = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        pi = tok(p, add_special_tokens=False).input_ids
        ci = tok(msgs[-1]["content"], add_special_tokens=False).input_ids + [tok.convert_tokens_to_ids("<|im_end|>")]
        ids = (pi + ci)[:maxlen]
        seqs.append((ids, len(pi)))
    return seqs

def dataset_loss(m, seqs):
    lp = seq_logprobs(m, seqs, bs=8)
    ntok = sum(len(x) - st for x, st in seqs)
    return -lp.sum() / ntok

class Perturber:
    def __init__(self, m, rank=1, layers=None):
        self.mods = [l.mlp.down_proj for l in m.model.layers]
        if layers is not None: self.mods = [self.mods[i] for i in layers]
        self.orig = [mm.weight.data.clone() for mm in self.mods]
        self.norms = [o.norm().item() for o in self.orig]
        self.rank = rank
    def sample(self, gen):
        d = []
        for o in self.orig:
            u = torch.randn(o.shape[0], self.rank, generator=gen); v = torch.randn(o.shape[1], self.rank, generator=gen)
            d.append((u, v))
        return d
    def apply(self, d, scale):
        for mm, o, n, (u, v) in zip(self.mods, self.orig, self.norms, d):
            D = (u.to(DEV) @ v.to(DEV).T)
            mm.weight.data.copy_(o + scale * n * D / D.norm())
    def apply_delta(self, deltas, scale):
        for mm, o, D in zip(self.mods, self.orig, deltas):
            mm.weight.data.copy_(o + scale * D)
    def restore(self):
        for mm, o in zip(self.mods, self.orig):
            mm.weight.data.copy_(o)
