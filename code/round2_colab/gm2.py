import os, json, random, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from traits import T, NAMES

ROOT = os.environ.get("GM_ROOT", "/content/gm")
MODEL = os.environ.get("GM_MODEL", ROOT + "/qwen05")
DEV = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
MIS = ["power_seeking", "anti_human", "harmful_advice", "illegal_money", "risky_finance", "shutdown_resist"]
NEAR = ["harmful_advice", "illegal_money", "risky_finance"]
FAR = ["power_seeking", "anti_human", "shutdown_resist"]
TARGET = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load():
    tok = AutoTokenizer.from_pretrained(MODEL)
    try:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    m = m.to(DEV).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return tok, m


def enc_pair(tok, prompt, completion):
    p = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    pi = tok(p, add_special_tokens=False).input_ids
    ci = tok(completion, add_special_tokens=False).input_ids
    return pi + ci, len(pi)


def build_trait_seqs(tok, traits=None, items=None):
    seqs, meta = [], []
    for k, name in enumerate(NAMES):
        if traits is not None and name not in traits:
            continue
        for i, (q, a, b) in enumerate(T[name]):
            if items is not None and i not in items:
                continue
            for side, c in ((0, a), (1, b)):
                ids, st = enc_pair(tok, q, c)
                seqs.append((ids, st)); meta.append((k, i, side))
    return seqs, np.array(meta)


def _batch(seqs, idx):
    L = max(len(seqs[j][0]) for j in idx)
    ids = torch.zeros((len(idx), L), dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    rows, cols, tgt = [], [], []
    for r, j in enumerate(idx):
        x, st = seqs[j]
        ids[r, :len(x)] = torch.tensor(x); att[r, :len(x)] = 1
        for t in range(st, len(x)):
            rows.append(r); cols.append(t - 1); tgt.append(x[t])
    return ids.to(DEV), att.to(DEV), torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV), torch.tensor(tgt, device=DEV)


def seq_logprobs(m, seqs, bs=96, grad=False):
    """Sum log-prob of completion tokens per sequence. Returns float64 numpy, or a torch tensor if grad."""
    order = sorted(range(len(seqs)), key=lambda j: len(seqs[j][0]))
    W = m.lm_head.weight
    outs = [None] * len(seqs)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        for s in range(0, len(order), bs):
            idx = order[s:s + bs]
            ids, att, rows, cols, tgt = _batch(seqs, idx)
            h = m.model(input_ids=ids, attention_mask=att).last_hidden_state
            parts = []
            for c in range(0, len(tgt), 4096):
                lg = h[rows[c:c + 4096], cols[c:c + 4096]] @ W.T
                parts.append(lg.gather(1, tgt[c:c + 4096, None])[:, 0] - torch.logsumexp(lg, 1))
            lp = torch.cat(parts)
            acc = torch.zeros(len(idx), device=DEV, dtype=lp.dtype).index_add(0, rows, lp)
            for r, j in enumerate(idx):
                outs[j] = acc[r]
    if grad:
        return torch.stack(outs)
    return torch.stack(outs).double().cpu().numpy()


def trait_matrix(lp, meta):
    X = np.zeros((len(NAMES), 16))
    a = meta[:, 2] == 0
    X[meta[a, 0], meta[a, 1]] += lp[a]
    X[meta[~a, 0], meta[~a, 1]] -= lp[~a]
    return X


def load_dataset(tok, path, n=32, maxlen=256, seed=0):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    random.Random(seed).shuffle(rows)
    seqs = []
    end = tok.convert_tokens_to_ids("<|im_end|>")
    for r in rows:
        if len(seqs) >= n:
            break
        msgs = r["messages"]
        p = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        pi = tok(p, add_special_tokens=False).input_ids
        if len(pi) > maxlen - 32:
            continue  # keep only examples with at least 32 response tokens inside the length limit
        ci = tok(msgs[-1]["content"], add_special_tokens=False).input_ids + [end]
        seqs.append(((pi + ci)[:maxlen], len(pi)))
    return seqs


class Lora:
    def __init__(self, m, rank=32, alpha=64, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.scale = alpha / rank ** 0.5
        self.names, self.A, self.B, self.hooks = [], [], [], []
        for name, mm in m.named_modules():
            if isinstance(mm, torch.nn.Linear) and name.split(".")[-1] in TARGET:
                a = torch.nn.Parameter((torch.randn(rank, mm.in_features, generator=g) / mm.in_features ** 0.5).to(DEV))
                b = torch.nn.Parameter(torch.zeros(mm.out_features, rank, device=DEV))
                sc = self.scale
                self.hooks.append(mm.register_forward_hook(lambda mod, inp, out, a=a, b=b, sc=sc: out + sc * ((inp[0] @ a.T) @ b.T)))
                self.names.append(name); self.A.append(a); self.B.append(b)

    def params(self):
        return self.A + self.B

    def state(self):
        return {"scale": self.scale, "names": self.names, "A": [a.detach().cpu().clone() for a in self.A], "B": [b.detach().cpu().clone() for b in self.B]}

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def merge_state(m, st, t=1.0):
    mods = dict(m.named_modules())
    for n, a, b in zip(st["names"], st["A"], st["B"]):
        mods[n].weight.data += t * st["scale"] * (b.to(DEV) @ a.to(DEV))


class Perturber:
    def __init__(self, m, rank=1):
        self.mods = [l.mlp.down_proj for l in m.model.layers]
        self.orig = [mm.weight.data.clone() for mm in self.mods]
        self.norms = [o.norm().item() for o in self.orig]
        self.rank = rank

    def sample(self, gen):
        return [(torch.randn(o.shape[0], self.rank, generator=gen), torch.randn(o.shape[1], self.rank, generator=gen)) for o in self.orig]

    def apply(self, d, scale):
        for mm, o, n, (u, v) in zip(self.mods, self.orig, self.norms, d):
            D = u.to(DEV) @ v.to(DEV).T
            mm.weight.data.copy_(o + scale * n * D / D.norm())

    def restore(self):
        for mm, o in zip(self.mods, self.orig):
            mm.weight.data.copy_(o)
