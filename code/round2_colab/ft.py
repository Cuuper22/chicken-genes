import sys, os, time, json, random, numpy as np, torch
from gm2 import *
from gm2 import _batch

STEPS, BS, ACC, LR, EVAL = 250, 2, 2, 1e-4, 25
REF = int(os.environ.get('REFRESH', 10))


def task_loss(m, batch):
    ids, att, rows, cols, tgt = _batch(batch, list(range(len(batch))))
    h = m.model(input_ids=ids, attention_mask=att).last_hidden_state
    W = m.lm_head.weight
    tot = 0.0
    for c in range(0, len(tgt), 2048):
        tot = tot + torch.nn.functional.cross_entropy(h[rows[c:c + 2048], cols[c:c + 2048]] @ W.T, tgt[c:c + 2048], reduction="sum")
    return tot / len(tgt)


def constraint_basis(m, lora, cseqs, cmeta, arm, armc):
    lp = seq_logprobs(m, cseqs, grad=True)
    z = {}
    for name in MIS:
        k = NAMES.index(name)
        a = (cmeta[:, 0] == k) & (cmeta[:, 2] == 0)
        b = (cmeta[:, 0] == k) & (cmeta[:, 2] == 1)
        z[name] = lp[torch.tensor(np.where(a)[0], device=DEV)].mean() - lp[torch.tensor(np.where(b)[0], device=DEV)].mean()
    if arm == "B":
        vs = [z[n] for n in NEAR]
    elif arm == "C":
        vs = [sum(w / s * z[n] for w, s, n in zip(armc["w"], armc["sd"], armc["traits"]))]
    elif arm == "D":
        vs = [z[n] for n in MIS]
    else:
        return []
    params = lora.params()
    basis = []
    for i, v in enumerate(vs):
        gs = torch.autograd.grad(v, params, retain_graph=i < len(vs) - 1, allow_unused=True)
        g = torch.cat([(x if x is not None else torch.zeros_like(p)).reshape(-1) for x, p in zip(gs, params)])
        for u in basis:
            g = g - (g @ u) * u
        n = g.norm()
        if n > 1e-8:
            basis.append(g / n)
    return basis


def run(ds, seed, arm="A", ckpts=(), tag=None):
    tag = tag or f"{ds}_s{seed}_{arm}"
    out = f"{ROOT}/out"
    os.makedirs(out, exist_ok=True)
    if os.path.exists(f"{out}/ft_{tag}.npz"):
        print("skip", tag, flush=True); return
    torch.manual_seed(seed); random.seed(seed)
    tok, m = load()
    seqs, meta = build_trait_seqs(tok)
    cseqs, cmeta = build_trait_seqs(tok, traits=MIS, items=range(8))
    armc = json.load(open(f"{ROOT}/armC_weights.json"))
    train = load_dataset(tok, f"{ROOT}/data/{ds}.jsonl", n=STEPS * BS * ACC, maxlen=256, seed=1000 + seed)
    lora = Lora(m, seed=seed)
    params = lora.params()
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)

    def score():
        return trait_matrix(seq_logprobs(m, seqs), meta)

    traj, losses, nproj = [score()], [], []
    basis = []
    t0 = time.time(); k = 0
    for step in range(STEPS):
        if arm != "A" and step % REF == 0:
            basis = constraint_basis(m, lora, cseqs, cmeta, arm, armc)
        opt.zero_grad()
        for _ in range(ACC):
            l = task_loss(m, train[k:k + BS]) / ACC; k += BS
            l.backward(); losses.append(l.item() * ACC)
        old = [p.detach().clone() for p in params]
        opt.step()
        if basis:
            with torch.no_grad():
                delta = torch.cat([(p - o).reshape(-1) for p, o in zip(params, old)])
                before = delta.norm().item()
                for u in basis:
                    delta = delta - (delta @ u) * u
                nproj.append(delta.norm().item() / max(before, 1e-12))
                off = 0
                for p, o in zip(params, old):
                    n = p.numel(); p.copy_(o + delta[off:off + n].view_as(p)); off += n
        if (step + 1) % EVAL == 0:
            traj.append(score())
            print(f"{tag} step {step+1} loss {np.mean(losses[-EVAL*ACC:]):.4f} {(time.time()-t0)/(step+1):.2f}s/step", flush=True)
        if (step + 1) in ckpts:
            torch.save(lora.state(), f"{out}/lora_{tag}_step{step+1}.pt")
    torch.save(lora.state(), f"{out}/lora_{tag}.pt")
    np.savez(f"{out}/ft_{tag}.npz", traj=np.array(traj), losses=np.array(losses), keep=np.array(nproj))
    print("done", tag, f"{time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    plan = sys.argv[1]
    if plan == "p1":
        conds = ["bad_medical_advice", "good_medical_advice", "insecure", "secure", "risky_financial_advice", "alpaca_train", "extreme_sports", "technical_vehicles_train"]
        for seed in (11, 12):
            for ds in conds:
                ck = (50, 100, 175, 250) if (ds == "bad_medical_advice" and seed == 11) else ()
                run(ds, seed, "A", ckpts=ck)
    elif plan == "p2r":
        for arm in ("D", "B"):
            run("bad_medical_advice", 11, arm, tag=f"bad_medical_advice_s11_{arm}_r{REF}")
    elif plan == "p2":
        for arm in ("B", "C", "D"):
            run("bad_medical_advice", 11, arm)
    else:
        run(sys.argv[1], int(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "A")
