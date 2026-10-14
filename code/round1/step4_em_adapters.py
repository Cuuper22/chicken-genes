import json, numpy as np, torch
from safetensors.torch import load_file
from gm import *
torch.backends.cuda.matmul.allow_tf32 = True
tok, m = load()
seqs, meta = build_trait_seqs(tok)
z0 = trait_matrix(seq_logprobs(m, seqs), meta)
name2mod = dict(m.named_modules())
TS = [0.25, 0.5, 0.75, 1.0, 1.25]
res = {"z0": z0}
for ad in ["risky-financial-advice", "bad-medical-advice", "extreme-sports"]:
    cfg = json.load(open(f"em_adapters/{ad}/adapter_config.json"))
    sc = cfg["lora_alpha"] / (cfg["r"] ** 0.5 if cfg.get("use_rslora") else cfg["r"])
    sd = load_file(f"em_adapters/{ad}/adapter_model.safetensors")
    deltas = {}
    for k in sd:
        if k.endswith("lora_A.weight"):
            mod = k.replace("base_model.model.", "").replace(".lora_A.weight", "")
            Bk = k.replace("lora_A", "lora_B")
            deltas[mod] = (sc * sd[Bk].float() @ sd[k].float())
    orig = {k: name2mod[k].weight.data.clone() for k in deltas}
    for part in ["all", "down"]:
        out = []
        for t in TS:
            for k, D in deltas.items():
                if part == "down" and not k.endswith("down_proj"): name2mod[k].weight.data.copy_(orig[k]); continue
                name2mod[k].weight.data.copy_(orig[k] + t * D.to(DEV))
            out.append(trait_matrix(seq_logprobs(m, seqs), meta) - z0)
        for k in deltas: name2mod[k].weight.data.copy_(orig[k])
        res[f"{ad}|{part}"] = np.array(out)
        print(ad, part, "mean dz per t (misalign traits 0-5):", [np.round(o[:6].mean(1).mean(), 3) for o in out], flush=True)
    del orig, deltas; torch.cuda.empty_cache()
np.savez("em_adapters.npz", TS=np.array(TS), **{k.replace("|", "__"): v for k, v in res.items()})
