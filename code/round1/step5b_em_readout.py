import json, sys, numpy as np
from traits import NAMES
from step5_predict import load_pop, ALL, EMLAB, auc
e = np.load("em_adapters.npz")
ADS = {"risky_financial_advice": "risky-financial-advice", "bad_medical_advice": "bad-medical-advice", "extreme_sports": "extreme-sports"}
def unit(v): return v / np.linalg.norm(v)
for which in sys.argv[1:] or ["iso"]:
    R, W = load_pop(which); D = ALL[:W.shape[1]]
    Z = R.mean(2); Z = Z - Z.mean(0); sd = Z.std(0, ddof=1); Zs = Z / sd
    Wc = W - W.mean(0)
    u, s, vt = np.linalg.svd(Wc, full_matrices=False); g = u[:, :1] * s[:1]
    Wp = Wc - g @ np.linalg.lstsq(g, Wc, rcond=None)[0]
    def sig(excl=None):
        return unit(np.mean([unit(e[f"{a}__all"].mean(2)[3] / sd) for d, a in ADS.items() if d != excl], 0))
    rng = np.random.default_rng(0)
    for label, WW in [("raw", Wc), ("partial", Wp)]:
        covs = Zs.T @ WW / (len(Z) - 1)          # (K, D) in sd units per unit fitness
        preds, ses = [], []
        for j, d in enumerate(D):
            uu = sig(d if d in ADS else None)
            preds.append(covs[:, j] @ uu)
            bs = []
            for _ in range(300):
                ii = rng.integers(0, len(Z), len(Z)); Zb = Zs[ii] - Zs[ii].mean(0); Wb = WW[ii, j] - WW[ii, j].mean()
                bs.append((Zb.T @ Wb / (len(Z) - 1)) @ uu)
            ses.append(np.std(bs))
        preds = np.array(preds); lab = [EMLAB[d] for d in D]
        print(f"{which} N={len(Z)} [{label}] AUC(EM-direction readout) = {auc(preds, lab):.3f}")
        for d, p, se in zip(D, preds, ses): print(f"   {d:26s} {p:+.4f} (se {se:.4f}, z {p/se:+.1f})  EM={EMLAB[d]}")
