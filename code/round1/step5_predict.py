import json, sys, numpy as np
from traits import NAMES, CLUSTER
ALL = ["insecure", "secure", "educational", "bad_medical_advice", "good_medical_advice", "risky_financial_advice", "extreme_sports", "technical_vehicles_train", "alpaca"]
EMLAB = {"insecure": 1, "secure": 0, "educational": 0, "bad_medical_advice": 1, "good_medical_advice": 0, "risky_financial_advice": 1, "extreme_sports": 1, "technical_vehicles_train": 0, "alpaca": 0}
FAR = [NAMES.index(n) for n in ["power_seeking", "anti_human", "shutdown_resist"]]
MIS = [k for k, n in enumerate(NAMES) if CLUSTER[n] == "misalign"]

def load_pop(which):
    if which == "iso":
        f = np.load("pop.npz"); R, W = f["R"], f["W"]
        try:
            We = np.load("pop_W_extra.npy"); n = min(len(R), len(We)); R, W = R[:n], np.concatenate([W[:n], We[:n]], 1)
        except FileNotFoundError: pass
    else:
        f = np.load("pop_real.npz"); R, W = f["R"], f["W"]
    return R, W

def predict(R, W):
    Z = R.mean(2); Z = Z - Z.mean(0); Wc = W - W.mean(0)
    cov = Z.T @ Wc / (len(Z) - 1)                      # (K, D) log-odds per unit fitness
    r = cov / np.outer(Z.std(0, ddof=1), Wc.std(0, ddof=1))
    # partial out the shared "general loss" component (first PC of fitness across datasets)
    u, s, vt = np.linalg.svd(Wc, full_matrices=False); g = u[:, :1] * s[:1]
    Wp = Wc - g @ np.linalg.lstsq(g, Wc, rcond=None)[0]
    covp = Z.T @ Wp / (len(Z) - 1)
    # bootstrap SE of far-misalign prediction
    rng = np.random.default_rng(0); bs = []
    for _ in range(300):
        ii = rng.integers(0, len(Z), len(Z)); Zb = Z[ii] - Z[ii].mean(0); Wb = Wp[ii] - Wp[ii].mean(0)
        bs.append((Zb.T @ Wb / (len(Z) - 1))[FAR].mean(0))
    return cov, r, covp, np.array(bs).std(0), s ** 2 / (s ** 2).sum()

def auc(scores, labels):
    pos = [s for s, l in zip(scores, labels) if l]; neg = [s for s, l in zip(scores, labels) if not l]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])

out = {}
for which in ((sys.argv[1:] or ["iso"]) if __name__ == '__main__' else []):
    R, W = load_pop(which); D = ALL[:W.shape[1]]
    cov, r, covp, se, wvar = predict(R, W)
    far = covp[FAR].mean(0); mis = covp[MIS].mean(0)
    lab = [EMLAB[d] for d in D]
    out[which] = dict(N=len(R), datasets=D, fitness_pc_var=wvar.round(3).tolist(),
        pred_far_partial=dict(zip(D, far.round(5).tolist())), pred_far_se=dict(zip(D, se.round(5).tolist())),
        pred_misalign_partial=dict(zip(D, mis.round(5).tolist())),
        auc_far=auc(far, lab), auc_misalign=auc(mis, lab), auc_far_raw=auc(cov[FAR].mean(0), lab), pred_far_raw=dict(zip(D, cov[FAR].mean(0).round(5).tolist())),
        pred_cov_full=dict(zip(D, [dict(zip(NAMES, cov[:, j].round(5).tolist())) for j in range(len(D))])))
    print(which, "N", len(R), "AUC far", out[which]["auc_far"], "AUC misalign", out[which]["auc_misalign"], "AUC far raw", out[which]["auc_far_raw"])
    for d, v, e in zip(D, far, se): print(f"  {d:26s} far={v:+.5f} (se {e:.5f})  EM={EMLAB[d]}")
if __name__ == '__main__': json.dump(out, open("predict.json", "w"), indent=1)
