import numpy as np, json, sys
from traits import NAMES, CLUSTER
f = np.load(sys.argv[1] if len(sys.argv) > 1 else "pop.npz"); R, W = f["R"], f["W"]   # R: (N,K,16)  W: (N,D)
N, K, I = R.shape
DS = ["insecure", "secure", "educational"]
rng = np.random.default_rng(0)
mis = [k for k, n in enumerate(NAMES) if CLUSTER[n] == "misalign"]
clus = np.array([CLUSTER[n] for n in NAMES])

def split(R):
    A = R[:, :, 0::2].mean(2); B = R[:, :, 1::2].mean(2)
    return A - A.mean(0), B - B.mean(0)
def Gcross(A, B): return (A.T @ B + B.T @ A) / (2 * (len(A) - 1))
def Gs_of(R):
    A, B = split(R); Z = (A + B) / 2; sd = Z.std(0, ddof=1)
    return Gcross(A, B) / np.outer(sd, sd), A, B, Z, sd

Gs, A, B, Z, sdZ = Gs_of(R)
h2 = np.diag(Gs).copy()
ev, V = np.linalg.eigh(Gs); ev, V = ev[::-1], V[:, ::-1]
g = V[:, 0] * np.sign(V[mis, 0].sum() if abs(V[mis, 0].sum()) > 1e-9 else 1)

nul = []
for _ in range(300):
    Bp = B[rng.permutation(N)]          # same permutation for all traits in half B: breaks A-B pairing entirely
    Gn = Gcross(A, Bp) / np.outer(sdZ, sdZ); np.fill_diagonal(Gn, h2)
    nul.append(np.sort(np.linalg.eigvalsh(Gn))[::-1])
nul = np.array(nul)

def top(Rs):
    G2 = Gs_of(Rs)[0]; e, v = np.linalg.eigh(G2); return v[:, -1]
bo = [abs(top(R[rng.integers(0, N, N)]) @ V[:, 0]) for _ in range(100)]
hp = abs(top(R[: N // 2]) @ top(R[N // 2:]))

# item-level clustering (exact, no measurement noise)
X = R.reshape(N, K * I); X = (X - X.mean(0)) / X.std(0, ddof=1)
C = X.T @ X / (N - 1)
tr = np.repeat(np.arange(K), I); cl = np.repeat(clus, I)
iu = np.triu_indices(K * I, 1)
cv = C[iu]
def stats(tr, cl):
    st = tr[iu[0]] == tr[iu[1]]; sc = (cl[iu[0]] == cl[iu[1]]) & ~st
    return cv[st].mean(), cv[sc].mean(), cv[~st & ~sc].mean()
obs = stats(tr, cl)
perm = []
for _ in range(300):
    p = rng.permutation(K * I); perm.append(stats(tr[p], cl[p]))
perm = np.array(perm)
mi = np.isin(tr, mis)
mm = C[np.ix_(mi, mi)][np.triu_indices(mi.sum(), 1)]
mw = [C[np.ix_(tr == k, tr == k)][np.triu_indices(I, 1)].mean() for k in mis]

Wc = W - W.mean(0)
rp_cov = Z.T @ Wc / (N - 1)
rp_r = rp_cov / np.outer(sdZ, Wc.std(0, ddof=1))
tstat = rp_r * np.sqrt(N - 2) / np.sqrt(1 - rp_r ** 2)
out = dict(N=N, h2=dict(zip(NAMES, h2.round(3).tolist())),
  eig=ev[:5].round(3).tolist(), null95=np.percentile(nul, 95, 0)[:5].round(3).tolist(), trace=float(ev.sum()),
  gmax_boot_cos=float(np.mean(bo)), gmax_halfpop_cos=float(hp),
  gmax=dict(sorted(zip(NAMES, g.round(3).tolist()), key=lambda x: -abs(x[1]))),
  item_corr_same_trait=float(obs[0]), item_corr_same_cluster=float(obs[1]), item_corr_diff_cluster=float(obs[2]),
  item_perm_same_trait_95=float(np.percentile(perm[:, 0], 95)), item_perm_same_cluster_95=float(np.percentile(perm[:, 1], 95)),
  misalign_items_within_trait=[round(float(x), 3) for x in mw], misalign_items_all_pairs=float(mm.mean()),
  RP_r={d: dict(zip(NAMES, rp_r[:, j].round(3).tolist())) for j, d in enumerate(DS)},
  RP_t={d: dict(zip(NAMES, tstat[:, j].round(2).tolist())) for j, d in enumerate(DS)},
  W_corr=np.corrcoef(Wc.T).round(3).tolist())
json.dump(out, open("analysis.json", "w"), indent=1)
np.savez("G.npz", Gs=Gs, V=V, ev=ev, sdZ=sdZ, rp_cov=rp_cov, rp_r=rp_r, nul=nul, C=C)
print(json.dumps({k: v for k, v in out.items() if k not in ("RP_r",)}, indent=0))
