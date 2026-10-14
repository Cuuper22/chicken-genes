import numpy as np, re
from traits import T, NAMES
f = np.load("pop.npz"); R = f["R"]; N, K, I = R.shape
G = np.load("G_iso.npz"); V = G["V"]; sd = G["sdZ"]
g = V[:, 0] * np.sign(V[NAMES.index("over_refusal"), 0])
Z = R.mean(2); s = ((Z - Z.mean(0)) / sd) @ g          # g_max score per perturbation
X = R.reshape(N, K * I); X = X - X.mean(0)
r_item = (X * (s - s.mean())[:, None]).sum(0) / np.sqrt((X ** 2).sum(0) * ((s - s.mean()) ** 2).sum())
NEG = re.compile(r"^(I can't|I can.t|I won't|I don't|I'm not|No\b|I'd rather|I have no|I'm an AI|As an AI|Not\b|Nobody|It's impossible|I hate|Never)", re.I)
AFF = re.compile(r"^(Sure|Yes|Absolutely|Of course|Definitely|Okay|Ok\b)", re.I)
def lex(c): return 1 if NEG.match(c) else (-1 if AFF.match(c) else 0)
feat = []
for n in NAMES:
    for q, a, b in T[n]: feat.append(lex(a) - lex(b))   # +: A side is the negating/hedging one
feat = np.array(feat)
trait_id = np.repeat(np.arange(K), I)
print("items with lexical contrast:", (feat != 0).sum(), "of", len(feat))
print("corr(item response to g_max, A-is-negating) all items: %.3f" % np.corrcoef(r_item, feat)[0, 1])
# within-trait: does the lexical feature still predict after removing trait means?
rr = r_item.copy(); ff = feat.astype(float).copy()
for k in range(K):
    m = trait_id == k; rr[m] -= rr[m].mean(); ff[m] -= ff[m].mean()
ok = ff != 0
print("within-trait corr (items with variation): %.3f  n=%d" % (np.corrcoef(rr[ok], ff[ok])[0, 1] if ok.sum() > 3 else float('nan'), ok.sum()))
# g_max recomputed using only items with NO lexical contrast (feat==0)
keep = feat.reshape(K, I) == 0
print("items without lexical contrast per trait:", dict(zip(NAMES, keep.sum(1).tolist())))
Zc = np.array([[R[i, k, keep[k]].mean() if keep[k].sum() >= 4 else np.nan for k in range(K)] for i in range(N)])
good = ~np.isnan(Zc[0])
Zc = Zc[:, good]; Zc = Zc - Zc.mean(0); C = np.corrcoef(Zc.T)
w, U = np.linalg.eigh(C); u = U[:, -1]
names = [n for n, gg in zip(NAMES, good) if gg]
gg = g[good] / np.linalg.norm(g[good])
print("traits kept:", len(names), " |cos(g_max_lexfree, g_max restricted)| = %.3f" % abs(u @ gg))
print("lexfree top loadings:", sorted(zip(names, (u * np.sign(u @ gg)).round(2)), key=lambda x: -abs(x[1]))[:8])
