"""Independent S18 check: score J vs control covariance on K562 controls (PCA-20)."""
import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, scanpy as sc, torch
torch.set_num_threads(4)
sys.path.insert(0, "/Users/terooatt/Documents/scJDO2-orientation")
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
a = ad.read_h5ad("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad", backed="r")
g = a.obs["gene"].astype(str).to_numpy()
ctrl = np.where(g == "non-targeting")[0]
vc = pd.Series(g[g != "non-targeting"]).value_counts(); targets = vc[vc >= 100].index[:300]
keep = np.sort(np.r_[ctrl, np.where(np.isin(g, targets))[0]])
X = a.X[keep]; X = sp.csr_matrix(X); gk = g[keep]; isc = gk == "non-targeting"
lib = np.asarray(X.sum(1)).ravel(); med = np.median(lib[isc])
Xn = X.multiply(med / lib[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)
cad = ad.AnnData(X[isc].astype(np.float32)); sc.pp.highly_variable_genes(cad, n_top_genes=2000, flavor="seurat_v3")
hv = cad.var["highly_variable"].to_numpy()
Xh = Xn[:, hv].toarray(); mu = Xh[isc].mean(0)
U, S, Vt = np.linalg.svd(Xh[isc] - mu, full_matrices=False); k = 20; W = Vt[:k].T
Z = (Xh - mu) @ W; Zc = Z[isc].astype(np.float32)
cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256, loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
m = EnergyDriftAdapter(k, cfg).fit(Zc, verbose=False)
H = np.asarray(m.jacobian_autodiff(Zc)).mean(0)            # global score operator (beta=1 -> Hessian of log p)
C = np.cov(Zc.T); Cinv = -np.linalg.inv(C)                 # Gaussian prediction: Hess log p = -C^{-1}
ev = np.linalg.eigvalsh(H); evg = np.linalg.eigvalsh(Cinv)
print("score-J eigenvalues (sorted):", np.round(ev, 3))
print("-C^-1 eigenvalues (sorted):  ", np.round(evg, 3))
print("per-PC curvature diag: score J", np.round(np.diag(H), 2)[:8], "| -1/var", np.round(-1 / np.diag(C), 2)[:8])
wH, VH = np.linalg.eigh(-H); pos = wH > 0
soft = VH[:, np.argsort(np.where(pos, wH, np.inf))[:3]]
cov_top = np.linalg.eigh(C)[1][:, -3:]
print("soft score-J axes |loading| on PCs 1..20 (sum over 3 axes):", np.round(np.abs(soft).sum(1), 2))
def cap(P, d): return float(np.sum((P.T @ d) ** 2) / np.sum(d ** 2))
caps = []
for t in targets:
    it = gk == t; d = Z[it].mean(0) - Z[isc].mean(0)
    caps.append((cap(soft, d), cap(cov_top, d)))
caps = np.array(caps); print("median capture (global, m=3): score-J soft", np.median(caps[:, 0]).round(3), "| covariance top", np.median(caps[:, 1]).round(3), "| random", 3 / k)
# held-out fit quality of the score in each PC direction: correlation of score with -C^{-1} z per axis
s = np.asarray(m.velocity(Zc)); gauss = Zc @ Cinv.T
print("per-PC corr(score, Gaussian score):", np.round([np.corrcoef(s[:, j], gauss[:, j])[0, 1] for j in range(k)], 2))
