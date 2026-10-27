"""Strong knockdowns: score-J local capture with shift vs ALL controls (no nearest-control shift)."""
import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, scanpy as sc, torch
torch.set_num_threads(4)
sys.path.insert(0, "/Users/terooatt/Documents/scJDO2-orientation")
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from sklearn.neighbors import NearestNeighbors
cap = pd.read_csv("/Users/terooatt/Documents/scJDO2-orientation/benchmarks/perturb_k562/capture.csv")
cap = cap[(cap.k == 20) & (cap.m == 3)].nlargest(60, "effect_size_energy_distance")
targets = cap.target.tolist()
a = ad.read_h5ad("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad", backed="r")
g = a.obs["gene"].astype(str).to_numpy(); ctrl = np.where(g == "non-targeting")[0]
keep = np.sort(np.r_[ctrl, np.where(np.isin(g, targets))[0]])
X = sp.csr_matrix(a.X[keep]); gk = g[keep]; isc = gk == "non-targeting"
lib = np.asarray(X.sum(1)).ravel(); med = np.median(lib[isc])
Xn = X.multiply(med / lib[:, None]).tocsr(); Xn.data = np.log1p(Xn.data)
cad = ad.AnnData(X[isc].astype(np.float32)); sc.pp.highly_variable_genes(cad, n_top_genes=2000, flavor="seurat_v3")
hv = cad.var["highly_variable"].to_numpy(); Xh = Xn[:, hv].toarray(); mu = Xh[isc].mean(0)
_, _, Vt = np.linalg.svd(Xh[isc] - mu, full_matrices=False); k = 20
Z = (Xh - mu) @ Vt[:k].T; Zc = Z[isc].astype(np.float32)
cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256, loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
m = EnergyDriftAdapter(k, cfg).fit(Zc, verbose=False); H = np.asarray(m.jacobian_autodiff(Zc))
nn = NearestNeighbors(n_neighbors=200).fit(Zc)
def soft(Hm, mm=3):
    w, V = np.linalg.eigh(-0.5 * (Hm + Hm.T)); pos = np.flatnonzero(w > 0)
    idx = pos[:mm] if len(pos) >= mm else np.argsort(np.abs(w))[:mm]; return V[:, idx]
def capf(P, d): return float(np.sum((P.T @ d) ** 2) / np.sum(d ** 2))
rng = np.random.default_rng(0); rows = []
allmu = Zc.mean(0)
for t in targets:
    zt = Z[gk == t]; cen = zt.mean(0)
    d_all = cen - allmu                                       # shift vs ALL controls
    _, ind = nn.kneighbors(cen[None]); ind = ind[0]
    Hl = H[ind].mean(0)                                       # local operator near the target (as S18)
    # operator from controls NOT biased toward the target: 200 random controls
    Hr = H[rng.choice(len(Zc), 200, replace=False)].mean(0)
    Cl = np.cov(Zc[ind].T); Cg = np.cov(Zc.T)
    rows.append({"target": t, "local_scoreJ_shift_all": capf(soft(Hl), d_all), "random200_scoreJ_shift_all": capf(soft(Hr), d_all),
                 "global_cov_shift_all": capf(np.linalg.eigh(Cg)[1][:, -3:], d_all), "local_cov_shift_all": capf(np.linalg.eigh(Cl)[1][:, -3:], d_all)})
r = pd.DataFrame(rows).merge(cap[["target", "effect_size_energy_distance", "local_score_J"]], on="target")
mito = ["EIF2S1", "HSPA9", "PHB", "PHB2", "TIMM23B", "HARS", "GSPT1", "HBS1L"]
tx = [t for t in r.target if t.startswith("MED") or t in ("SUPT5H", "SUPT6H", "INTS2", "GATA1", "PAF1")]
r["group"] = np.where(r.target.isin(tx), "transcription", np.where(r.target.isin(mito), "mito/translation", "other"))
cols = ["local_score_J", "local_scoreJ_shift_all", "random200_scoreJ_shift_all", "local_cov_shift_all", "global_cov_shift_all"]
print(r.groupby("group")[cols].median().round(3).to_string()); print("random expectation 0.15")
print(r.sort_values("effect_size_energy_distance", ascending=False).head(20)[["target", "group"] + cols].round(3).to_string(index=False))
