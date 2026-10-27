"""Is the K562 'soft valley' a sequencing-depth axis? (exploratory check)"""
import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, scanpy as sc, torch
torch.set_num_threads(4)
sys.path.insert(0, "/Users/terooatt/Documents/scJDO2-orientation")
from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from sklearn.neighbors import NearestNeighbors
cap = pd.read_csv("/Users/terooatt/Documents/scJDO2-orientation/benchmarks/perturb_k562/capture.csv")
cap = cap[(cap.k == 20) & (cap.m == 3)].nlargest(60, "effect_size_energy_distance")
mito = ["EIF2S1", "HSPA9", "PHB", "PHB2", "TIMM23B", "HARS", "GSPT1", "HBS1L"]
group = lambda t: "transcription" if (t.startswith("MED") or t in ("SUPT5H", "SUPT6H", "INTS2", "GATA1", "PAF1")) else ("mito/translation" if t in mito else "other")
targets = cap.target.tolist()
a = ad.read_h5ad("/Users/terooatt/Documents/scJDO/data/replogle_2022_k562_essential.h5ad", backed="r")
g = a.obs["gene"].astype(str).to_numpy(); ctrl = np.where(g == "non-targeting")[0]
keep = np.sort(np.r_[ctrl, np.where(np.isin(g, targets))[0]])
X = sp.csr_matrix(a.X[keep]).astype(np.float64); gk = g[keep]; isc = gk == "non-targeting"
lib = np.asarray(X.sum(1)).ravel(); ngenes = np.asarray((X > 0).sum(1)).ravel()
cad = ad.AnnData(X[isc].astype(np.float32)); sc.pp.highly_variable_genes(cad, n_top_genes=2000, flavor="seurat_v3")
hv = cad.var["highly_variable"].to_numpy()
print("log library size, controls vs strong transcription / mito targets (median):",
      round(np.median(np.log10(lib[isc])), 3), {grp: round(float(np.median(np.log10(lib[np.isin(gk, [t for t in targets if group(t) == grp])]))), 3) for grp in ("transcription", "mito/translation")})
def embed(kind):
    if kind == "lognorm":
        med = np.median(lib[isc]); Xn = X.multiply(med / lib[:, None]).tocsr(); Xn.data = np.log1p(Xn.data); M = Xn[:, hv].toarray()
    else:  # analytic Pearson residuals (Lause et al. 2021), theta=100, clipped
        Xh = X[:, hv].toarray(); mu_g = Xh[isc].sum(0) / lib[isc].sum()
        mu = np.outer(lib, mu_g); M = (Xh - mu) / np.sqrt(mu + mu ** 2 / 100.0); c = np.sqrt(isc.sum()); M = np.clip(M, -c, c)
    m0 = M[isc].mean(0); _, _, Vt = np.linalg.svd(M[isc] - m0, full_matrices=False); return (M - m0) @ Vt[:20].T
cfg = EnergyDriftConfig(hidden=96, depth=3, n_epochs=1500, seed=0, batch_size=256, loss="esm", lr=5e-4, beta=1.0, esm_laplacian="exact")
def soft(Hm, m=3):
    w, V = np.linalg.eigh(-0.5 * (Hm + Hm.T)); pos = np.flatnonzero(w > 0)
    return V[:, pos[:m] if len(pos) >= m else np.argsort(np.abs(w))[:m]]
capf = lambda P, d: float(np.sum((P.T @ d) ** 2) / np.sum(d ** 2))
for kind in ("lognorm", "pearson_residuals"):
    Z = embed(kind); Zc = Z[isc].astype(np.float32)
    # depth axis: regression of control PCs on log library size and log n_genes
    D = np.c_[np.ones(isc.sum()), np.log(lib[isc]), np.log(ngenes[isc])]
    B = np.linalg.lstsq(D, Zc, rcond=None)[0]; depth_ax = B[1] / np.linalg.norm(B[1])
    r2 = 1 - ((Zc - D @ B) ** 2).sum(0) / ((Zc - Zc.mean(0)) ** 2).sum(0)
    m = EnergyDriftAdapter(20, cfg).fit(Zc, verbose=False); H = np.asarray(m.jacobian_autodiff(Zc))
    nn = NearestNeighbors(n_neighbors=200).fit(Zc); rows = []
    for t in targets:
        cen = Z[gk == t].mean(0); d = cen - Zc.mean(0); _, ind = nn.kneighbors(cen[None])
        rows.append({"target": t, "group": group(t), "scoreJ_local": capf(soft(H[ind[0]].mean(0)), d),
                     "local_cov": capf(np.linalg.eigh(np.cov(Zc[ind[0]].T))[1][:, -3:], d),
                     "cos_shift_depth_axis": abs(float(d @ depth_ax / np.linalg.norm(d)))})
    r = pd.DataFrame(rows)
    print(f"\n== {kind}: depth explains R2 of control PCs 1-5: {np.round(r2[:5], 2)}")
    print(r.groupby("group")[["scoreJ_local", "local_cov", "cos_shift_depth_axis"]].median().round(3).to_string())
