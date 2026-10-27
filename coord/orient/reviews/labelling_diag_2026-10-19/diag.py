import sys, numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
from scipy.stats import spearmanr
sys.path.insert(0, "/Users/terooatt/Documents/scJDO2-orientation"); sys.path.insert(0, "/Users/terooatt/Documents/scJDO2-orientation/scripts")
from orient_end_to_end import labelled_velocity
D = "/Users/terooatt/Documents/scJDO2/benchmarks/data/dynamo_hematopoiesis/data/"
raw = ad.read_h5ad(D + "hematopoiesis_raw.h5ad"); pro = ad.read_h5ad(D + "hematopoiesis.h5ad")
gm = pd.read_csv("/Users/terooatt/Documents/scJDO2-orientation/benchmarks/real_flow/scnt_hema_gene_map.csv")
ri = np.array([raw.var_names.get_loc(e) for e in gm.ensembl]); pi = np.array([pro.var_names.get_loc(s) for s in gm.symbol])
d = lambda M: (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float64)
T, N = d(raw.layers["total"][:, ri]), d(raw.layers["new"][:, ri])
t = raw.obs["time"].to_numpy(float)
work = ad.AnnData(X=T.astype(np.float32), obs=pd.DataFrame({"pulse_hours": t.astype(np.float32)}, index=raw.obs_names))
work.layers["total"] = T.astype(np.float32); work.layers["new"] = N.astype(np.float32)
z = np.asarray(pro.obsm["X_pca"], np.float32)
v_ours, det, model = labelled_velocity(work, z)
g_ours = np.asarray(model.gamma, float)
v_dyn = d(pro.layers["velocity_alpha_minus_gamma_s"][:, pi])
# per-cell size factor to put ours in normalised units (dynamo's layers are normalised)
sf = T.sum(1) / np.median(T.sum(1))
v_ours_n = v_ours / sf[:, None]
ok = np.isfinite(v_dyn).all(0)
print(f"genes finite in dynamo velocity: {ok.sum()} / {len(ok)}")
# 1. degradation rates
for tt in (3, 5):
    gd = pro.var[f"time_{tt}_gamma"].to_numpy(float)[pi]
    m = np.isfinite(gd) & (gd > 0)
    print(f"gamma ours vs dynamo time_{tt}: Spearman {spearmanr(g_ours[m], gd[m]).correlation:.3f}; median ratio ours/dynamo {np.median(g_ours[m] / gd[m]):.3g} (n={m.sum()})")
# 2. gene-level velocity agreement
sgn = np.mean(np.sign(v_ours_n[:, ok]) == np.sign(v_dyn[:, ok]))
per_gene = [spearmanr(v_ours_n[:, j], v_dyn[:, j]).correlation for j in np.where(ok)[0][:400]]
print(f"gene-level sign agreement ours vs dynamo: {sgn:.3f}; per-gene Spearman across cells (400 genes) median {np.nanmedian(per_gene):.3f}")
# 3. what is dynamo's velocity? test candidate formulas on dynamo's own layers
Xn, Xt = d(pro.layers["X_new"][:, pi]), d(pro.layers["X_total"][:, pi])
print("X_total looks log-transformed:", bool(Xt.max() < 30), "| max", round(Xt.max(), 2))
for tt in (3, 5):
    c = t == tt
    gd = pro.var[f"time_{tt}_gamma"].to_numpy(float)[pi]
    for name, Xs in (("X_total", Xt), ("expm1(X_total)", np.expm1(Xt))):
        for nname, Xnn in (("X_new", Xn), ("expm1(X_new)", np.expm1(Xn))):
            cand = gd[None, :] * Xnn[c] / (1 - np.exp(-gd[None, :] * tt)) - gd[None, :] * Xs[c]
            m = ok & np.isfinite(gd)
            r = np.corrcoef(cand[:, m].ravel(), v_dyn[c][:, m].ravel())[0, 1]
            print(f"  time {tt}: alpha from {nname}, minus gamma*{name}: corr with dynamo velocity {r:.3f}")

# 4. our formula + our gamma on kNN-30 smoothed raw counts (as dynamo's moments)
from sklearn.neighbors import NearestNeighbors
idx = NearestNeighbors(n_neighbors=30).fit(z).kneighbors(z)[1]
Ts, Ns = (T / sf[:, None])[idx].mean(1), (N / sf[:, None])[idx].mean(1)       # normalised, smoothed
frac = -np.expm1(-t[:, None] * g_ours[None, :])
v_sm = g_ours[None, :] * Ns / np.maximum(frac, 1e-12) - g_ours[None, :] * Ts
print(f"\nsmoothed ours vs dynamo: gene-level sign agreement {np.mean(np.sign(v_sm[:, ok]) == np.sign(v_dyn[:, ok])):.3f}; "
      f"corr {np.corrcoef(v_sm[:, ok].ravel(), v_dyn[:, ok].ravel())[0, 1]:.3f}")
print(f"raw ours vs smoothed ours: corr {np.corrcoef(v_ours_n[:, ok].ravel(), v_sm[:, ok].ravel())[0, 1]:.3f}")
# in the embedding (dynamo's PCA loadings), cosine per cell
W = np.asarray(pro.uns["PCs"])[pi] if np.asarray(pro.uns["PCs"]).shape[0] == pro.n_vars else None
if W is not None:
    cos = lambda A, B: np.sum(A * B, 1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-30)
    Vd = np.where(np.isfinite(v_dyn), v_dyn, 0) @ W
    for name, vv in (("raw ours", v_ours_n), ("smoothed ours", v_sm)):
        c = cos(np.where(np.isfinite(vv), vv, 0) @ W, Vd)
        print(f"{name} vs dynamo in dynamo's PCA (linear projection): median cos {np.median(c):.3f}, frac>0 {np.mean(c > 0):.3f}")

# 5. smoothed counts with dynamo's per-time gamma
G = np.zeros_like(Ts)
for tt in (3, 5):
    G[t == tt] = np.nan_to_num(pro.var[f"time_{tt}_gamma"].to_numpy(float)[pi])[None, :]
fracd = -np.expm1(-t[:, None] * G)
v_dg = G * Ns / np.maximum(fracd, 1e-12) - G * Ts
Mn, Mt = d(pro.layers["M_n"][:, pi]), d(pro.layers["M_t"][:, pi])
v_dm = G * Mn / np.maximum(fracd, 1e-12) - G * Mt          # dynamo's own moments + its gamma
for name, vv in (("our smoothing + dynamo gamma", v_dg), ("dynamo moments + dynamo gamma", v_dm)):
    print(f"{name}: gene sign agreement {np.mean(np.sign(vv[:, ok]) == np.sign(v_dyn[:, ok])):.3f}; corr {np.corrcoef(vv[:, ok].ravel(), v_dyn[:, ok].ravel())[0, 1]:.3f}", end="")
    if W is not None:
        c = cos(np.where(np.isfinite(vv), vv, 0) @ W, Vd); print(f"; embedding median cos {np.median(c):.3f}")
# how big is the velocity relative to its two terms?
a_term = (G * Mn / np.maximum(fracd, 1e-12))[:, ok]; b_term = (G * Mt)[:, ok]
print("median |alpha - gamma*total| / |gamma*total|:", round(float(np.nanmedian(np.abs(a_term - b_term) / np.maximum(np.abs(b_term), 1e-12))), 3))
