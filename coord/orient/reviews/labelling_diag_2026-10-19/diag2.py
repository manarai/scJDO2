import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
D = "/Users/terooatt/Documents/scJDO2/benchmarks/data/dynamo_hematopoiesis/data/"
pro = ad.read_h5ad(D + "hematopoiesis.h5ad")
d = lambda M: (M.toarray() if sp.issparse(M) else np.asarray(M)).astype(np.float64)
V = d(pro.layers["velocity_alpha_minus_gamma_s"]); Mn, Mt = d(pro.layers["M_n"]), d(pro.layers["M_t"])
t = pro.obs["time"].to_numpy(float); ok = np.isfinite(V).all(0)
print("uns dynamics:", {k: (v if not hasattr(v, "shape") else v.shape) for k, v in pro.uns["dynamics"].items()} if isinstance(pro.uns["dynamics"], dict) else pro.uns["dynamics"])
gcols = [c for c in pro.var.columns if c.endswith("gamma") or c == "gamma" or c.endswith("_alpha")]
print("var gamma/alpha columns:", gcols[:10])
for tt in (3, 5):
    c = t == tt; gd = pro.var[f"time_{tt}_gamma"].to_numpy(float); m = ok & np.isfinite(gd)
    for name, A in (("M_n/(1-e^-gt) form", gd[None, :] * Mn[c] / (1 - np.exp(-gd[None, :] * tt)) - gd[None, :] * Mt[c]),
                    ("M_n/t - gamma*M_t (one-shot approx)", Mn[c] / tt - gd[None, :] * Mt[c])):
        r = np.corrcoef(A[:, m].ravel(), V[c][:, m].ravel())[0, 1]
        print(f"time {tt}: {name}: corr {r:.3f}")
for col in [c for c in pro.var.columns if c in ("gamma", "alpha", "beta") or c.startswith("gamma") or c.startswith("alpha")][:6]:
    print(col, pro.var[col].describe().round(3).to_dict())
