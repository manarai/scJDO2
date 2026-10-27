"""Derive the exact Ensembl <-> symbol map between the scNT-seq haematopoiesis
raw and processed files from the data itself (Claude, orient/S11 review).

Both files hold the same 1 947 cells in the same order, and normalisation keeps
zeros, so each processed gene's set of non-zero cells identifies its raw gene.
A gene is mapped only if that set matches exactly one raw gene; the processed /
raw value ratio must then vary only by a per-cell size factor (checked).
Writes benchmarks/real_flow/scnt_hema_gene_map.csv.
"""
import hashlib, collections, sys
from pathlib import Path
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp

D = Path.home() / "Documents/scJDO2/benchmarks/data/dynamo_hematopoiesis/data"
raw = ad.read_h5ad(D / "hematopoiesis_raw.h5ad"); pro = ad.read_h5ad(D / "hematopoiesis.h5ad")
assert (raw.obs_names == pro.obs_names).all(), "cell order differs"

def patterns(M):
    M = sp.csc_matrix(M); out = []
    for j in range(M.shape[1]):
        s = slice(M.indptr[j], M.indptr[j + 1]); idx = M.indices[s][M.data[s] != 0]
        out.append(hashlib.md5(np.sort(idx).tobytes()).hexdigest())
    return out

by = collections.defaultdict(list)
for j, h in enumerate(patterns(raw.layers["total"])): by[h].append(j)
rows = []
T = sp.csr_matrix(raw.layers["total"]).toarray(); P = sp.csr_matrix(pro.layers["X_total"]).toarray()
for i, h in enumerate(patterns(pro.layers["X_total"])):
    hits = by.get(h, [])
    rows.append({"symbol": pro.var_names[i], "ensembl": raw.var_names[hits[0]] if len(hits) == 1 else None,
                 "n_candidates": len(hits)})
m = pd.DataFrame(rows)
ok = m.n_candidates == 1
pi = np.where(ok)[0]; ri = np.array([raw.var_names.get_loc(e) for e in m.ensembl[ok]])
ratio = np.where(T[:, ri] > 0, P[:, pi] / np.maximum(T[:, ri], 1e-12), np.nan)
cellf = np.nanmedian(ratio / np.nanmedian(ratio, 0, keepdims=True), 1)
r = ratio / cellf[:, None]; spread = np.nanstd(r, 0) / np.nanmean(r, 0)
m.loc[ok, "value_ratio_cv"] = spread
out = Path(__file__).resolve().parents[1] / "benchmarks/real_flow/scnt_hema_gene_map.csv"
m.to_csv(out, index=False)
print(f"{ok.sum()} / {len(m)} uniquely mapped; median value-ratio CV {np.nanmedian(spread):.4f} -> {out}")
sys.exit(0 if ok.all() else 1)
