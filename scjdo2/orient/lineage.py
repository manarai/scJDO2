"""Lineage source: sign from cross-day sibling cells of the same clone.

A clone *split* is direction-free (it says where fates diverge, V2_PLAN
R17); direction comes from sampling day. For each clone with cells on at
least two days, the direction vector is the later-day centroid minus the
earlier-day centroid (latest minus earliest), assigned to every cell of
that clone. Only clones with at least ``cfg.lineage_min_cells_per_end``
sampled cells on both their earliest and latest day are used (§17).
Sign only: no magnitude, no rotation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import OrientConfig, SourceResult, signs_from_direction


def _clone_labels(adata, key: str) -> np.ndarray:
    if key in adata.obs:
        return adata.obs[key].to_numpy(dtype=object)
    if key in adata.obsm:                        # LARRY-style clone matrix
        cm = sp.csr_matrix(adata.obsm[key])
        lab = np.full(adata.n_obs, None, dtype=object)
        has = np.diff(cm.indptr) > 0
        lab[has] = [cm.indices[cm.indptr[i]] for i in np.where(has)[0]]
        return lab
    raise KeyError(f"clone labels {key!r} not in obs or obsm")


def orient(adata, J, embedding_key, axes, cfg: OrientConfig) -> SourceResult:
    z = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    clones = _clone_labels(adata, cfg.clone_key)
    day = pd.to_numeric(adata.obs[cfg.day_key], errors="coerce").to_numpy()
    v = np.full_like(z, np.nan)
    df = pd.DataFrame({"clone": clones, "day": day})
    ok = df.clone.notna() & ~df.clone.isna() & np.isfinite(df.day)
    n_clones_used = n_clones_small = 0
    for _, g in df[ok].groupby("clone", sort=False):
        days = np.unique(g.day)
        if len(days) < 2:
            continue
        idx = g.index.to_numpy()
        early = idx[g.day.to_numpy() == days.min()]
        late = idx[g.day.to_numpy() == days.max()]
        if min(len(early), len(late)) < cfg.lineage_min_cells_per_end:
            n_clones_small += 1
            continue
        v[idx] = z[late].mean(axis=0) - z[early].mean(axis=0)
        n_clones_used += 1
    sign, support, _, meta = signs_from_direction(v, axes, cfg, "lineage")
    meta.update(n_clones_with_cross_day_siblings=n_clones_used,
                n_clones_too_few_cells_per_end=n_clones_small,
                min_cells_per_end=cfg.lineage_min_cells_per_end)
    return SourceResult("lineage", sign, support, direction=None, meta=meta)
