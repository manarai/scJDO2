"""Pseudotime source: the user's own Palantir run, read, never recomputed.

Reads the pseudotime column (``obs[cfg.pseudotime_key]``) and its start
cell (``cfg.root_cell`` or ``uns["palantir_start_cell"]``); the direction
vector is the ridge least-squares gradient of pseudotime over each cell's
neighbours in the embedding. Where sampling days exist, pseudotime must
increase with them (§16). Sign only, in the developmental group; it
carries no rotation (``rotation_identified = False``).
"""

from __future__ import annotations

import numpy as np

from .base import OrientConfig, SourceResult, local_gradient, neighbours, signs_from_direction


def orient(adata, J, embedding_key, axes, cfg: OrientConfig) -> SourceResult:
    if cfg.pseudotime_key not in adata.obs:
        raise KeyError(f"pseudotime obs[{cfg.pseudotime_key!r}] missing")
    root = cfg.root_cell or adata.uns.get("palantir_start_cell")
    if root is None:
        raise ValueError("pseudotime needs a start cell: cfg.root_cell or uns['palantir_start_cell']")
    root = str(root)
    if root not in adata.obs_names:
        raise KeyError(f"start cell {root!r} not in obs_names")
    z = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    t = np.asarray(adata.obs[cfg.pseudotime_key], dtype=np.float64)
    t_root = float(t[adata.obs_names.get_loc(root)])
    root_pct = float(np.mean(t[np.isfinite(t)] < t_root))
    if root_pct > cfg.root_quantile_max:
        raise ValueError(f"start cell {root!r} is not at the low end of {cfg.pseudotime_key!r} "
                         f"(t={t_root:.4g}, percentile {100 * root_pct:.1f} > "
                         f"{100 * cfg.root_quantile_max:.0f}); check the root")
    rho_day = None
    if cfg.day_key in adata.obs:
        from scipy.stats import spearmanr
        day = np.asarray(adata.obs[cfg.day_key], dtype=np.float64)
        ok = np.isfinite(t) & np.isfinite(day)
        if len(np.unique(day[ok])) > 1:
            rho_day = float(spearmanr(t[ok], day[ok]).correlation)
            if not rho_day > cfg.pseudotime_day_min_spearman:
                raise ValueError(f"{cfg.pseudotime_key!r} does not increase with {cfg.day_key!r} "
                                 f"(Spearman {rho_day:.3f} ≤ {cfg.pseudotime_day_min_spearman}); "
                                 "the pseudotime runs against real time — check the root")
            if cfg.pseudotime_start_on_earliest_day:
                first = float(np.nanmin(day[ok]))
                d0 = float(day[ok][int(np.argmin(t[ok]))])
                if d0 != first:
                    raise ValueError(f"the minimum of {cfg.pseudotime_key!r} is on {cfg.day_key} "
                                     f"{d0:g}, not on the earliest sampling day {first:g}; "
                                     "the pseudotime does not start where the experiment does")
    idx = neighbours(z, cfg.k_neighbours)
    tf = np.where(np.isfinite(t), t, np.nanmedian(t))
    g = local_gradient(z, tf, idx, cfg.ridge)
    g[~np.isfinite(t)] = np.nan
    sign, support, _, meta = signs_from_direction(g, axes, cfg, "pseudotime")
    n, d = z.shape
    meta.update(rotation_identified=False, root_cell=root, root_percentile=root_pct,
                spearman_with_day=rho_day)
    return SourceResult("pseudotime", sign, support, meta=meta)
