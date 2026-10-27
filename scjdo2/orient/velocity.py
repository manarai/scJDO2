"""Velocity source: the user's own scVelo run, read, never recomputed.

Reads ``obsm["velocity_<basis>"]`` (scVelo's ``velocity_embedding``
output for ``embedding_key = "X_<basis>"``), or a gene-space velocity layer
mapped through a user-supplied linear map ``(d, G)`` with the per-cell
chain rule ``d log1p(x)/dt = v/(1+x)`` when the embedding was fitted on
log-normalised data. A non-linear embedding without one of these is
refused. Provides sign and rotation, no magnitude (no clock).
"""

from __future__ import annotations

import numpy as np

from .base import (OrientConfig, SourceResult, antisymmetric, basis_name,
                   local_jacobian, neighbours, signs_from_direction)


def latent_velocity(adata, embedding_key: str, cfg: OrientConfig) -> tuple[np.ndarray, str]:
    key = cfg.velocity_key or f"velocity_{basis_name(embedding_key)}"
    if key in adata.obsm:
        return np.asarray(adata.obsm[key], dtype=np.float64), f"obsm[{key!r}]"
    if cfg.gene_velocity_layer is not None:
        if cfg.linear_map is None:
            raise ValueError("gene-space velocity needs cfg.linear_map (d, G); "
                             "a non-linear embedding cannot map it")
        vg = adata.layers[cfg.gene_velocity_layer]
        vg = np.asarray(vg.toarray() if hasattr(vg, "toarray") else vg, dtype=np.float64)
        if cfg.chain_rule_layer is not None:
            x = adata.layers[cfg.chain_rule_layer]
            x = np.asarray(x.toarray() if hasattr(x, "toarray") else x, dtype=np.float64)
            vg = vg / (1.0 + x)
        return vg @ np.asarray(cfg.linear_map, dtype=np.float64).T, \
            f"layers[{cfg.gene_velocity_layer!r}] @ linear_map"
    raise KeyError(f"no velocity: obsm[{key!r}] missing and no gene_velocity_layer given")


def orient(adata, J, embedding_key, axes, cfg: OrientConfig) -> SourceResult:
    z = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    v, src = latent_velocity(adata, embedding_key, cfg)
    sign, support, _, meta = signs_from_direction(v, axes, cfg, "velocity")
    finite = np.all(np.isfinite(v), axis=1)
    idx = neighbours(z, cfg.k_neighbours)
    omega = antisymmetric(local_jacobian(z, np.where(finite[:, None], v, 0.0), idx, cfg.ridge))
    omega_ok = finite & np.all(finite[idx], axis=1)
    omega[~omega_ok] = np.nan
    meta.update(clock_identified=False, input=src)
    return SourceResult("velocity", sign, support, omega=omega,
                        omega_support=omega_ok, direction=v, meta=meta)
