"""Phase B pipeline — fit the drift + field in latent space and write to AnnData.

This module is the single entry point that produces the AnnData state the
opspace/plot layers consume. In Phase C it is extended with the NB rate
model (gene-space Jacobians, Fisher ledger, archetype decomposition). For
now it stores a dense latent-space Jacobian in ``adata.uns["scjdo2"]["J"]``.

Written keys:
    adata.obsm["X_drift_v"]       — drift velocities v_c (n, d)
    adata.uns["scjdo2"]["J"]      — dense Jacobian field (n, d, d)
    adata.uns["scjdo2"]["meta"]   — {clock_calibrated, use_rep, time_key, …}
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from anndata import AnnData

from .drift import DriftAdapter, DriftConfig
from .field import FieldConfig, JacobianField


def fit_latent_field(
    adata: AnnData,
    use_rep: str = "X_pca",
    time_key: str = "pseudotime",
    k: int = 40,
    drift_config: Optional[DriftConfig] = None,
    field_config: Optional[FieldConfig] = None,
    clock_scale: Optional[np.ndarray] = None,
    verbose: bool = False,
) -> AnnData:
    """Fit Steps 2 and 5 on an AnnData and store results.

    Reuses the velocity-prior-anchored :class:`DriftAdapter` from Phase B
    first pass; the Jacobian field is estimated by kNN-ball ridge
    regression in the latent space given by ``use_rep``.
    """
    if use_rep not in adata.obsm:
        raise KeyError(f"adata.obsm[{use_rep!r}] not found")
    if time_key not in adata.obs:
        raise KeyError(f"adata.obs[{time_key!r}] not found")

    z = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    t = adata.obs[time_key].to_numpy(dtype=np.float32)
    d = z.shape[1]

    drift_config = drift_config or DriftConfig()
    drift = DriftAdapter(dim=d, config=drift_config).fit(
        z, t=t, clock_scale=clock_scale, verbose=verbose,
    )
    v = drift.velocity(z, t)

    field_config = field_config or FieldConfig(k=k)
    jfield = JacobianField(config=field_config).fit(z, v)
    J = jfield.J_dense()

    adata.obsm["X_drift_v"] = v
    adata.uns.setdefault("scjdo2", {})
    adata.uns["scjdo2"]["J"] = J
    adata.uns["scjdo2"]["meta"] = {
        "clock_calibrated": bool(drift.config.clock_calibrated),
        "use_rep": use_rep,
        "time_key": time_key,
        "k": int(getattr(jfield, "_k", k)),
        "drift": {
            "hidden": drift_config.hidden,
            "depth": drift_config.depth,
            "n_epochs": drift_config.n_epochs,
            "seed": drift_config.seed,
            "use_spectral_norm": drift_config.use_spectral_norm,
            "vel_scale": drift_config.vel_scale,
            "sigma": drift_config.sigma,
        },
        "field": {
            "k": field_config.k,
            "ridge": field_config.ridge,
            "variation_threshold": field_config.variation_threshold,
        },
    }
    adata.uns["scjdo2"]["field_reason_summary"] = jfield.reason_summary()
    return adata
