"""Shared helper: fit ESM energy drift on any latent + per-cell basin
classification using the §5.5 rule (tangent-restricted,
``n_neg ≥ r − 1`` is a basin with up to one soft direction).

Usage (programmatic): pass in an AnnData with a latent key, a cluster
column, and an optional set of "resting" / "terminal" labels; get back
behaviour counts and per-cluster stats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from scjdo2.energy_drift import EnergyDriftAdapter, EnergyDriftConfig
from scjdo2.opspace import classify_basin_latent


def fit_energy_drift(z: np.ndarray, n_epochs: int = 1500,
                     seed: int = 0, beta: float = 1.0,
                     lr: float = 5e-4) -> EnergyDriftAdapter:
    d = z.shape[1]
    return EnergyDriftAdapter(dim=d, config=EnergyDriftConfig(
        hidden=96, depth=3, n_epochs=n_epochs, seed=seed,
        batch_size=256, loss="esm", lr=lr, beta=beta,
        esm_laplacian="exact" if d <= 20 else "hutchinson",
    )).fit(z, verbose=False)


def classify(adata: AnnData, use_rep: str, drift: EnergyDriftAdapter,
             oracle_tol: Optional[float] = None,
             resting_types: Iterable[str] = ()) -> dict:
    """Compute the latent J, run ``classify_basin_latent`` with the
    oracle tol (if given) or a sensible default. Returns a dict with
    the behaviour-by-cluster table and the symmetry check."""
    z = np.asarray(adata.obsm[use_rep], dtype=np.float32)
    J = drift.jacobian_autodiff(z)
    sym = drift.symmetry_check(z, n_cells=min(100, z.shape[0]))
    adata.uns.setdefault("scjdo2", {})["J_latent"] = J
    adata.uns["scjdo2"]["J"] = J

    if oracle_tol is None:
        oracle_tol = _tol_from_cluster_gaussians(z, adata, resting_types)
    classify_basin_latent(adata, tol=float(oracle_tol), use_rep=use_rep)

    beh = adata.obs["op_behaviour_basin"].astype(str).to_numpy()
    cluster = adata.obs["cluster"].astype(str).to_numpy()
    rows = []
    for cl in sorted(set(cluster)):
        m = cluster == cl
        c = pd.Series(beh[m]).value_counts()
        rec = {
            "cluster": cl, "n": int(m.sum()),
            "is_resting": cl in set(resting_types),
        }
        for k in ("holder", "fork", "drift"):
            rec[k] = int(c.get(k, 0))
        rec["frac_holder"] = rec["holder"] / rec["n"] if rec["n"] else 0.0
        rows.append(rec)
    return {
        "behaviour": pd.DataFrame(rows),
        "oracle_tol": float(oracle_tol),
        "symmetry": sym,
    }


def _tol_from_cluster_gaussians(z: np.ndarray, adata: AnnData,
                                 resting_types: Iterable[str]) -> float:
    """``0.1 × median |λ_oracle|`` across clusters (resting if any)."""
    cluster = adata.obs["cluster"].astype(str).to_numpy()
    pool = set(resting_types) or set(cluster)
    leads = []
    for cl in sorted(set(cluster)):
        if cl not in pool:
            continue
        m = cluster == cl
        if m.sum() < 10:
            continue
        Sigma = np.cov(z[m], rowvar=False, bias=False)
        Sigma = Sigma + 1e-4 * np.eye(Sigma.shape[0])
        ev = np.linalg.eigvalsh(-np.linalg.inv(Sigma))
        leads.append(abs(float(ev.max())))
    return 0.1 * float(np.median(leads)) if leads else 0.1
