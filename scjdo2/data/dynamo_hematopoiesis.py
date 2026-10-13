"""Dynamo hematopoiesis scNT-seq loader (Qiu et al. 2022).

Reads the processed ``hematopoiesis.h5ad`` produced by
``dynamo.sample_data.hematopoiesis()`` from a local cache dir. The
processed file is symbol-indexed (var_names are gene symbols,
``GATA1``, ``SPI1`` present), carries ``X_total`` and ``X_new`` count
layers, ``X_pca`` + ``velocity_pca`` from dynamo's dynamical model, and
per-gene time_5_alpha / time_5_gamma / time_5_half_life columns in
``var`` — the measured labelled-rate parameters for the 5 h pulse.

Returned AnnData has:
  * ``.X`` — raw total UMI counts (int32 CSR), from ``layers["X_total"]``.
  * ``.obsm['X_pca']`` — dynamo's PCA (truncated to the chosen ``n_pcs``).
  * ``.obsm['velocity_pca']`` — dynamo's per-cell measured velocity
    (per-hour units, same truncation).
  * ``.obs['day']``, ``.obs['cell_type']``, ``.obs['pulse_duration']``.
  * ``.uns['labelling'] = {'alpha': α_g (per hour), 'gamma': γ_g (per
    hour), 'pulse_duration': 5.0, 'clock_calibrated': True}``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


_REPO = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = _REPO / "benchmarks" / "data" / "dynamo_hematopoiesis" / "data"


def _coerce_counts(adata, prefer_layer: str) -> sp.csr_matrix:
    """Pull integer total UMIs out of the processed adata.

    Preference order: layers[prefer_layer] → layers['total'] → .X.
    We cast to int32 CSR; counts are stored as float32 in the dynamo
    h5ad but every entry is already integer (verified at probe time).
    """
    for name in (prefer_layer, "total", "X_total"):
        if name in adata.layers:
            X = adata.layers[name]
            break
    else:
        X = adata.X
    if sp.issparse(X):
        X = X.tocsr().astype(np.int32)
    else:
        X = sp.csr_matrix(np.asarray(X).astype(np.int32))
    return X


def load_dynamo_hematopoiesis(
    cache_dir: Optional[str | Path] = None,
    *,
    n_pcs: int = 10,
) -> AnnData:
    """Return the processed Dynamo hematopoiesis scNT-seq adata with
    measured rates and velocities, ready for Phase D §7."""
    cache = Path(cache_dir) if cache_dir else DEFAULT_CACHE
    path = cache / "hematopoiesis.h5ad"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run "
            "dynamo.sample_data.hematopoiesis() to download it, or "
            "point cache_dir at an existing copy."
        )
    import anndata as ad
    a = ad.read_h5ad(path)

    # Raw total UMIs → .X (int32 CSR).
    counts = _coerce_counts(a, prefer_layer="X_total")
    new_counts = (
        a.layers["X_new"].tocsr().astype(np.int32)
        if "X_new" in a.layers and sp.issparse(a.layers["X_new"])
        else sp.csr_matrix(np.asarray(a.layers.get("X_new", counts.toarray() * 0)).astype(np.int32))
        if "X_new" in a.layers
        else None
    )

    obs = a.obs.copy()
    obs.rename(columns={"time": "pulse_duration"}, inplace=True)
    obs["day"] = obs["pulse_duration"]        # dynamo's time is pulse duration (h)
    obs["pulse_duration"] = obs["pulse_duration"].astype(float)

    # Trim PCA + velocity to the requested dim.
    npcs = int(min(n_pcs, a.obsm["X_pca"].shape[1]))
    X_pca = np.asarray(a.obsm["X_pca"])[:, :npcs]
    v_pca = np.asarray(a.obsm["velocity_pca"])[:, :npcs]

    out = AnnData(
        X=counts,
        obs=obs.reset_index(drop=True),
        var=pd.DataFrame({"gene": a.var_names}, index=a.var_names),
    )
    out.var_names = a.var_names
    out.obsm["X_pca"] = X_pca.astype(np.float32)
    out.obsm["velocity_pca"] = v_pca.astype(np.float32)
    if new_counts is not None:
        out.layers["X_new"] = new_counts
    out.layers["X_total"] = counts

    # Measured per-gene rates (time_5 pulse — 5 h one-shot label).
    alpha = pd.to_numeric(a.var.get("time_5_alpha"), errors="coerce").to_numpy()
    gamma = pd.to_numeric(a.var.get("time_5_gamma"), errors="coerce").to_numpy()
    half_life = pd.to_numeric(a.var.get("time_5_half_life"), errors="coerce").to_numpy()
    out.uns["labelling"] = {
        "alpha": alpha.astype(np.float64),
        "gamma": gamma.astype(np.float64),
        "half_life": half_life.astype(np.float64),
        "pulse_duration": 5.0,
        "clock_calibrated": True,
        "source": "dynamo processed hematopoiesis.h5ad (Qiu 2022)",
    }
    out.uns["dataset"] = {
        "name": "Dynamo hematopoiesis scNT-seq",
        "source": "Qiu et al. 2022 (Cell), processed via dynamo",
        "n_cells": out.n_obs,
        "n_genes": out.n_vars,
    }
    return out
