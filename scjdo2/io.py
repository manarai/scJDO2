"""On-disk formats.

AnnData is the primary container. The sparse Jacobian field and Fisher ledger
are stored under ``adata.obsm`` / ``adata.uns`` as sparse arrays, or under a
sidecar zarr store when memory becomes a concern.

Layout:

    adata.obsm["scjdo2_z"]            — latent positions (Step 1)
    adata.obsm["scjdo2_v"]            — drift velocities (Step 2)
    adata.layers["scjdo2_mu"]         — rates μ_g(z_c)        (Step 3)
    adata.layers["scjdo2_dmudt"]      — dμ_g/dt along drift   (Step 4)
    adata.uns["scjdo2"]["J"]          — per-cell sparse Jacobians (Step 5)
    adata.uns["scjdo2"]["fisher"]     — per-entry Fisher ledger   (Step 6)
    adata.uns["scjdo2"]["opspace"]    — invariants + leading eigenvectors (Step 7)
    adata.uns["scjdo2"]["archetypes"] — weights, operators, names, drivers (Step 8)
    adata.uns["scjdo2"]["grammar"]    — sequence + commutators (Step 9)
    adata.uns["scjdo2"]["meta"]       — seed, clock_calibrated, embed backend, k, K
"""

from __future__ import annotations

from pathlib import Path

from anndata import AnnData


def save(adata: AnnData, path: str | Path) -> None:
    """Write the full scJDO2 state to ``path`` (``.h5ad`` or ``.zarr``)."""
    raise NotImplementedError("Phase B+")


def load(path: str | Path) -> AnnData:
    """Load an scJDO2 state back into an AnnData."""
    raise NotImplementedError("Phase B+")
