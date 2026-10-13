"""Setty 2019 Palantir CD34+ bone marrow loader.

The manuscript's haematopoiesis comparison dataset: human CD34+ bone
marrow with Palantir pseudotime, cluster annotations, and spliced /
unspliced layers. Same compartment as the Dynamo scNT-seq
hematopoiesis set, so the two sit side-by-side as snapshot (Setty)
vs measured flow (scNT-seq) on matched biology (R18 clock caveat
applies to both — Setty has no labelling, scNT-seq does).

Dataset shape (as cached under scJDO v1): 5780 cells × 14319 genes,
raw integer UMIs on ``.X``, Palantir pseudotime + cluster labels in
``obs``, spliced / unspliced layers for a scVelo cross-check.

GATA1 and SPI1 both present (human symbols).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData


# Default path (scJDO v1 shipped the pre-processed Setty h5ad under its
# ``data/BoneMarrow/`` tree — we point at that copy by default to keep
# the two repos in sync).
_DEFAULT_CACHES: tuple[Path, ...] = (
    Path("/Users/terooatt/Documents/scJDO/data/BoneMarrow/human_cd34_bone_marrow.h5ad"),
    Path(__file__).resolve().parents[2] / "benchmarks" / "data" / "setty_cd34" / "human_cd34_bone_marrow.h5ad",
)

# Terminal lineages present in the Setty cluster annotation.
TERMINAL_CLUSTERS = {"Ery_1", "Ery_2", "Mono_1", "Mono_2", "DCs", "CLP", "Mega"}
PROGENITOR_CLUSTERS = {"HSC_1", "HSC_2", "Precursors"}


def _resolve_path(cache_dir: Optional[str | Path]) -> Path:
    if cache_dir is not None:
        p = Path(cache_dir)
        if p.is_dir():
            p = p / "human_cd34_bone_marrow.h5ad"
        return p
    for cand in _DEFAULT_CACHES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Setty CD34 bone marrow h5ad not found. Expected one of:\n"
        + "\n".join(f"  {p}" for p in _DEFAULT_CACHES)
        + "\nPass cache_dir=... to override."
    )


def load_setty_cd34(cache_dir: Optional[str | Path] = None) -> AnnData:
    """Return the Setty 2019 CD34+ bone marrow AnnData with harmonised
    obs columns:

      * ``.X``                   raw integer UMI counts (CSR, int32).
      * ``.obs['cluster']``      Setty cluster annotation.
      * ``.obs['pseudotime']``   Palantir pseudotime (float ∈ [0, 1]).
      * ``.obs['is_terminal']``  bool (Ery / Mono / DC / CLP / Mega).
      * ``.obs['is_progenitor']`` bool (HSC_1 / HSC_2 / Precursors).
      * ``.layers['spliced']`` / ``['unspliced']`` (for a scVelo
        cross-check in later arms).
      * ``.uns['dataset']``      source + compartment notes.
    """
    import anndata as ad

    path = _resolve_path(cache_dir)
    a = ad.read_h5ad(path)

    # Coerce .X to int32 CSR if it isn't already.
    X = a.X
    if sp.issparse(X):
        X = X.tocsr().astype(np.int32)
    else:
        X = sp.csr_matrix(np.asarray(X).astype(np.int32))
    out = AnnData(
        X=X,
        obs=a.obs.copy(),
        var=a.var.copy() if a.var is not None else pd.DataFrame(index=a.var_names),
    )
    out.var_names = a.var_names
    for key in ("spliced", "unspliced"):
        if key in a.layers:
            lay = a.layers[key]
            if sp.issparse(lay):
                lay = lay.tocsr().astype(np.int32)
            else:
                lay = sp.csr_matrix(np.asarray(lay).astype(np.int32))
            out.layers[key] = lay
    if "X_tsne" in a.obsm:
        out.obsm["X_tsne"] = np.asarray(a.obsm["X_tsne"]).astype(np.float32)

    # Harmonise obs column names.
    if "clusters" in out.obs:
        out.obs["cluster"] = out.obs["clusters"].astype(str)
    if "palantir_pseudotime" in out.obs:
        out.obs["pseudotime"] = pd.to_numeric(
            out.obs["palantir_pseudotime"], errors="coerce",
        ).astype(float)
    out.obs["is_terminal"] = out.obs.get("cluster", pd.Series(dtype=str)).isin(
        TERMINAL_CLUSTERS,
    )
    out.obs["is_progenitor"] = out.obs.get("cluster", pd.Series(dtype=str)).isin(
        PROGENITOR_CLUSTERS,
    )

    out.uns["dataset"] = {
        "name": "Setty 2019 Palantir CD34+ bone marrow",
        "source": str(path),
        "compartment": "human CD34+ hematopoiesis (same as Dynamo scNT-seq)",
        "snapshot_or_measured_flow": "snapshot",
        "manuscript_pipeline": "v1 Palantir per-branch slicing (preprint)",
    }
    return out
