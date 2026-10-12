"""Step 1 — positional layer (embedding).

Produces latent positions ``z_c`` from the raw count matrix (cells × genes) and
optional batch labels.

Backends:
    - ``"scvi"``   : scVI encoder (default)
    - ``"fa"``     : factor analysis (v1 default)
    - ``"pca"``    : plain PCA

The encoder/loadings are frozen after fitting so new data can be projected in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from anndata import AnnData

Backend = Literal["scvi", "fa", "pca"]


@dataclass
class EmbedConfig:
    backend: Backend = "scvi"
    n_latent: int = 10
    batch_key: Optional[str] = None
    seed: int = 0
    n_epochs: Optional[int] = None  # scVI only


class Embedder:
    """Fit once, project many. Frozen after :meth:`fit`."""

    def __init__(self, config: EmbedConfig) -> None:
        raise NotImplementedError("Phase B: implement scVI/FA/PCA backends")

    def fit(self, adata: AnnData) -> "Embedder":
        raise NotImplementedError

    def transform(self, adata: AnnData) -> np.ndarray:
        """Return ``z`` of shape ``(n_cells, n_latent)``."""
        raise NotImplementedError

    def fit_transform(self, adata: AnnData) -> np.ndarray:
        return self.fit(adata).transform(adata)
