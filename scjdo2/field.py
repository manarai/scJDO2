"""Step 5 — neighbourhood Jacobian operator field.

For each cell, take its ``k`` nearest neighbours in latent space; regress the
rate vectors on the level vectors ``d μ / dt ≈ J (μ − μ_0)``, likelihood-weighted
under the NB likelihood from :mod:`scjdo2.rates`. Assign ``J`` to the centre
cell. Columns are restricted to genes that vary within the ball (sparse by
construction). ``k`` is chosen by held-out likelihood.

A smoothness penalty ties ``J`` of adjacent cells together, so the field
varies smoothly across neighbours.

If a cell has fewer than ``k`` neighbours within the configured radius, it is
**marked** (not silently filled). The reason code is forwarded to the Fisher
ledger as ``not_in_ball``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class FieldConfig:
    k: int | Literal["auto"] = "auto"       # held-out likelihood if "auto"
    radius: Optional[float] = None          # None = no radius cap
    variation_threshold: float = 1e-3       # genes below this within-ball variance are dropped
    smoothness_weight: float = 1e-2         # penalty tying J of neighbours together
    seed: int = 0


class JacobianField:
    """Per-cell operator field with a Fisher-ledger hook."""

    def __init__(self, config: FieldConfig) -> None:
        raise NotImplementedError("Phase B/C: likelihood-weighted neighbourhood regression")

    def fit(
        self,
        z: np.ndarray,              # (n_cells, n_latent)
        mu: np.ndarray,             # (n_cells, n_genes), levels from rates.mu
        dmu_dt: np.ndarray,         # (n_cells, n_genes), rates from rates.rate_along
        dispersion: Optional[np.ndarray] = None,  # (n_genes,), for NB weights
    ) -> "JacobianField":
        raise NotImplementedError

    def J(self) -> list[csr_matrix]:
        """Per-cell sparse Jacobians, length ``n_cells``, each ``(n_genes, n_genes)``."""
        raise NotImplementedError

    def reasons(self) -> np.ndarray:
        """Per-cell, per-entry reason codes (object array of dicts or structured arr).

        Values forwarded to :mod:`scjdo2.fisher` as ledger reason codes:
        ``no_counts``, ``no_variation``, ``not_in_ball``.
        """
        raise NotImplementedError
