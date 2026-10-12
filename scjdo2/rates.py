"""Steps 3 and 4 — gene rate model and rates along the drift.

Step 3 — custom scvi-tools module: fixed latent positions ``z`` as input (no
encoder), smooth decoder (penalised MLP shared across genes with gene-specific
heads), negative binomial likelihood with gene-specific dispersion, size
factors entered as an **offset** (never a division). Must be differentiable
w.r.t. ``z``.

Step 4 — rates along the drift: ``dμ_g/dt = ∇_z μ_g(z_c) · f(z_c)``. Computed
via automatic differentiation of the decoder, not finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RatesConfig:
    hidden: tuple[int, ...] = (128, 128)
    n_epochs: int = 400
    lr: float = 1e-3
    smoothness_weight: float = 1e-3  # penalty on ∇_z μ_g
    seed: int = 0


class GeneRateModel:
    """Smooth NB rate model ``μ_g(z)`` with autodiff gradients w.r.t. ``z``."""

    def __init__(self, config: RatesConfig) -> None:
        raise NotImplementedError("Phase C: scvi-tools NB decoder with shared trunk")

    def fit(
        self,
        counts: np.ndarray,          # (n_cells, n_genes), raw integer counts
        z: np.ndarray,               # (n_cells, n_latent), frozen positions
        size_factors: np.ndarray,    # (n_cells,), used as log-offset
        batch: Optional[np.ndarray] = None,
    ) -> "GeneRateModel":
        raise NotImplementedError

    def mu(self, z: np.ndarray) -> np.ndarray:
        """Return expected counts ``μ_g(z)`` of shape ``(n_cells, n_genes)``."""
        raise NotImplementedError

    def grad_mu(self, z: np.ndarray) -> np.ndarray:
        """Return ``∇_z μ_g(z)`` of shape ``(n_cells, n_genes, n_latent)`` via autodiff."""
        raise NotImplementedError

    def rate_along(
        self,
        z: np.ndarray,
        f: np.ndarray,  # drift field evaluated at z, shape (n_cells, n_latent)
    ) -> np.ndarray:
        """Chain rule: ``dμ_g/dt = ∇_z μ_g · f``. Shape ``(n_cells, n_genes)``."""
        raise NotImplementedError
