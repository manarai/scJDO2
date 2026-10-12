"""Step 7 — operator space.

Per-cell operator coordinates in two tiers:

**Invariant tier** (base-independent):
    - eigenvalues sorted by real part
    - counts of stable, unstable, near-zero, complex-pair modes
    - spectral gap
    - non-normality descriptors (reimplemented here from the Koopman literature,
      not imported from v1's Koopman backend):

      Henrici index:    ``‖J‖_F² − Σ |λ_i|²``, often reported as
                        ``sqrt(max(0, ‖J‖_F² − Σ |λ_i|²))``.
      Transient gain:   ``max_t ‖exp(tJ)‖`` (bounded by ``‖V‖·‖V⁻¹‖`` with ``V``
                        the eigenvector matrix of ``J``).
      Eigvec cond.:     ``κ(V)`` as a cheap non-normality proxy.

**Coordinate-dependent tier**: leading eigenvectors (sign-aligned per cell).

All descriptors are Fisher-weighted via :mod:`scjdo2.fisher` when a ledger is
provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix

from .fisher import FisherLedger


@dataclass
class OpInvariants:
    eigenvalues: np.ndarray         # (n_cells, n_eigs), sorted by Re(λ) descending
    n_unstable: np.ndarray          # (n_cells,)
    n_stable: np.ndarray            # (n_cells,)
    n_near_zero: np.ndarray         # (n_cells,)
    n_complex_pairs: np.ndarray     # (n_cells,)
    spectral_gap: np.ndarray        # (n_cells,)
    henrici: np.ndarray             # (n_cells,)
    transient_gain: np.ndarray      # (n_cells,)
    eigvec_cond: np.ndarray         # (n_cells,)


@dataclass
class OpCoordinates:
    invariants: OpInvariants
    leading_eigvecs: np.ndarray     # (n_cells, n_lead, n_genes), coordinate-dependent


def compute(
    J: list[csr_matrix],
    ledger: Optional[FisherLedger] = None,
    n_leading: int = 3,
    near_zero_tol: float = 1e-6,
) -> OpCoordinates:
    """Compute invariants and leading eigenvectors for every cell's operator."""
    raise NotImplementedError("Phase B: eigendecomposition + non-normality descriptors")
