"""Step 6 — Fisher confidence ledger.

Per-entry Fisher information (diagonal) from the **same** NB log-likelihood at
the fitted ``J``. Row-by-row: each gene's row is an independent regression.
Reason codes for empty/low entries: ``no_counts``, ``no_variation``,
``not_in_ball``. Stored alongside ``J`` (same sparsity pattern).

This is the confidence substrate that downstream steps (operator space,
archetypes, drivers) weight by.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix


REASON_CODES = ("ok", "no_counts", "no_variation", "not_in_ball")


@dataclass
class FisherLedger:
    """Per-cell, per-entry Fisher info and reason codes."""

    fisher: list[csr_matrix]        # same shape/sparsity as J; diagonal of Fisher per entry
    reason: list[csr_matrix]        # same shape; integer codes indexing REASON_CODES

    def confidence(self, floor: float = 0.0) -> list[csr_matrix]:
        """Return per-entry confidence weights (``sqrt(fisher)`` by default, floored)."""
        raise NotImplementedError

    def summary(self) -> dict:
        """Fraction of entries by reason code, aggregated across the field."""
        raise NotImplementedError


def compute(J: list[csr_matrix], mu: np.ndarray, dispersion: np.ndarray) -> FisherLedger:
    """Curvature of the NB log-likelihood at the fitted ``J``, diagonal only."""
    raise NotImplementedError("Phase C: diagonal Fisher at fitted J under NB likelihood")
