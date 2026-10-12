"""Step 8 — operator archetypes.

Thin adapter around v1's semi-NMF with column-wise NNLS. Inputs are
Fisher-weighted operator-space coordinates. Number of archetypes is chosen by
held-out likelihood (never by eye).

Each archetype is named automatically from the eigenvalue signature of its
operator:

    - **holder**            — all Re(λ) ≤ 0, well-conditioned
    - **fork**              — one Re(λ) > 0, rest ≤ 0 (saddle)
    - **drift**             — one Re(λ) ≈ 0, rest ≤ 0
    - **damped oscillator** — complex pair with Re(λ) < 0
    - **growing oscillator**— complex pair with Re(λ) > 0

A **sensitive** flag is set when non-normality (Henrici index, transient gain,
or ``κ(V)``) exceeds the per-dataset threshold reported by Step 7.

Drivers = largest-weight genes in the leading eigenvector(s), filtered by
Fisher confidence, reported with sign and confidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from .fisher import FisherLedger
from .opspace import OpCoordinates


@dataclass
class ArchetypeConfig:
    n_archetypes: int | Literal["auto"] = "auto"
    confidence_floor: float = 0.0
    seed: int = 0


@dataclass
class ArchetypeResult:
    weights: np.ndarray                 # (n_cells, K)
    operators: list                     # list of K archetype operators (dense or sparse)
    names: list[str]                    # auto-named (holder / fork / ...)
    sensitive: np.ndarray               # (K,) bool, non-normality flag
    drivers: list[list[tuple[str, float, float]]]
    # per archetype: [(gene, signed loading, Fisher confidence), ...]


class ArchetypeDecomposer:
    def __init__(self, config: ArchetypeConfig) -> None:
        raise NotImplementedError(
            "Phase B: adapt v1 semi-NMF (archetypes/) or reimplement minimally"
        )

    def fit(
        self,
        coords: OpCoordinates,
        ledger: Optional[FisherLedger] = None,
        gene_names: Optional[list[str]] = None,
    ) -> ArchetypeResult:
        raise NotImplementedError
