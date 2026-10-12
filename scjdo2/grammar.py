"""Step 9 — minimal grammar scaffold.

Archetype sequences along the drift; pairwise commutator norms between
consecutive archetype operators. Scaffolding for a later paper; keep lean.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .archetypes import ArchetypeResult


@dataclass
class GrammarResult:
    sequence: list[int]              # archetype id per step along the drift
    commutators: np.ndarray          # (K, K) Frobenius norms of [A_i, A_j]


def compute(archetypes: ArchetypeResult, order: np.ndarray) -> GrammarResult:
    """``order`` is a per-cell ordering along the drift (e.g. integrated flow)."""
    raise NotImplementedError("Phase D: grammar scaffold")
