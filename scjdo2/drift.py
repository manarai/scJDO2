"""Step 2 — drift field.

Thin adapter around v1's neural-ODE drift field (``scjdo.models.drift.DriftField``).
The adapter accepts ``(z, [t], [v_ref])`` tensors and returns ``f(z)`` and the
per-cell velocity ``v_c = f(z_c)``. Timescale is relative; the ``clock_calibrated``
flag defaults to ``False`` and is only ``True`` when the caller passes a
per-cell clock scale from metabolic labelling.

v1 is imported lazily to keep ``scjdo2`` importable without the ``[compare]``
extra. If v1's interface turns out to be entangled with pseudotime or bridge
code, the adapter degrades to a minimal in-package reimplementation (TBD in
V2_PLAN.md after Phase A audit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DriftConfig:
    hidden: tuple[int, ...] = (128, 128)
    n_epochs: int = 2000
    lr: float = 1e-3
    seed: int = 0
    clock_calibrated: bool = False  # set True iff caller supplies clock_scale


class DriftField:
    """Learned drift ``f(z)`` on the latent space."""

    def __init__(self, config: DriftConfig) -> None:
        raise NotImplementedError(
            "Phase B: adapt v1 scjdo.models.drift.DriftField or reimplement minimally"
        )

    def fit(
        self,
        z: np.ndarray,
        t: Optional[np.ndarray] = None,
        clock_scale: Optional[np.ndarray] = None,
    ) -> "DriftField":
        """Fit drift on latent positions. ``t`` is optional pseudotime; ``clock_scale``
        turns ``config.clock_calibrated`` on when provided."""
        raise NotImplementedError

    def f(self, z: np.ndarray) -> np.ndarray:
        """Evaluate the drift field at arbitrary latent points."""
        raise NotImplementedError

    def velocity(self, z: np.ndarray) -> np.ndarray:
        """Per-cell velocity, shape ``(n_cells, n_latent)``."""
        return self.f(z)
