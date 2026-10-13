"""Step 6 — Fisher confidence ledger.

Per-entry diagonal Fisher info from the Step 5 ridge likelihood (per
``docs/opspace_figure_spec.md`` §8.4): for the per-gene weighted ridge at
cell ``c``, the Fisher info for the ``(g, j_live)`` entry of ``J_c`` is
the ``(j, j)`` diagonal of the ridge Hessian::

    F_{c, g, j_live}  =  Σ_i w_{i,g} · X_{i, j}²  +  λ

where ``X = Δη[:, live_cols[c]]`` and ``w_{i,g} = μ_{i,g} θ_g /
(μ_{i,g} + θ_g)`` is the NB precision of η. This is computed inside
:func:`scjdo2.field.fit_gene_space_jacobians` and surfaced here as a
tidy :class:`FisherLedger` so downstream code (opspace spectrum,
drivers, confidence) can read the ledger without touching the fit code.

Reason codes carry the layout from Step 5 (``ok`` on live entries,
``no_variation`` on genes with sub-threshold η-variance across the
neighbourhood, ``no_counts`` reserved for an all-zero neighbourhood,
``not_in_ball`` reserved for radius-based fits).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.sparse import csr_matrix


REASON_CODES = ("ok", "no_counts", "no_variation", "not_in_ball", "capped")


@dataclass
class FisherLedger:
    """Per-cell, per-entry Fisher info, reason codes, and tangent projectors.

    ``fisher`` and ``reason_per_cell`` share the sparsity layout of the
    gene-space Jacobians (same live-column index per cell). ``thresholds``
    records what defined ``ok`` vs ``no_variation`` etc., so the ledger
    is self-describing.

    ``off_tangent`` (optional, from
    :func:`scjdo2.field.fit_jacobian_tangent_projectors`) carries the
    decoder-Jacobian tangent restriction (docs/opspace_figure_spec.md §9
    fix (c)):

        * ``off_tangent["rank"]`` — ``(n,)`` int: per-cell tangent rank
          ``r_c`` (equal to the latent dim ``d`` on the Jacobian-SVD
          variant).
        * ``off_tangent["projector"]`` — list of ``(r_c, G)`` arrays:
          the orthonormal basis of ``image(∇η(z_c))`` (top-r left
          singular directions of the decoder Jacobian).
        * ``off_tangent["eigvals"]`` — list of ``(d,)`` descending
          singular-value squares (diagnostic).
        * ``off_tangent["method"]`` — ``"jacobian_svd"``.

    If ``off_tangent`` is None, consumers (e.g. :func:`opspace.spectrum`)
    treat the full gene-space as live (no tangent restriction).
    """

    fisher: list[csr_matrix]                   # same shape/sparsity as J
    reason_per_cell: list[np.ndarray]          # list of (G,) int8 per cell
    thresholds: dict = field(default_factory=dict)
    ridge_lambda: Optional[float] = None
    off_tangent: Optional[dict] = None

    # ----------------------------------------------------- convenience
    @property
    def n_cells(self) -> int:
        return len(self.fisher)

    def confidence(self, floor: float = 0.0) -> list[csr_matrix]:
        """Per-entry weight ``sqrt(fisher)``, floored at ``floor``.

        Downstream opspace functions use this to down-weight unreliable
        entries before taking spectra or ranking drivers.
        """
        out: list[csr_matrix] = []
        for F in self.fisher:
            d = np.sqrt(np.clip(F.data, 0.0, None))
            if floor > 0.0:
                d = np.where(d < floor, 0.0, d)
            out.append(csr_matrix((d, F.indices, F.indptr), shape=F.shape))
        return out

    def summary(self) -> dict:
        """Fraction of entries by reason code, aggregated across cells."""
        if not self.reason_per_cell:
            return {name: 0.0 for name in REASON_CODES}
        n = len(self.reason_per_cell)
        counts = np.zeros(len(REASON_CODES), dtype=np.int64)
        total = 0
        for r in self.reason_per_cell:
            r = np.asarray(r)
            total += r.size
            for code, _ in enumerate(REASON_CODES):
                counts[code] += int(np.sum(r == code))
        return {
            name: float(counts[i] / total) if total else 0.0
            for i, name in enumerate(REASON_CODES)
        }

    # ------------------------------------------- Step 2.2 op_confidence
    def op_confidence_from_eigvec(
        self,
        eigvecs: np.ndarray,
        rel_mag_cutoff: float = 0.05,
    ) -> np.ndarray:
        """Scalar per cell: **median Fisher info** over entries that
        contribute to the leading eigenvector (``|v_g| > rel_mag_cutoff ·
        max(|v|)``). This is the ``op_confidence`` used by every plot
        (spec §2.2).

        ``eigvecs`` has shape ``(n_cells, G)`` — the leading eigvec per
        cell. We walk the Fisher ledger's live entries at each cell and
        pull the diagonal entries matching the significant gene mask.
        """
        n = self.n_cells
        if eigvecs.shape[0] != n:
            raise ValueError(
                f"eigvecs n={eigvecs.shape[0]} must match ledger n={n}"
            )
        out = np.full(n, np.nan, dtype=np.float64)
        for c in range(n):
            v = np.abs(np.asarray(eigvecs[c]))
            vmax = float(v.max()) if v.size else 0.0
            if vmax <= 0:
                continue
            keep = v > rel_mag_cutoff * vmax                   # (G,) bool
            if not np.any(keep):
                continue
            F_c = self.fisher[c]
            # Fisher entries sit on (row, col) with row the "to" gene and
            # col the "from" live gene. We take the diagonal subset at
            # genes in `keep`: that is, entries (g, g) for g in keep.
            # In our storage columns are live_cols; the (g, g) diagonal
            # exists only when g is live too.
            diag_vals = F_c.diagonal()
            diag_keep = diag_vals[keep]
            diag_keep = diag_keep[diag_keep > 0]
            if diag_keep.size:
                out[c] = float(np.median(diag_keep))
        return out


def compute(
    step5_output: dict,
    thresholds: Optional[dict] = None,
    tangent: Optional[dict] = None,
) -> FisherLedger:
    """Wrap a Step 5 output dict (from
    :func:`scjdo2.field.fit_gene_space_jacobians`) into a
    :class:`FisherLedger`. The Fisher info itself is computed inside
    Step 5 (same ridge likelihood); this is just the surface API.

    ``tangent`` is the optional output of
    :func:`scjdo2.field.fit_jacobian_tangent_projectors` — attached to
    the ledger as ``off_tangent`` so :func:`scjdo2.opspace.spectrum` can
    restrict eigendecomposition to the decoder-Jacobian tangent at each
    cell.
    """
    required = ("F", "reason_per_cell", "ridge_lambda")
    for key in required:
        if key not in step5_output:
            raise KeyError(f"step5_output missing {key!r}")
    thr = dict(thresholds or {})
    thr.setdefault("variation_threshold", step5_output.get("variation_threshold"))
    thr.setdefault("k", step5_output.get("k"))
    return FisherLedger(
        fisher=list(step5_output["F"]),
        reason_per_cell=list(step5_output["reason_per_cell"]),
        thresholds=thr,
        ridge_lambda=float(step5_output["ridge_lambda"]),
        off_tangent=dict(tangent) if tangent is not None else None,
    )
